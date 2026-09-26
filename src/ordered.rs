//! Ordered parallel map with bounded look-ahead (the `synth` prepare stage).
//!
//! `threads` workers compute `f(i)` for i in 0..n; results reach `sink` strictly in index order.
//! A worker may only start item i once i < emitted + window, so in-flight + buffered work never
//! exceeds `window` items (a slow early item cannot let the reorder buffer grow with the input).
//! Any exit of the sequencer — done, `f` error, or `sink` returning false (downstream closed) —
//! sets a stop flag and wakes every waiting worker; no further items are started.

use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex};

/// Returns the number of items emitted (== n unless `sink` stopped early). An error from `f(i)` is
/// returned after items 0..i have been emitted, and the workers have stopped.
pub fn ordered_parallel_map<T, F, S>(n: usize, threads: usize, window: usize, f: F, mut sink: S) -> Result<usize>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync,
    S: FnMut(usize, T) -> bool,
{
    let window = window.max(1);
    let next = AtomicUsize::new(0);
    let emitted = Mutex::new(0usize);
    let cv = Condvar::new();
    let stop = AtomicBool::new(false);
    struct StopOnExit<'a>(&'a AtomicBool, &'a Mutex<usize>, &'a Condvar);
    impl Drop for StopOnExit<'_> {
        fn drop(&mut self) {
            self.0.store(true, SeqCst);
            let _g = self.1.lock().unwrap_or_else(|e| e.into_inner());
            self.2.notify_all();
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<T>)>();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            let tx = tx.clone();
            let (next, emitted, cv, stop, f) = (&next, &emitted, &cv, &stop, &f);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, SeqCst);
                if i >= n {
                    break;
                }
                {
                    let mut e = emitted.lock().unwrap_or_else(|e| e.into_inner());
                    while i >= *e + window && !stop.load(SeqCst) {
                        e = cv.wait(e).unwrap_or_else(|e| e.into_inner());
                    }
                }
                if stop.load(SeqCst) {
                    break;
                }
                let r = f(i);
                let fail = r.is_err();
                if tx.send((i, r)).is_err() || fail {
                    break;
                }
            });
        }
        drop(tx);
        let _stop = StopOnExit(&stop, &emitted, &cv);
        let mut pending = BTreeMap::new();
        let mut want = 0usize;
        for (i, r) in rx {
            // errors are buffered like values and raised at their ORDERED position: every item
            // before a failing one is still emitted (deterministic, independent of thread timing)
            pending.insert(i, r);
            debug_assert!(pending.len() <= window);
            while let Some(v) = pending.remove(&want) {
                if !sink(want, v?) {
                    return Ok(want);
                }
                want += 1;
                *emitted.lock().unwrap_or_else(|e| e.into_inner()) = want;
                cv.notify_all();
            }
        }
        Ok(want)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Delayed first item: later items must NOT run ahead beyond the window while item 0 is slow;
    /// emission stays in order; everything is emitted.
    #[test]
    fn delayed_first_item_respects_window_and_order() {
        let (n, window) = (200, 8);
        let started = AtomicUsize::new(0);
        let max_started_while_0_pending = AtomicUsize::new(0);
        let done0 = AtomicBool::new(false);
        let mut order = vec![];
        let got = ordered_parallel_map(
            n,
            16,
            window,
            |i| {
                started.fetch_add(1, SeqCst);
                if i == 0 {
                    std::thread::sleep(Duration::from_millis(300));
                    done0.store(true, SeqCst);
                } else if !done0.load(SeqCst) {
                    max_started_while_0_pending.fetch_max(i, SeqCst);
                }
                Ok(i * 10)
            },
            |i, v| {
                order.push((i, v));
                true
            },
        )
        .unwrap();
        assert_eq!(got, n);
        assert_eq!(order, (0..n).map(|i| (i, i * 10)).collect::<Vec<_>>());
        assert!(max_started_while_0_pending.load(SeqCst) < window, "worker ran ahead to item {} (window {window})", max_started_while_0_pending.load(SeqCst));
        assert_eq!(started.load(SeqCst), n);
    }

    /// Downstream cancellation: sink stops after 5 items; no more than `window` further items are
    /// ever started, and the call returns promptly.
    #[test]
    fn sink_cancellation_stops_workers() {
        let started = AtomicUsize::new(0);
        let got = ordered_parallel_map(
            10_000,
            8,
            16,
            |i| {
                started.fetch_add(1, SeqCst);
                Ok(i)
            },
            |i, _| i < 5,
        )
        .unwrap();
        assert_eq!(got, 5);
        assert!(started.load(SeqCst) <= 5 + 1 + 16 + 8, "started {} items after cancellation", started.load(SeqCst));
    }

    /// An error in f is returned; items before it were emitted in order; the rest are not started
    /// beyond the window.
    #[test]
    fn error_is_propagated_and_stops_workers() {
        let started = AtomicUsize::new(0);
        let mut seen = vec![];
        let r = ordered_parallel_map(
            10_000,
            8,
            16,
            |i| {
                started.fetch_add(1, SeqCst);
                if i == 50 {
                    anyhow::bail!("boom at {i}")
                }
                Ok(i)
            },
            |i, _| {
                seen.push(i);
                true
            },
        );
        assert!(format!("{:#}", r.unwrap_err()).contains("boom at 50"));
        assert_eq!(seen, (0..50).collect::<Vec<_>>());
        assert!(started.load(SeqCst) <= 50 + 16 + 8 + 1, "started {}", started.load(SeqCst));
    }

    /// Negative control for the probe above: with an effectively unbounded window, workers DO run
    /// far ahead of the slow first item (so the bounded-window assertion is not vacuous).
    #[test]
    fn negative_control_unbounded_window_runs_ahead() {
        let done0 = AtomicBool::new(false);
        let ahead = AtomicUsize::new(0);
        ordered_parallel_map(200, 16, 1_000_000, |i| {
            if i == 0 {
                std::thread::sleep(Duration::from_millis(300));
                done0.store(true, SeqCst);
            } else if !done0.load(SeqCst) {
                ahead.fetch_max(i, SeqCst);
            }
            Ok(i)
        }, |_, _| true)
        .unwrap();
        assert!(ahead.load(SeqCst) > 100, "probe failed to observe run-ahead ({})", ahead.load(SeqCst));
    }

    #[test]
    fn empty_and_single_thread() {
        assert_eq!(ordered_parallel_map(0, 4, 4, |i| Ok(i), |_, _| true).unwrap(), 0);
        let mut v = vec![];
        ordered_parallel_map(37, 1, 1, |i| Ok(i), |_, x| {
            v.push(x);
            true
        })
        .unwrap();
        assert_eq!(v, (0..37).collect::<Vec<_>>());
    }
}
