//! Permanent in-engine stage timing, enabled with KOKORO_PROFILE=1 (zero work otherwise).

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("KOKORO_PROFILE").map(|v| v == "1").unwrap_or(false))
}

thread_local! {
    static ACC: RefCell<Vec<(&'static str, f64, u64)>> = const { RefCell::new(Vec::new()) };
}

pub struct Scope {
    name: &'static str,
    t0: Option<Instant>,
}

pub fn scope(name: &'static str) -> Scope {
    Scope { name, t0: if enabled() { Some(Instant::now()) } else { None } }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(t0) = self.t0 {
            let dt = t0.elapsed().as_secs_f64();
            ACC.with(|a| {
                let mut a = a.borrow_mut();
                match a.iter_mut().find(|e| e.0 == self.name) {
                    Some(e) => {
                        e.1 += dt;
                        e.2 += 1;
                    }
                    None => a.push((self.name, dt, 1)),
                }
            });
        }
    }
}

/// Drain and format the accumulated stage table (empty string when profiling is off).
pub fn report() -> String {
    if !enabled() {
        return String::new();
    }
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let total: f64 = a.iter().filter(|e| !e.0.contains('/')).map(|e| e.1).sum();
        let mut s = String::from("stage                          total_s    calls   share\n");
        for (n, t, c) in a.iter() {
            s += &format!("{n:<30} {t:>8.3} {c:>8} {:>6.1}%\n", 100.0 * t / total.max(1e-12));
        }
        a.clear();
        s
    })
}
