//! Lightweight wall-clock span recorder for overlap-aware stage attribution of `synth`.
//! Times are seconds since `process_start()` (captured first thing in main). Spans are pushed from
//! the pipeline threads; a few thousand per run, so the overhead is negligible (no device syncs).

use serde::Serialize;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static START: OnceLock<Instant> = OnceLock::new();

/// Process time origin (first call wins; call at the top of main).
pub fn process_start() -> Instant {
    *START.get_or_init(Instant::now)
}

pub fn now() -> f64 {
    process_start().elapsed().as_secs_f64()
}

#[derive(Clone, Debug, Serialize)]
pub struct Span {
    pub pass: usize,
    pub thread: &'static str,
    pub stage: &'static str,
    pub start: f64,
    pub end: f64,
    /// lines (or chunks) covered by the span, when meaningful
    pub items: usize,
}

#[derive(Default)]
pub struct Timeline {
    spans: Mutex<Vec<Span>>,
}

impl Timeline {
    pub fn push(&self, pass: usize, thread: &'static str, stage: &'static str, start: f64, items: usize) {
        let end = now();
        self.spans.lock().unwrap().push(Span { pass, thread, stage, start, end, items });
    }

    pub fn spans(&self) -> Vec<Span> {
        self.spans.lock().unwrap().clone()
    }

    /// Per-stage summary for one pass: busy seconds per (thread, stage), the pass window, and the
    /// union (overlap-merged) busy time of each thread's WORK stages (waits excluded).
    pub fn summary(&self, pass: usize) -> serde_json::Value {
        let spans: Vec<Span> = self.spans().into_iter().filter(|s| s.pass == pass).collect();
        let mut by: std::collections::BTreeMap<String, (f64, usize, usize)> = Default::default();
        for s in &spans {
            let e = by.entry(format!("{}.{}", s.thread, s.stage)).or_default();
            e.0 += s.end - s.start;
            e.1 += 1;
            e.2 += s.items;
        }
        let window = spans.iter().fold((f64::INFINITY, 0f64), |(a, b), s| (a.min(s.start), b.max(s.end)));
        let union = |pred: &dyn Fn(&Span) -> bool| -> f64 {
            let mut iv: Vec<(f64, f64)> = spans.iter().filter(|s| pred(s)).map(|s| (s.start, s.end)).collect();
            iv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (mut tot, mut cur) = (0f64, None::<(f64, f64)>);
            for (a, b) in iv {
                cur = match cur {
                    Some((ca, cb)) if a <= cb => Some((ca, cb.max(b))),
                    Some((ca, cb)) => {
                        tot += cb - ca;
                        Some((a, b))
                    }
                    None => Some((a, b)),
                };
            }
            tot + cur.map(|(a, b)| b - a).unwrap_or(0.0)
        };
        let work = |t: &'static str| move |s: &Span| s.thread == t && !s.stage.ends_with("wait");
        serde_json::json!({
            "window_s": [window.0, window.1],
            "stages": by.iter().map(|(k, v)| (k.clone(), serde_json::json!({"busy_s": v.0, "spans": v.1, "items": v.2}))).collect::<serde_json::Map<_, _>>(),
            "thread_work_union_s": {
                "prepare": union(&work("prepare")),
                "gpu": union(&work("gpu")),
                "writer": union(&work("writer")),
                "any_pipeline_work": union(&|s: &Span| s.thread != "main" && !s.stage.ends_with("wait")),
            },
        })
    }
}
