//! Dev probe: wall time of each model-load step (cold process).
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let t0 = Instant::now();
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("model dir"));
    let lap = |t: &mut Instant, what: &str| {
        eprintln!("{what:<28} {:>7.3} s", t.elapsed().as_secs_f64());
        *t = Instant::now();
    };
    let mut t = Instant::now();
    let raw = std::fs::read(dir.join("kokoro-v1_0.pth"))?;
    lap(&mut t, "fs::read .pth only");
    drop(raw);
    let _ = kokoro::torchpt::load_kmodel_checkpoint(&dir.join("kokoro-v1_0.pth"))?;
    lap(&mut t, "torchpt parse (incl read)");
    let w = kokoro::weights::Weights::load_pth(&dir.join("kokoro-v1_0.pth"))?;
    lap(&mut t, "load_pth (read+parse)");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let m = kokoro::model::Kokoro::from_weights(&w, &cfg)?;
    lap(&mut t, "Kokoro::from_weights (cpu)");
    let _h = kokoro::engine::sha256_file(&dir.join("kokoro-v1_0.pth"))?;
    lap(&mut t, "sha256 weights (serial)");
    #[cfg(feature = "cuda")]
    {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        lap(&mut t, "CUDA context");
        let _m = ctx.load_module(cudarc::nvrtc::Ptx::from_src(kokoro::gpu::PTX_SRC))?;
        lap(&mut t, "load PTX module (JIT)");
        let _m2 = ctx.load_module(cudarc::nvrtc::Ptx::from_src(kokoro::gpu::PTX_SRC))?;
        lap(&mut t, "load PTX module again (JIT cache)");
        let s = ctx.default_stream();
        let _b = cudarc::cublas::CudaBlas::new(s.clone())?;
        lap(&mut t, "cuBLAS handle");
        drop(ctx);
        let g = kokoro::gpu::GpuKokoro::new(&m, 0)?;
        lap(&mut t, "GpuKokoro::new (ctx+PTX+upload)");
        drop(g);
        lap(&mut t, "drop GpuKokoro");
    }
    eprintln!("total {:.3} s", t0.elapsed().as_secs_f64());
    Ok(())
}
