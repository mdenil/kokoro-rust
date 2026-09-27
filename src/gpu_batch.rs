//! B1: batched CUDA forward over a ragged, gap-separated layout (docs/design/BATCHING.md).
//! Child module of `gpu`: reuses the same hydrated weights, GEMM helper and kernels. Per-item
//! semantics are preserved by construction: conv inputs have zeroed gaps (exact per-item zero
//! padding), normalization statistics and style conditioning are per item, LSTMs and attention never
//! mix items, and every item has its own noise stream.

use super::*;
use crate::vocoder::RngNoise;
use cudarc::driver::CudaSlice;

/// Per-item excitation noise: a counter-based stream keyed by seed (product path) or replayed
/// fixture draws (parity tests).
pub enum ItemNoise {
    Counter(u64),
    Fixed { rand_ini: [f32; HARMONICS], sine: Vec<f32> },
}

pub struct BatchItem<'a> {
    pub ids: &'a [i64],
    pub ref_s: &'a [f32],
    pub speed: f32,
    pub noise: ItemNoise,
}

/// Minimum gap (in N-frame units) between items; see the halo analysis in BATCHING.md.
pub const GAP_N: usize = 2;
/// Token-layout gap (text-encoder k=5 conv halo).
pub const GAP_T: usize = 2;

/// One time domain of the layout: buffer width `l`, item spans, and a per-column item table.
struct Dom {
    l: usize,
    start: Vec<usize>,
    len: Vec<usize>,
    col_item: CudaSlice<i32>,
    start_d: CudaSlice<i32>,
    len_d: CudaSlice<i32>,
}

impl Dom {
    fn new(g: &Gpu, l: usize, start: Vec<usize>, len: Vec<usize>) -> Result<Self> {
        let mut col = vec![-1i32; l];
        for (b, (&s, &n)) in start.iter().zip(&len).enumerate() {
            ensure!(s + n <= l, "layout overflow");
            for c in &mut col[s..s + n] {
                ensure!(*c < 0, "overlapping item spans");
                *c = b as i32;
            }
        }
        let si: Vec<i32> = start.iter().map(|&v| v as i32).collect();
        let li: Vec<i32> = len.iter().map(|&v| v as i32).collect();
        Ok(Self { l, col_item: g.stream.clone_htod(&col)?, start_d: g.stream.clone_htod(&si)?, len_d: g.stream.clone_htod(&li)?, start, len })
    }

    fn b(&self) -> usize {
        self.start.len()
    }
}

static STATS_1PASS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("KOKORO_STATS_1PASS").map(|v| v != "0").unwrap_or(true));
static FUSE_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("KOKORO_FUSE_RES_CONV").map(|v| v != "0").unwrap_or(true));

impl GpuKokoro {
    fn mask(&self, x: &mut Buf, c: usize, d: &Dom) -> Result<()> {
        // negative-control hook for tests (proves the batch tests detect padding contamination)
        if std::env::var("KOKORO_BATCH_NEGCTL_NOMASK").is_ok() {
            return Ok(());
        }
        let g = &self.gpu;
        let (ci, li) = (c as i32, d.l as i32);
        launch!(g, mask_gaps, cfg1(c * d.l), x, &d.col_item, &ci, &li)
    }

    /// stride-1 conv on a domain (input gaps zeroed first); output keeps the same layout.
    fn conv_b(&self, conv: &GConv, x: &mut Buf, d: &Dom) -> Result<Buf> {
        ensure!(conv.stride == 1 && conv.pad * 2 == conv.dil * (conv.k - 1), "conv_b: layout-preserving convs only");
        self.mask(x, conv.cin, d)?;
        let (y, t) = conv.fwd(&self.gpu, x, d.l)?;
        ensure!(t == d.l, "conv_b changed length");
        Ok(y)
    }

    /// AdaIN statistics: style projection gb [B, 2C] and per-item channel mean / rstd.
    fn adain_stats(&self, a: &GAdaIn, x: &Buf, d: &Dom, styles: &Buf) -> Result<(Buf, Buf, Buf)> {
        let g = &self.gpu;
        let gb = a.fc.fwd(g, styles, d.b())?;
        let mut mean = g.alloc(d.b() * a.c)?;
        let mut rstd = g.alloc(d.b() * a.c)?;
        let (li, eps, ci) = (d.l as i32, 1e-5f32, a.c as i32);
        let cfg = LaunchConfig { grid_dim: (a.c as u32, d.b() as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        if *STATS_1PASS {
            // LEVER PL-014 (kill switch KOKORO_STATS_1PASS=0): single-pass statistics
            launch!(g, chan_stats_seg1, cfg, x, &li, &d.start_d, &d.len_d, &eps, &mut mean, &mut rstd, &ci)?;
        } else {
            launch!(g, chan_stats_seg, cfg, x, &li, &d.start_d, &d.len_d, &eps, &mut mean, &mut rstd, &ci)?;
        }
        Ok((gb, mean, rstd))
    }

    /// AdaIN with per-item statistics and per-item style (gb [B, 2C]); gaps -> 0.
    fn adain_b(&self, a: &GAdaIn, x: &Buf, d: &Dom, styles: &Buf, act: Act) -> Result<Buf> {
        let g = &self.gpu;
        let (gb, mean, rstd) = self.adain_stats(a, x, d, styles)?;
        let (li, ci) = (d.l as i32, a.c as i32);
        let mut y = g.alloc(a.c * d.l)?;
        let null = 0u64;
        match act {
            Act::Leaky(sl) => launch!(g, adain_apply_seg, cfg1(a.c * d.l), x, &mut y, &d.col_item, &mean, &rstd, &a.nw, &a.nb, &gb, &null, &ci, &li, &1i32, &sl)?,
            Act::Snake(al) => launch!(g, adain_apply_seg, cfg1(a.c * d.l), x, &mut y, &d.col_item, &mean, &rstd, &a.nw, &a.nb, &gb, al, &ci, &li, &2i32, &0.0f32)?,
        }
        Ok(y)
    }

    /// AdainResBlk1d on domain `d` (output domain `d2` = d, or the ×2 domain when upsampling).
    fn resblk_b(&self, blk: &GResBlk, x: &Buf, d: &Dom, d2: &Dom, styles: &Buf) -> Result<Buf> {
        let g = &self.gpu;
        let r = self.adain_b(&blk.norm1, x, d, styles, Act::Leaky(0.2))?;
        let mut r = match &blk.pool {
            Some((w, b)) => {
                let mut y = g.alloc(blk.dim_in * d2.l)?;
                let (ci, ti) = (blk.dim_in as i32, d.l as i32);
                ensure!(d2.l == 2 * d.l, "upsample domain must be exactly 2x");
                launch!(g, dw_convT_k3s2, cfg1(blk.dim_in * d2.l), &r, w, b, &mut y, &ci, &ti)?;
                y
            }
            None => r,
        };
        let r = self.conv_b(&blk.conv1, &mut r, d2)?;
        let mut r = self.adain_b(&blk.norm2, &r, d2, styles, Act::Leaky(0.2))?;
        let mut r = self.conv_b(&blk.conv2, &mut r, d2)?;
        let mut sc = if blk.upsample {
            let mut y = g.alloc(blk.dim_in * d2.l)?;
            let (ci, ti) = (blk.dim_in as i32, d.l as i32);
            launch!(g, upsample_nearest2, cfg1(blk.dim_in * d2.l), x, &mut y, &ci, &ti)?;
            y
        } else {
            g.stream.clone_dtod(x)?
        };
        if let Some(c) = &blk.conv1x1 {
            sc = self.conv_b(c, &mut sc, d2)?;
        }
        let n = r.len() as i64;
        let s2 = 1.0f32 / 2.0f32.sqrt();
        launch!(g, residual_scale, cfg1(r.len()), &mut r, &sc, &n, &s2)?;
        Ok(r)
    }

    fn snake_b(&self, blk: &GSnakeBlk, x: &Buf, d: &Dom, styles: &Buf) -> Result<Buf> {
        let g = &self.gpu;
        let mut x = g.stream.clone_dtod(x)?;
        let fuse = *FUSE_ON && (0..3).all(|i| blk.convs1[i].igemm_applicable(g) && blk.convs2[i].igemm_applicable(g));
        let lp_fuse = *FUSE_ON && (0..3).all(|i| blk.convs1[i].wmma_ok(d.l, d.l) && blk.convs2[i].wmma_ok(d.l, d.l));
        for i in 0..3 {
            if lp_fuse && blk.convs1[i].wmma_pro_ok(d.l) && blk.convs2[i].wmma_pro_ok(d.l) {
                // PHASE 2 (P2-L4, kill switch KOKORO_LP_PROLOGUE=0): AdaIN + Snake applied inside the
                // conv's input staging; only the statistics are computed separately.
                let (a1, a2) = (&blk.adain1[i], &blk.adain2[i]);
                let (gb, mean, rstd) = self.adain_stats(a1, &x, d, styles)?;
                let mut h = g.alloc(blk.convs1[i].cout * d.l)?;
                blk.convs1[i].fwd_wmma_pro(g, &x, d.l, &mut h, false, &d.col_item, &mean, &rstd, &a1.nw, &a1.nb, &gb, &blk.alpha1[i])?;
                let (gb, mean, rstd) = self.adain_stats(a2, &h, d, styles)?;
                blk.convs2[i].fwd_wmma_pro(g, &h, d.l, &mut x, true, &d.col_item, &mean, &rstd, &a2.nw, &a2.nb, &gb, &blk.alpha2[i])?;
                continue;
            }
            if lp_fuse {
                // PHASE 2: tensor-core convs; AdaIN output has zero gaps (no mask); conv2 accumulates
                // into x in its epilogue.
                let h = self.adain_b(&blk.adain1[i], &x, d, styles, Act::Snake(&blk.alpha1[i]))?;
                let (h, _) = blk.convs1[i].fwd(g, &h, d.l)?;
                let h = self.adain_b(&blk.adain2[i], &h, d, styles, Act::Snake(&blk.alpha2[i]))?;
                blk.convs2[i].fwd_wmma_res(g, &h, d.l, &mut x)?;
                continue;
            }
            if fuse {
                // LEVER PL-009 (kill switch KOKORO_FUSE_RES_CONV=0): AdaIN+Snake applied separately (its
                // output already has zero gaps, so no mask is needed); conv2 accumulates into x in its
                // epilogue (x + conv, the add_inplace order) -> bitwise identical to the unfused path
                let h = self.adain_b(&blk.adain1[i], &x, d, styles, Act::Snake(&blk.alpha1[i]))?;
                let (h, _) = blk.convs1[i].fwd(g, &h, d.l)?;
                let h = self.adain_b(&blk.adain2[i], &h, d, styles, Act::Snake(&blk.alpha2[i]))?;
                blk.convs2[i].fwd_igemm_res(g, &h, d.l, &mut x)?;
                continue;
            }
            let mut xt = self.adain_b(&blk.adain1[i], &x, d, styles, Act::Snake(&blk.alpha1[i]))?;
            let xt = self.conv_b(&blk.convs1[i], &mut xt, d)?;
            let mut xt = self.adain_b(&blk.adain2[i], &xt, d, styles, Act::Snake(&blk.alpha2[i]))?;
            let xt = self.conv_b(&blk.convs2[i], &mut xt, d)?;
            g.add(&mut x, &xt)?;
        }
        Ok(x)
    }

    /// Batched BiLSTM over row spans of `d` (rows = d.l): x [l, din] -> [l, 2H] (gap rows 0).
    fn lstm_b(&self, l: &GLstm, x: &Buf, d: &Dom) -> Result<Buf> {
        let g = &self.gpu;
        let gf = l.wih[0].fwd(g, x, d.l)?;
        let gbk = l.wih[1].fwd(g, x, d.l)?;
        let b = d.b();
        let mut hbuf = g.stream.alloc_zeros::<f32>(2 * b * 2 * l.h)?;
        let mut c = g.stream.alloc_zeros::<f32>(b * 2 * l.h)?;
        let mut out = g.stream.alloc_zeros::<f32>(d.l * 2 * l.h)?;
        let max_t = *d.len.iter().max().unwrap_or(&0) as i32;
        let (bi, hi) = (b as i32, l.h as i32);
        let cfg = LaunchConfig { grid_dim: (l.h as u32, 2, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let mut lb = g.stream.launch_builder(&g.k.lstm_seq_batched);
        lb.arg(&gf).arg(&gbk).arg(&l.whh[0]).arg(&l.whh[1]).arg(&l.bhh[0]).arg(&l.bhh[1]);
        lb.arg(&mut hbuf).arg(&mut c).arg(&mut out).arg(&d.start_d).arg(&d.len_d).arg(&bi).arg(&max_t).arg(&hi);
        // SAFETY: arguments match lstm_seq_batched; buffers sized as indexed (hbuf 2*B*2H, c B*2H,
        // out l*2H, gx l*4H); grid (H,2)x128 co-resident (cooperative launch fails loudly otherwise).
        unsafe { lb.launch_cooperative(cfg) }.context("lstm_seq_batched")?;
        Ok(out)
    }

    /// Batched forward. Items keep their own ids, style, speed and noise; results are returned in
    /// input order. Semantics per item are those of `forward_ids`.
    pub fn forward_batch(&self, m: &Kokoro, items: &[BatchItem]) -> Result<Vec<Output>> {
        let g = &self.gpu;
        let nb = items.len();
        ensure!(nb > 0, "empty batch");
        for it in items {
            ensure!(it.ref_s.len() == 2 * STYLE_DIM, "ref_s must have 256 values");
            ensure!(it.ids.len() >= 2 && it.ids.len() <= model::CONTEXT_LEN, "input_ids length outside [2, 512]");
            ensure!(it.speed.is_finite() && it.speed > 0.0, "speed must be positive and finite");
        }
        // ---- token layout
        let mut tstart = vec![];
        let mut tlen = vec![];
        let mut rows = 0;
        for it in items {
            tstart.push(rows);
            tlen.push(it.ids.len());
            rows += it.ids.len() + GAP_T;
        }
        let dt = Dom::new(g, rows, tstart.clone(), tlen.clone())?;
        let row_item: Vec<i32> = {
            let mut v = vec![-1i32; rows];
            for (b, (&s, &n)) in tstart.iter().zip(&tlen).enumerate() {
                v[s..s + n].iter_mut().for_each(|x| *x = b as i32);
            }
            v
        };
        let row_item_d = g.stream.clone_htod(&row_item)?;
        let mut sdec_h = Vec::with_capacity(nb * STYLE_DIM);
        let mut spro_h = Vec::with_capacity(nb * STYLE_DIM);
        for it in items {
            sdec_h.extend_from_slice(&it.ref_s[..STYLE_DIM]);
            spro_h.extend_from_slice(&it.ref_s[STYLE_DIM..]);
        }
        let s_dec = g.up(&sdec_h)?;
        let s_pro = g.up(&spro_h)?;

        // ALBERT: embeddings on the host per item, placed in the token layout
        let mut emb = vec![0.0f32; rows * albert::EMB];
        for (b, it) in items.iter().enumerate() {
            let e = m.albert.embeddings(it.ids)?;
            emb[tstart[b] * albert::EMB..(tstart[b] + tlen[b]) * albert::EMB].copy_from_slice(&e);
        }
        let emb = g.up(&emb)?;
        let p_front = crate::prof::scope("gpub.albert+duration");
        let bert = self.albert_b(&emb, &dt)?;
        let d_en = self.bert_encoder.fwd(g, &bert, rows)?;
        // duration encoder
        let dd = HIDDEN + STYLE_DIM;
        let cat = |x: &Buf, w: usize| -> Result<Buf> {
            let mut y = g.alloc(rows * (w + STYLE_DIM))?;
            let (ri, wi, si) = (rows as i32, w as i32, STYLE_DIM as i32);
            launch!(g, cat_style_rows_seg, cfg1(rows * (w + STYLE_DIM)), x, &s_pro, &row_item_d, &mut y, &ri, &wi, &si)?;
            Ok(y)
        };
        let mut x = cat(&d_en, HIDDEN)?;
        for (lstm, fc) in self.dur_lstms.iter().zip(&self.dur_norms) {
            let mut h = self.lstm_b(lstm, &x, &dt)?;
            let gb = fc.fwd(g, &s_pro, nb)?;
            let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let (di, eps) = (HIDDEN as i32, 1e-5f32);
            launch!(g, adaln_rows_seg, cfg, &mut h, &row_item_d, &gb, &di, &eps)?;
            x = cat(&h, HIDDEN)?;
        }
        let d = x;
        let xl = self.lstm_b(&self.pred_lstm, &d, &dt)?;
        let logits = g.down(&self.dur_proj.fwd(g, &xl, rows)?)?;
        let mut pred = vec![];
        for (b, it) in items.iter().enumerate() {
            let lg = &logits[tstart[b] * model::MAX_DUR..(tstart[b] + tlen[b]) * model::MAX_DUR];
            pred.push(model::durations_from_logits(lg, tlen[b], it.speed));
        }

        drop(p_front);
        // ---- frame layout (N domain) and derived domains
        let nfs: Vec<usize> = pred.iter().map(|p| p.iter().sum::<i64>() as usize).collect();
        let mut a = vec![];
        let mut ln = 0;
        for &n in &nfs {
            a.push(ln);
            ln += n + GAP_N;
        }
        let scale = |f: usize| -> Vec<usize> { a.iter().map(|&x| x * f).collect() };
        let dn = Dom::new(g, ln, a.clone(), nfs.clone())?;
        let d2 = Dom::new(g, 2 * ln, scale(2), nfs.iter().map(|&n| 2 * n).collect())?;
        let d20 = Dom::new(g, 20 * ln, scale(20), nfs.iter().map(|&n| 20 * n).collect())?;
        let l120 = 120 * ln + 1;
        let d120 = Dom::new(g, l120, scale(120), nfs.iter().map(|&n| 120 * n + 1).collect())?;

        // frame expansion tables (frame row -> token row), -1 in gaps
        let mut src_row = vec![-1i32; ln];
        for b in 0..nb {
            let mut f = a[b];
            for (tok, &dur) in pred[b].iter().enumerate() {
                for _ in 0..dur {
                    src_row[f] = (tstart[b] + tok) as i32;
                    f += 1;
                }
            }
        }
        let src_row_d = g.stream.clone_htod(&src_row)?;
        let mut en = g.alloc(ln * dd)?;
        let (li, ddi) = (ln as i32, dd as i32);
        launch!(g, gather_rows, cfg1(ln * dd), &d, &src_row_d, &mut en, &li, &ddi)?;
        // F0 / N
        let p_f0 = crate::prof::scope("gpub.f0n+textenc");
        let sh = self.lstm_b(&self.shared, &en, &dn)?;
        let sh = g.transpose(&sh, ln, HIDDEN)?;
        let run = |blocks: &[GResBlk], proj: &GConv| -> Result<Buf> {
            let h = self.resblk_b(&blocks[0], &sh, &dn, &dn, &s_pro)?;
            let h = self.resblk_b(&blocks[1], &h, &dn, &d2, &s_pro)?;
            let mut h = self.resblk_b(&blocks[2], &h, &d2, &d2, &s_pro)?;
            self.conv_b(proj, &mut h, &d2)
        };
        let mut f0 = run(&self.f0, &self.f0_proj)?;
        let mut ncur = run(&self.n, &self.n_proj)?;
        self.mask(&mut f0, 1, &d2)?;
        self.mask(&mut ncur, 1, &d2)?;
        if let Ok(dir) = std::env::var("KOKORO_DEBUG_F0_DIR") {
            let all = g.down(&f0)?;
            for (b, it) in items.iter().enumerate() {
                let key = crate::engine::sha256_bytes(&it.ids.iter().flat_map(|v| v.to_le_bytes()).chain(it.ref_s.iter().flat_map(|v| v.to_le_bytes())).chain(it.speed.to_le_bytes()).collect::<Vec<u8>>());
                let v: Vec<u8> = all[2 * a[b]..2 * a[b] + 2 * nfs[b]].iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("batch-{}.f32", &key[..16])), v)?;
            }
        }
        // text encoder on the token layout
        let t_en = self.text_encoder_b(m, items, &dt)?;
        let mut asr = g.alloc(HIDDEN * ln)?;
        let (ci, lti) = (HIDDEN as i32, rows as i32);
        launch!(g, gather_cols, cfg1(HIDDEN * ln), &t_en, &lti, &src_row_d, &mut asr, &ci, &li)?;

        g.prof_sync_pub();
        drop(p_f0);
        // ---- decoder (pre-generator)
        let p_dec = crate::prof::scope("gpub.decoder");
        let (f0n, _) = self.f0_conv.fwd(g, &f0, 2 * ln)?;
        let (nn_, _) = self.n_conv.fwd(g, &ncur, 2 * ln)?;
        let mut xcat = g.cat_channels(&[&asr, &f0n, &nn_])?;
        self.mask(&mut xcat, HIDDEN + 2, &dn)?;
        let mut xd = self.resblk_b(&self.encode, &xcat, &dn, &dn, &s_dec)?;
        let mut asr_m = asr;
        let asr_res = self.conv_b(&self.asr_res, &mut asr_m, &dn)?;
        let mut t_is_2n = false;
        for block in &self.decode {
            let x_in = if !t_is_2n { g.cat_channels(&[&xd, &asr_res, &f0n, &nn_])? } else { xd };
            xd = if block.upsample {
                t_is_2n = true;
                self.resblk_b(block, &x_in, &dn, &d2, &s_dec)?
            } else {
                self.resblk_b(block, &x_in, &dn, &dn, &s_dec)?
            };
        }
        ensure!(t_is_2n, "decoder did not upsample");

        g.prof_sync_pub();
        drop(p_dec);
        // ---- harmonic source + STFT per item (own noise stream per item)
        let p_src = crate::prof::scope("gpub.source+stft");
        let ls = 600 * ln;
        let mut har_src = g.stream.alloc_zeros::<f32>(ls)?;
        let mut spec = g.stream.alloc_zeros::<f32>(22 * l120)?;
        let near = (1.0 / UPSAMPLE_SCALE as f64) as f32;
        let down = (1.0 / (1.0 / UPSAMPLE_SCALE as f64)) as f32;
        let up = (1.0 / UPSAMPLE_SCALE as f64) as f32;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        for (b, it) in items.iter().enumerate() {
            let len2n = 2 * nfs[b];
            let s_len = len2n * UPSAMPLE_SCALE;
            let (ri, nz) = match &it.noise {
                ItemNoise::Counter(seed) => {
                    let mut nz = g.alloc(s_len * HARMONICS)?;
                    let n = (s_len * HARMONICS) as i64;
                    launch!(g, gen_noise, cfg1(s_len * HARMONICS / 2 + 1), seed, &mut nz, &n)?;
                    (g.up(&RngNoise::new(*seed).rand_ini())?, nz)
                }
                ItemNoise::Fixed { rand_ini, sine } => {
                    ensure!(sine.len() == s_len * HARMONICS, "fixed noise size mismatch for item {b}");
                    (g.up(rand_ini)?, g.up(sine)?)
                }
            };
            let dsz = (s_len as f64 * (1.0 / UPSAMPLE_SCALE as f64)).floor() as usize;
            let mut pp = g.alloc(HARMONICS * dsz)?;
            let (lni, si, di) = (len2n as i32, s_len as i32, dsz as i32);
            let f0v = f0.slice(2 * a[b]..2 * a[b] + len2n);
            launch!(g, sine_phase_pre, one, &f0v, &lni, &si, &di, &near, &down, &ri, &mut pp)?;
            {
                let mut hv = har_src.slice_mut(600 * a[b]..600 * a[b] + s_len);
                launch!(g, sine_har_source, cfg1(s_len), &f0v, &lni, &si, &di, &near, &up, &pp, &nz, &self.l_w, &self.l_b, &mut hv)?;
            }
            let frames = 1 + s_len / vocoder::HOP;
            let hv = har_src.slice(600 * a[b]..600 * a[b] + s_len);
            let mut sv = spec.slice_mut(120 * a[b]..);
            let (fi, ldi) = (frames as i32, l120 as i32);
            launch!(g, stft20_ld, cfg1(frames), &hv, &si, &mut sv, &fi, &ldi, &g.tw, &g.win)?;
        }

        g.prof_sync_pub();
        drop(p_src);
        // ---- generator on the scaled layouts
        let mut xg = xd;
        for i in 0..2 {
            let _pg = crate::prof::scope(if i == 0 { "gpub.gen.stage0 (256ch)" } else { "gpub.gen.stage1 (128ch)" });
            let (din, dout) = if i == 0 { (&d2, &d20) } else { (&d20, &d120) };
            g.leaky(&mut xg, 0.1)?;
            let mut har_in = g.stream.clone_dtod(&spec)?;
            self.mask(&mut har_in, 22, &d120)?;
            let (xs, ts) = self.noise_convs[i].fwd(g, &har_in, l120)?;
            ensure!(ts >= dout.l, "noise conv output shorter than the layout");
            let xs = if ts == dout.l { xs } else {
                let mut t = g.alloc(self.noise_convs[i].cout * dout.l)?;
                for c in 0..self.noise_convs[i].cout {
                    let mut dst = t.slice_mut(c * dout.l..(c + 1) * dout.l);
                    g.stream.memcpy_dtod(&xs.slice(c * ts..c * ts + dout.l), &mut dst)?;
                }
                t
            };
            let xs = self.snake_b(&self.noise_res[i], &xs, dout, &s_dec)?;
            self.mask(&mut xg, self.ups[i].cin, din)?;
            let (y, ty) = self.ups[i].fwd(g, &xg, din.l)?;
            let c = self.ups[i].cout;
            let mut y = if i == 1 {
                ensure!(ty + 1 == dout.l, "stage-1 length");
                let mut p = g.alloc(c * dout.l)?;
                let (ci, li) = (c as i32, dout.l as i32);
                let mut yy = g.alloc(c * dout.l)?;
                for ch in 0..c {
                    let mut dst = yy.slice_mut(ch * dout.l..ch * dout.l + ty);
                    g.stream.memcpy_dtod(&y.slice(ch * ty..(ch + 1) * ty), &mut dst)?;
                }
                launch!(g, reflect_pad_left1_seg, cfg1(c * dout.l), &yy, &mut p, &dout.col_item, &dout.start_d, &ci, &li)?;
                p
            } else {
                ensure!(ty == dout.l, "stage-0 length");
                y
            };
            g.add(&mut y, &xs)?;
            let mut acc = self.snake_b(&self.resblocks[i * 3], &y, dout, &s_dec)?;
            for j in 1..3 {
                let r = self.snake_b(&self.resblocks[i * 3 + j], &y, dout, &s_dec)?;
                g.add(&mut acc, &r)?;
            }
            let (n, three) = (acc.len() as i64, 3.0f32);
            launch!(g, div_inplace, cfg1(acc.len()), &mut acc, &three, &n)?;
            xg = acc;
            g.prof_sync_pub();
        }
        g.leaky(&mut xg, 0.01)?;
        let _pp = crate::prof::scope("gpub.post+istft");
        let post = self.conv_b(&self.conv_post, &mut xg, &d120)?;
        let mut outs = Vec::with_capacity(nb);
        for b in 0..nb {
            let frames = 120 * nfs[b] + 1;
            // SAFETY: fully written by istft_frames_ld before istft_ola reads it.
            let mut fr = unsafe { g.stream.alloc::<f64>(frames * 20) }?;
            let pv = post.slice(120 * a[b]..);
            let (fi, ldi) = (frames as i32, l120 as i32);
            launch!(g, istft_frames_ld, cfg1(frames), &pv, &fi, &ldi, &mut fr, &g.tw, &g.win)?;
            let len = vocoder::HOP * (frames - 1);
            let mut audio = g.alloc(len)?;
            let lni = len as i32;
            launch!(g, istft_ola, cfg1(len), &fr, &fi, &mut audio, &lni, &g.win)?;
            outs.push(Output { audio: g.down(&audio)?, pred_dur: pred[b].clone() });
        }
        Ok(outs)
    }

    fn albert_b(&self, emb: &Buf, dt: &Dom) -> Result<Buf> {
        use albert::{HEADS, HEAD_DIM, HID};
        let g = &self.gpu;
        let al = &self.albert;
        let rows = dt.l;
        let mut h = al.map_in.fwd(g, emb, rows)?;
        for _ in 0..albert::LAYERS {
            let q = al.q.fwd(g, &h, rows)?;
            let k = al.k.fwd(g, &h, rows)?;
            let v = al.v.fwd(g, &h, rows)?;
            let mut ctx = g.stream.alloc_zeros::<f32>(rows * HID)?;
            for b in 0..dt.b() {
                let (s, t) = (dt.start[b], dt.len[b]);
                let mut scores = g.alloc(HEADS * t * t)?;
                g.gemm_batched(true, false, t, t, HEAD_DIM, &k, s * HID, HID, HEAD_DIM, &q, s * HID, HID, HEAD_DIM, &mut scores, 0, t, t * t, HEADS)?;
                let cfg = LaunchConfig { grid_dim: ((HEADS * t) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let (ni, scale) = (t as i32, (HEAD_DIM as f32).powf(-0.5));
                launch!(g, softmax_rows, cfg, &mut scores, &ni, &scale)?;
                g.gemm_batched(false, false, HEAD_DIM, t, t, &v, s * HID, HID, HEAD_DIM, &scores, 0, t, t * t, &mut ctx, s * HID, HID, HEAD_DIM, HEADS)?;
            }
            let mut a = al.dense.fwd(g, &ctx, rows)?;
            g.add(&mut a, &h)?;
            GAlbert::ln(g, &mut a, rows, HID, &al.attn_ln.0, &al.attn_ln.1, 1e-12)?;
            let mut f = al.ffn.fwd(g, &a, rows)?;
            let n = f.len() as i64;
            launch!(g, gelu_new, cfg1(f.len()), &mut f, &n)?;
            let mut o = al.ffn_out.fwd(g, &f, rows)?;
            g.add(&mut o, &a)?;
            GAlbert::ln(g, &mut o, rows, HID, &al.full_ln.0, &al.full_ln.1, 1e-12)?;
            h = o;
        }
        Ok(h)
    }

    fn text_encoder_b(&self, m: &Kokoro, items: &[BatchItem], dt: &Dom) -> Result<Buf> {
        let g = &self.gpu;
        let te = &m.text_encoder;
        let rows = dt.l;
        let mut e = vec![0.0f32; rows * HIDDEN];
        for (b, it) in items.iter().enumerate() {
            for (i, &id) in it.ids.iter().enumerate() {
                ensure!(id >= 0 && (id as usize) < te.n_token, "token id {id} out of range");
                let r = dt.start[b] + i;
                e[r * HIDDEN..(r + 1) * HIDDEN].copy_from_slice(&te.embedding[id as usize * HIDDEN..(id as usize + 1) * HIDDEN]);
            }
        }
        let e = g.up(&e)?;
        let mut x = g.transpose(&e, rows, HIDDEN)?;
        for (conv, ga, be) in &self.te_cnn {
            let y = self.conv_b(conv, &mut x, dt)?;
            let mut yt = g.transpose(&y, HIDDEN, rows)?;
            GAlbert::ln(g, &mut yt, rows, HIDDEN, ga, be, 1e-5)?;
            g.leaky(&mut yt, 0.2)?;
            x = g.transpose(&yt, rows, HIDDEN)?;
        }
        let xt = g.transpose(&x, HIDDEN, rows)?;
        let h = self.lstm_b(&self.te_lstm, &xt, dt)?;
        g.transpose(&h, rows, HIDDEN)
    }
}

