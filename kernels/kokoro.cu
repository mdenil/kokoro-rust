// Kokoro-82M CUDA kernels (f32). Compiled with -fmad=false so elementwise arithmetic rounds
// like the CPU path in src/ops.rs, src/nn.rs and src/vocoder.rs; explicit fmaf() appears only
// where the CPU path itself uses mul_add (torch interpolation semantics).
// Matrix products are done by cuBLAS in src/gpu.rs, not here.

#include <cooperative_groups.h>

#define PI_F 3.14159274101257324f  // (float)M_PI, as std::f32::consts::PI

extern "C" __global__ void fill_channels(float* y, const float* bias, int C, int T) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < (long)C * T) y[i] = bias ? bias[i / T] : 0.0f;
}

extern "C" __global__ void fill_rows(float* y, const float* bias, int rows, int d) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < (long)rows * d) y[i] = bias[i % d];
}

extern "C" __global__ void leaky_relu(float* x, long n, float slope) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = x[i] * slope;
}

extern "C" __global__ void add_inplace(float* a, const float* b, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] + b[i];
}

extern "C" __global__ void div_inplace(float* a, float d, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] / d;
}

// r = (r + sc) * s   (AdainResBlk1d output, rsqrt(2) scaling)
extern "C" __global__ void residual_scale(float* r, const float* sc, long n, float s) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) r[i] = (r[i] + sc[i]) * s;
}

// Per-channel mean / 1/sqrt(var+eps) over time of x [C, T] (biased var, f64 accumulation;
// mirrors ops::instance_norm). One block (256 threads) per channel.
extern "C" __global__ void chan_stats(const float* x, int T, float eps, float* mean_out, float* rstd_out) {
    __shared__ double sh[256];
    const float* row = x + (long)blockIdx.x * T;
    double s = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) s += (double)row[t];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float mean = (float)(sh[0] / (double)T);
    __syncthreads();
    double v = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        double c = (double)(row[t] - mean);
        v += c * c;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float var = (float)(sh[0] / (double)T);
        mean_out[blockIdx.x] = mean;
        rstd_out[blockIdx.x] = 1.0f / sqrtf(var + eps);
    }
}

// AdaIN1d apply: y = (1+gamma)*((x-mean)*rstd*nw + nb) + beta, then optional activation:
// act 0 = none, 1 = leaky relu(slope), 2 = snake(alpha). gb = fc(s) = [gamma(C), beta(C)].
extern "C" __global__ void adain_apply(const float* x, float* y, const float* mean, const float* rstd,
                                       const float* nw, const float* nb, const float* gb, const float* alpha,
                                       int C, int T, int act, float slope) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * T) return;
    int c = (int)(i / T);
    float v = (x[i] - mean[c]) * rstd[c];
    float g1 = 1.0f + gb[c];
    v = g1 * (v * nw[c] + nb[c]) + gb[C + c];
    if (act == 1) {
        if (v < 0.0f) v = v * slope;
    } else if (act == 2) {
        float a = alpha[c];
        float inv = 1.0f / a;
        float sn = sinf(a * v);
        v = v + inv * (sn * sn);
    }
    y[i] = v;
}

// LayerNorm over the last dim of [rows, d]; optional affine (g, b may be null). Block per row.
// mode 1: AdaLayerNorm: out = (1 + gb[j]) * ln + gb[d + j]   (no affine)
extern "C" __global__ void layer_norm_rows(float* x, const float* g, const float* b, const float* gb,
                                           int rows, int d, float eps) {
    __shared__ double sh[256];
    float* row = x + (long)blockIdx.x * d;
    double s = 0.0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s += (double)row[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float mean = (float)(sh[0] / (double)d);
    __syncthreads();
    double v = 0.0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        double c = (double)(row[j] - mean);
        v += c * c;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf((float)(sh[0] / (double)d) + eps);
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float o = (row[j] - mean) * inv;
        if (gb) {
            o = (1.0f + gb[j]) * o + gb[d + j];
        } else {
            if (g) o = o * g[j];
            if (b) o = o + b[j];
        }
        row[j] = o;
    }
}

extern "C" __global__ void gelu_new(float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = 0.797884560802865355f;  // (float)sqrt(2/pi) as computed in f64 then cast
    float a = x[i];
    x[i] = 0.5f * a * (1.0f + tanhf(c * (a + 0.044715f * a * a * a)));
}

// Row softmax of scale*x, block per row (row length n <= 512 here).
extern "C" __global__ void softmax_rows(float* x, int n, float scale) {
    __shared__ float sh[256];
    float* row = x + (long)blockIdx.x * n;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] = row[j] * scale;
        mx = fmaxf(mx, row[j]);
    }
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float s = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        s += row[j];
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float inv = 1.0f / sh[0];
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] = row[j] * inv;
}

extern "C" __global__ void transpose(const float* x, float* y, int R, int C) {
    __shared__ float tile[32][33];
    int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        int r = by + k, c = bx + threadIdx.x;
        if (r < R && c < C) tile[k][threadIdx.x] = x[(long)r * C + c];
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        int c = bx + k, r = by + threadIdx.x;
        if (r < R && c < C) y[(long)c * R + r] = tile[threadIdx.x][k];
    }
}

// out [T, d + sd] = concat(x[T, d], s[sd]) per row
extern "C" __global__ void cat_style_rows(const float* x, const float* s, float* out, int T, int d, int sd) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    int w = d + sd;
    if (i >= (long)T * w) return;
    int t = (int)(i / w), j = (int)(i % w);
    out[i] = j < d ? x[(long)t * d + j] : s[j - d];
}

// out[f, :] = d[aln[f], :]  (frame expansion of token rows, width dd)
extern "C" __global__ void expand_rows(const float* d, const int* aln, float* out, int nf, int dd) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)nf * dd) return;
    int f = (int)(i / dd), j = (int)(i % dd);
    out[i] = d[(long)aln[f] * dd + j];
}

// out[c, f] = x[c, aln[f]]  (frame expansion of channel-major [C, T] -> [C, nf])
extern "C" __global__ void expand_cols(const float* x, const int* aln, float* out, int C, int T, int nf) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * nf) return;
    int c = (int)(i / nf), f = (int)(i % nf);
    out[i] = x[(long)c * T + aln[f]];
}

// One LSTM time step for both directions (grid.x = H units, grid.y = 2 directions, 128 thr).
// gx: [T, 4H] per direction (x W_ih^T + b_ih); whh [4H, H]; bhh [4H].
// Gate order i, f, g, o. Mirrors nn::BiLstm::direction: g = gx + (dot + bhh).
extern "C" __global__ void lstm_step(const float* gx_f, const float* gx_b, const float* whh_f, const float* whh_b,
                                     const float* bhh_f, const float* bhh_b, const float* h_cur, float* h_nxt,
                                     float* c_state, float* out, int T, int H, int step) {
    int j = blockIdx.x, dir = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int t = dir == 0 ? step : T - 1 - step;
    const float* gx = dir == 0 ? gx_f : gx_b;
    const float* whh = dir == 0 ? whh_f : whh_b;
    const float* bhh = dir == 0 ? bhh_f : bhh_b;
    const float* h = h_cur + dir * H;
    __shared__ float gates[4];
    int r = warp * H + j;
    const float* w = whh + (long)r * H;
    float acc = 0.0f;
    for (int k = lane; k < H; k += 32) acc = acc + w[k] * h[k];
    for (int off = 16; off > 0; off >>= 1) acc = acc + __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) gates[warp] = gx[(long)t * 4 * H + r] + (acc + bhh[r]);
    __syncthreads();
    if (threadIdx.x == 0) {
        float ig = 1.0f / (1.0f + expf(-gates[0]));
        float fg = 1.0f / (1.0f + expf(-gates[1]));
        float gg = tanhf(gates[2]);
        float og = 1.0f / (1.0f + expf(-gates[3]));
        float c = fg * c_state[dir * H + j] + ig * gg;
        c_state[dir * H + j] = c;
        float hn = og * tanhf(c);
        h_nxt[dir * H + j] = hn;
        out[(long)t * 2 * H + dir * H + j] = hn;
    }
}

// Nearest x2 along time: y [C, 2T]
extern "C" __global__ void upsample_nearest2(const float* x, float* y, int C, int T) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * 2 * T) return;
    int c = (int)(i / (2 * T)), o = (int)(i % (2 * T));
    y[i] = x[(long)c * T + (o >> 1)];
}

// Depthwise ConvTranspose1d k=3, s=2, p=1, op=1 -> [C, 2T]; mirrors ops::conv_transpose1d_depthwise
// accumulation order (bias, then i ascending over contributing inputs).
extern "C" __global__ void dw_convT_k3s2(const float* x, const float* w, const float* b, float* y, int C, int T) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    int To = 2 * T;
    if (idx >= (long)C * To) return;
    int c = (int)(idx / To), o = (int)(idx % To);
    float acc = b[c];
    // contributions: o = 2i + k - 1  ->  i = (o + 1 - k) / 2 for k in 0..3, ascending i
    for (int i = (o - 1) / 2 - 1; i <= (o + 1) / 2; i++) {
        if (i < 0 || i >= T) continue;
        int k = o + 1 - 2 * i;
        if (k < 0 || k >= 3) continue;
        acc = acc + w[c * 3 + k] * x[(long)c * T + i];
    }
    y[idx] = acc;
}

// Direct Conv1d for small Cin*K (strided convs): y[co, o] = b + sum_{k, ci} w * x.
// Summation order mirrors ops::conv1d at element level only approximately (GEMM order differs).
extern "C" __global__ void conv_direct(const float* x, const float* w, const float* b, float* y,
                                       int Cin, int T, int Cout, int K, int stride, int pad, int Tout) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)Cout * Tout) return;
    int co = (int)(idx / Tout), o = (int)(idx % Tout);
    float acc = b ? b[co] : 0.0f;
    for (int k = 0; k < K; k++) {
        int t = o * stride + k - pad;
        if (t < 0 || t >= T) continue;
        float s = 0.0f;
        for (int ci = 0; ci < Cin; ci++) s = s + w[((long)co * Cin + ci) * K + k] * x[(long)ci * T + t];
        acc = acc + s;
    }
    y[idx] = acc;
}

// ConvTranspose1d gather from Z [(co*K + k), Tin] (Z = per-tap products from one GEMM):
// y[co, o] = b[co] + sum_{k asc, (o+pad-k) % s == 0} Z[co*K+k, (o+pad-k)/s]
extern "C" __global__ void convT_gather(const float* Z, const float* b, float* y, int Cout, int K, int Tin,
                                        int stride, int pad, int Tout) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)Cout * Tout) return;
    int co = (int)(idx / Tout), o = (int)(idx % Tout);
    float acc = b[co];
    for (int k = 0; k < K; k++) {
        int num = o + pad - k;
        if (num < 0 || num % stride != 0) continue;
        int i = num / stride;
        if (i >= Tin) continue;
        acc = acc + Z[((long)co * K + k) * Tin + i];
    }
    y[idx] = acc;
}

// y [C, T+1]: y[:, 0] = x[:, 1], y[:, 1:] = x   (ReflectionPad1d((1, 0)))
extern "C" __global__ void reflect_pad_left1(const float* x, float* y, int C, int T) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * (T + 1)) return;
    int c = (int)(i / (T + 1)), o = (int)(i % (T + 1));
    y[i] = o == 0 ? x[(long)c * T + 1] : x[(long)c * T + o - 1];
}

// ---------------------------------------------------------------- harmonic source (SineGen)

__device__ __forceinline__ float torch_remainder1(float a) {
    float m = fmodf(a, 1.0f);
    if (m != 0.0f && (m < 0.0f)) m = m + 1.0f;
    return m;
}

// rad value at sample t for harmonic h (f0 curve upsampled nearest by 300)
__device__ __forceinline__ float sine_rad(const float* f0c, int L, float near_scale, int t, int h,
                                          const float* rand_ini) {
    int src = (int)floorf((float)t * near_scale);
    if (src > L - 1) src = L - 1;
    float r = torch_remainder1(f0c[src] * (float)(h + 1) / 24000.0f);
    if (t == 0 && h > 0) r = r + rand_ini[h];
    return r;
}

// phase_pre[h, d] for d < D = floor(S/300): linear downsample of rad (scale 300 -> 1/300),
// cumulative sum (f64 accumulation), *2*pi, *300. One thread per harmonic (sequential in d).
extern "C" __global__ void sine_phase_pre(const float* f0c, int L, int S, int D, float near_scale,
                                          float down_scale, const float* rand_ini, float* phase_pre) {
    int h = threadIdx.x;
    if (h >= 9) return;
    double acc = 0.0;
    for (int d = 0; d < D; d++) {
        float real = fmaf(down_scale, (float)d + 0.5f, -0.5f);
        if (real < 0.0f) real = 0.0f;
        int i0 = (int)floorf(real);
        if (i0 > S - 1) i0 = S - 1;
        float l1 = fminf(fmaxf(real - (float)i0, 0.0f), 1.0f);
        int i1 = i0 < S - 1 ? i0 + 1 : i0;
        float l0 = 1.0f - l1;
        float v = fmaf(sine_rad(f0c, L, near_scale, i0, h, rand_ini), l0,
                       sine_rad(f0c, L, near_scale, i1, h, rand_ini) * l1);
        acc += (double)v;
        phase_pre[h * D + d] = (((float)acc) * 2.0f * PI_F) * 300.0f;
    }
}

// har_source[t] = tanh(sum_h lw[h] * (sin(phase(t,h))*0.1*uv + noise_amp*noise[t*9+h]) + lb)
extern "C" __global__ void sine_har_source(const float* f0c, int L, int S, int D, float near_scale,
                                           float up_scale, const float* phase_pre, const float* noise,
                                           const float* lw, float lb, float* har) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= S) return;
    int src = (int)floorf((float)t * near_scale);
    if (src > L - 1) src = L - 1;
    float f0 = f0c[src];
    float uv = f0 > 10.0f ? 1.0f : 0.0f;
    float noise_amp = uv * 0.003f + (1.0f - uv) * 0.1f / 3.0f;
    float real = fmaf(up_scale, (float)t + 0.5f, -0.5f);
    if (real < 0.0f) real = 0.0f;
    int i0 = (int)floorf(real);
    if (i0 > D - 1) i0 = D - 1;
    float l1 = fminf(fmaxf(real - (float)i0, 0.0f), 1.0f);
    int i1 = i0 < D - 1 ? i0 + 1 : i0;
    float l0 = 1.0f - l1;
    float acc = 0.0f;
    for (int h = 0; h < 9; h++) {
        float ph = fmaf(phase_pre[h * D + i0], l0, phase_pre[h * D + i1] * l1);
        float sine = sinf(ph) * 0.1f;
        float v = sine * uv + noise_amp * noise[(long)t * 9 + h];
        acc = acc + lw[h] * v;
    }
    har[t] = tanhf(acc + lb);
}

// ---------------------------------------------------------------- STFT / iSTFT (n_fft 20, hop 5)

// tw: [cos(11x20) | sin(11x20)] in f64 (same values as vocoder::twiddles), win: periodic Hann(20)
#define C_COS(k, n) tw[(k) * 20 + (n)]
#define C_SIN(k, n) tw[220 + (k) * 20 + (n)]

// out [22, F]: rows 0..10 magnitude, 11..21 phase (atan2); center=True reflect padding.
extern "C" __global__ void stft20(const float* x, int L, float* out, int F, const double* tw, const float* c_win) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    double buf[20];
    for (int n = 0; n < 20; n++) {
        int p = f * 5 + n - 10;  // index into unpadded signal
        int src = p < 0 ? -p : (p >= L ? 2 * L - 2 - p : p);
        buf[n] = (double)(x[src] * c_win[n]);
    }
    for (int k = 0; k < 11; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < 20; n++) {
            re += buf[n] * C_COS(k, n);
            im -= buf[n] * C_SIN(k, n);
        }
        if (k == 0 || k == 10) im = 0.0;
        float rf = (float)re, imf = (float)im;
        out[(long)k * F + f] = hypotf(rf, imf);
        out[(long)(11 + k) * F + f] = atan2f(imf, rf);
    }
}

// frame time signals: fr[f, n] = irfft(exp(post[0..11]) * e^{i sin(post[11..22])})[n] * w[n] (f64)
extern "C" __global__ void istft_frames(const float* post, int F, double* fr, const double* tw, const float* c_win) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    double re[11], im[11];
    for (int k = 0; k < 11; k++) {
        float m = expf(post[(long)k * F + f]);
        float p = sinf(post[(long)(11 + k) * F + f]);
        re[k] = (double)(m * cosf(p));
        im[k] = (double)(m * sinf(p));
    }
    for (int n = 0; n < 20; n++) {
        double v = re[0] + re[10] * ((n % 2 == 0) ? 1.0 : -1.0);
        for (int k = 1; k < 10; k++) v += 2.0 * (re[k] * C_COS(k, n) - im[k] * C_SIN(k, n));
        v = v / 20.0;
        fr[(long)f * 20 + n] = v * (double)c_win[n];
    }
}

// overlap-add gather (frames ascending, like the CPU scatter) / window envelope, trim 10.
extern "C" __global__ void istft_ola(const double* fr, int F, float* out, int len, const float* c_win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int j = i + 10;
    int f_lo = j - 19 > 0 ? (j - 19 + 4) / 5 : 0;
    int f_hi = j / 5;
    if (f_hi > F - 1) f_hi = F - 1;
    double acc = 0.0, env = 0.0;
    for (int f = f_lo; f <= f_hi; f++) {
        int n = j - 5 * f;
        acc += fr[(long)f * 20 + n];
        double w = (double)c_win[n];
        env += w * w;
    }
    out[i] = (float)(acc / env);
}

// ---------------------------------------------------------------- native noise (mirrors vocoder::RngNoise)

__device__ __forceinline__ unsigned long long splitmix_at(unsigned long long seed, unsigned long long ctr) {
    unsigned long long x = seed + (ctr + 1ULL) * 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// out[0..n): Gaussian values, pair i -> out[2i], out[2i+1]
extern "C" __global__ void gen_noise(unsigned long long seed, float* out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (2 * i >= n) return;
    unsigned long long x = splitmix_at(seed, 1024ULL + (unsigned long long)i);
    float u1 = (float)((x >> 40) + 1ULL) * (1.0f / 16777216.0f);
    float u2 = (float)((x >> 16) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
    float r = sqrtf(-2.0f * logf(u1));
    float th = 6.2831855f * u2;
    out[2 * i] = r * cosf(th);
    if (2 * i + 1 < n) out[2 * i + 1] = r * sinf(th);
}

// Whole-sequence BiLSTM in ONE cooperative launch: identical per-step arithmetic to lstm_step
// (same warp dot order, same gate math); grid-wide sync between steps. grid (H, 2), 128 threads.
extern "C" __global__ void lstm_seq(const float* gx_f, const float* gx_b, const float* whh_f, const float* whh_b,
                                    const float* bhh_f, const float* bhh_b, float* hbuf, float* c_state,
                                    float* out, int T, int H) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int j = blockIdx.x, dir = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* gx = dir == 0 ? gx_f : gx_b;
    const float* whh = dir == 0 ? whh_f : whh_b;
    const float* bhh = dir == 0 ? bhh_f : bhh_b;
    __shared__ float gates[4];
    int r = warp * H + j;
    const float* w = whh + (long)r * H;
    for (int step = 0; step < T; step++) {
        int t = dir == 0 ? step : T - 1 - step;
        const float* h = hbuf + (step & 1) * 2 * H + dir * H;
        float* h_nxt = hbuf + ((step + 1) & 1) * 2 * H;
        float acc = 0.0f;
        for (int k = lane; k < H; k += 32) acc = acc + w[k] * h[k];
        for (int off = 16; off > 0; off >>= 1) acc = acc + __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) gates[warp] = gx[(long)t * 4 * H + r] + (acc + bhh[r]);
        __syncthreads();
        if (threadIdx.x == 0) {
            float ig = 1.0f / (1.0f + expf(-gates[0]));
            float fg = 1.0f / (1.0f + expf(-gates[1]));
            float gg = tanhf(gates[2]);
            float og = 1.0f / (1.0f + expf(-gates[3]));
            float c = fg * c_state[dir * H + j] + ig * gg;
            c_state[dir * H + j] = c;
            float hn = og * tanhf(c);
            h_nxt[dir * H + j] = hn;
            out[(long)t * 2 * H + dir * H + j] = hn;
        }
        grid.sync();
    }
}

// ================================================================ B1: ragged batched layout
// A domain buffer is [C, L]; item b owns columns [seg_start[b], seg_start[b] + seg_len[b]);
// col_item[t] = item index or -1 for gap columns. See docs/design/BATCHING.md.

// zero every gap column (applied to conv inputs so per-item zero padding is exact)
extern "C" __global__ void mask_gaps(float* x, const int* col_item, int C, int L) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * L) return;
    if (col_item[i % L] < 0) x[i] = 0.0f;
}

// per (item, channel) mean / rstd over the item's span (f64 accumulation, as chan_stats).
// grid (C, B), 256 threads. Output [B, C].
extern "C" __global__ void chan_stats_seg(const float* x, int L, const int* seg_start, const int* seg_len,
                                          float eps, float* mean_out, float* rstd_out, int C) {
    __shared__ double sh[256];
    int c = blockIdx.x, b = blockIdx.y;
    const float* row = x + (long)c * L + seg_start[b];
    int T = seg_len[b];
    double s = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) s += (double)row[t];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float mean = (float)(sh[0] / (double)T);
    __syncthreads();
    double v = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        double d = (double)(row[t] - mean);
        v += d * d;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float var = (float)(sh[0] / (double)T);
        mean_out[b * C + c] = mean;
        rstd_out[b * C + c] = 1.0f / sqrtf(var + eps);
    }
}

// AdaIN apply with per-item statistics and per-item gamma/beta (gb [B, 2C]); gap columns -> 0.
extern "C" __global__ void adain_apply_seg(const float* x, float* y, const int* col_item, const float* mean,
                                           const float* rstd, const float* nw, const float* nb, const float* gb,
                                           const float* alpha, int C, int L, int act, float slope) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * L) return;
    int c = (int)(i / L);
    int b = col_item[i % L];
    if (b < 0) {
        y[i] = 0.0f;
        return;
    }
    float v = (x[i] - mean[b * C + c]) * rstd[b * C + c];
    float g1 = 1.0f + gb[(long)b * 2 * C + c];
    v = g1 * (v * nw[c] + nb[c]) + gb[(long)b * 2 * C + C + c];
    if (act == 1) {
        if (v < 0.0f) v = v * slope;
    } else if (act == 2) {
        float a = alpha[c];
        float inv = 1.0f / a;
        float sn = sinf(a * v);
        v = v + inv * (sn * sn);
    }
    y[i] = v;
}

// AdaLayerNorm rows with per-row item index (gb [B, 2d]); gap rows (row_item < 0) -> 0.
extern "C" __global__ void adaln_rows_seg(float* x, const int* row_item, const float* gb, int d, float eps) {
    __shared__ double sh[256];
    int b = row_item[blockIdx.x];
    float* row = x + (long)blockIdx.x * d;
    if (b < 0) {
        for (int j = threadIdx.x; j < d; j += blockDim.x) row[j] = 0.0f;
        return;
    }
    double s = 0.0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s += (double)row[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float mean = (float)(sh[0] / (double)d);
    __syncthreads();
    double v = 0.0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        double c = (double)(row[j] - mean);
        v += c * c;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf((float)(sh[0] / (double)d) + eps);
    const float* g = gb + (long)b * 2 * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float o = (row[j] - mean) * inv;
        row[j] = (1.0f + g[j]) * o + g[d + j];
    }
}

// out [R, d + sd]: row r = concat(x[r], styles[row_item[r]]); gap rows -> 0
extern "C" __global__ void cat_style_rows_seg(const float* x, const float* styles, const int* row_item, float* out,
                                              int R, int d, int sd) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    int w = d + sd;
    if (i >= (long)R * w) return;
    int r = (int)(i / w), j = (int)(i % w);
    int b = row_item[r];
    out[i] = b < 0 ? 0.0f : (j < d ? x[(long)r * d + j] : styles[(long)b * sd + (j - d)]);
}

// gather rows / cols through a source table (src < 0 -> zeros)
extern "C" __global__ void gather_rows(const float* src, const int* tab, float* out, int R, int d) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)R * d) return;
    int s = tab[i / d];
    out[i] = s < 0 ? 0.0f : src[(long)s * d + (i % d)];
}

extern "C" __global__ void gather_cols(const float* src, int Ls, const int* tab, float* out, int C, int L) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * L) return;
    int s = tab[i % L];
    out[i] = s < 0 ? 0.0f : src[(long)(i / L) * Ls + s];
}

// Batched BiLSTM: one cooperative launch for all items. Items occupy rows
// [seg_start[b], seg_start[b] + seg_len[b]) of gx/out ([R, 4H] / [R, 2H]); step s processes
// items with s < len. Same per-item arithmetic as lstm_seq (warp dot order, gate math).
// hbuf [2][B][2H] (double buffered), c_state [B][2H]. grid (H, 2), 128 threads.
extern "C" __global__ void lstm_seq_batched(const float* gx_f, const float* gx_b, const float* whh_f, const float* whh_b,
                                            const float* bhh_f, const float* bhh_b, float* hbuf, float* c_state,
                                            float* out, const int* seg_start, const int* seg_len, int B, int maxT, int H) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int j = blockIdx.x, dir = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* gx = dir == 0 ? gx_f : gx_b;
    const float* whh = dir == 0 ? whh_f : whh_b;
    const float* bhh = dir == 0 ? bhh_f : bhh_b;
    __shared__ float gates[4][64];
    int r = warp * H + j;
    const float* w = whh + (long)r * H;
    for (int step = 0; step < maxT; step++) {
        const float* hcur = hbuf + (long)(step & 1) * B * 2 * H;
        float* hnxt = hbuf + (long)((step + 1) & 1) * B * 2 * H;
        for (int b0 = 0; b0 < B; b0 += 64) {
            int nb = min(64, B - b0);
            for (int bi = 0; bi < nb; bi++) {
                int b = b0 + bi;
                if (step >= seg_len[b]) continue;
                int t = dir == 0 ? step : seg_len[b] - 1 - step;
                const float* h = hcur + (long)b * 2 * H + dir * H;
                float acc = 0.0f;
                for (int k = lane; k < H; k += 32) acc = acc + w[k] * h[k];
                for (int off = 16; off > 0; off >>= 1) acc = acc + __shfl_down_sync(0xffffffff, acc, off);
                if (lane == 0) gates[warp][bi] = gx[(long)(seg_start[b] + t) * 4 * H + r] + (acc + bhh[r]);
            }
            __syncthreads();
            for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) {
                int b = b0 + bi;
                if (step >= seg_len[b]) continue;
                int t = dir == 0 ? step : seg_len[b] - 1 - step;
                float ig = 1.0f / (1.0f + expf(-gates[0][bi]));
                float fg = 1.0f / (1.0f + expf(-gates[1][bi]));
                float gg = tanhf(gates[2][bi]);
                float og = 1.0f / (1.0f + expf(-gates[3][bi]));
                float c = fg * c_state[(long)b * 2 * H + dir * H + j] + ig * gg;
                c_state[(long)b * 2 * H + dir * H + j] = c;
                float hn = og * tanhf(c);
                hnxt[(long)b * 2 * H + dir * H + j] = hn;
                out[(long)(seg_start[b] + t) * 2 * H + dir * H + j] = hn;
            }
            __syncthreads();
        }
        grid.sync();
    }
}

// Reflection pad (1,0) per item on the stage-1 layout: out[s] = in[s+1]; out[s+1+j] = in[s+j].
// Items: seg_start (stage-1 domain), n_in (item length before padding). Other columns -> 0.
extern "C" __global__ void reflect_pad_left1_seg(const float* x, float* y, const int* col_item, const int* seg_start,
                                                 int C, int L) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)C * L) return;
    int o = (int)(i % L);
    int b = col_item[o];
    if (b < 0) {
        y[i] = 0.0f;
        return;
    }
    long row = (i / L) * (long)L;
    int local = o - seg_start[b];
    y[i] = local == 0 ? x[row + seg_start[b] + 1] : x[row + o - 1];
}

// strided variants of the STFT kernels: ld = row stride of the [22, *] spectrum buffer
extern "C" __global__ void stft20_ld(const float* x, int L, float* out, int F, int ld, const double* tw, const float* c_win) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    double buf[20];
    for (int n = 0; n < 20; n++) {
        int p = f * 5 + n - 10;
        int src = p < 0 ? -p : (p >= L ? 2 * L - 2 - p : p);
        buf[n] = (double)(x[src] * c_win[n]);
    }
    for (int k = 0; k < 11; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < 20; n++) {
            re += buf[n] * C_COS(k, n);
            im -= buf[n] * C_SIN(k, n);
        }
        if (k == 0 || k == 10) im = 0.0;
        float rf = (float)re, imf = (float)im;
        out[(long)k * ld + f] = hypotf(rf, imf);
        out[(long)(11 + k) * ld + f] = atan2f(imf, rf);
    }
}

extern "C" __global__ void istft_frames_ld(const float* post, int F, int ld, double* fr, const double* tw, const float* c_win) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    double re[11], im[11];
    for (int k = 0; k < 11; k++) {
        float m = expf(post[(long)k * ld + f]);
        float p = sinf(post[(long)(11 + k) * ld + f]);
        re[k] = (double)(m * cosf(p));
        im[k] = (double)(m * sinf(p));
    }
    for (int n = 0; n < 20; n++) {
        double v = re[0] + re[10] * ((n % 2 == 0) ? 1.0 : -1.0);
        for (int k = 1; k < 10; k++) v += 2.0 * (re[k] * C_COS(k, n) - im[k] * C_SIN(k, n));
        v = v / 20.0;
        fr[(long)f * 20 + n] = v * (double)c_win[n];
    }
}

// Tiled direct Conv1d (PL-004): identical per-output arithmetic order to conv_direct
// (acc = bias; for k: s = sum_ci w*x (ci ascending); acc += s), but each block stages the input
// window for 64 consecutive outputs x all Cin in shared memory and computes 16 output channels,
// so the input is read from DRAM once per block instead of once per output channel.
// grid (ceil(Tout/64), ceil(Cout/16)), block 64. Requires Cin * ((64-1)*stride + K) <= 12288.
#define CT_O 64
#define CT_C 16
extern "C" __global__ void conv_direct_tiled(const float* x, const float* w, const float* b, float* y,
                                             int Cin, int T, int Cout, int K, int stride, int pad, int Tout) {
    extern __shared__ float sh[];
    int o0 = blockIdx.x * CT_O, co0 = blockIdx.y * CT_C;
    int win = (CT_O - 1) * stride + K;
    int base = o0 * stride - pad;
    for (int i = threadIdx.x; i < Cin * win; i += blockDim.x) {
        int ci = i / win, j = i % win;
        int t = base + j;
        sh[i] = (t >= 0 && t < T) ? x[(long)ci * T + t] : 0.0f;
    }
    __syncthreads();
    int o = o0 + threadIdx.x;
    if (o >= Tout) return;
    int ncout = min(CT_C, Cout - co0);
    float acc[CT_C];
    for (int c = 0; c < ncout; c++) acc[c] = b ? b[co0 + c] : 0.0f;
    for (int k = 0; k < K; k++) {
        int t = o * stride + k - pad;
        if (t < 0 || t >= T) continue;  // same skip rule as conv_direct
        int j = threadIdx.x * stride + k;
        float s[CT_C];
        for (int c = 0; c < ncout; c++) s[c] = 0.0f;
        for (int ci = 0; ci < Cin; ci++) {
            float xv = sh[ci * win + j];
            for (int c = 0; c < ncout; c++) s[c] = s[c] + w[((long)(co0 + c) * Cin + ci) * K + k] * xv;
        }
        for (int c = 0; c < ncout; c++) acc[c] = acc[c] + s[c];
    }
    for (int c = 0; c < ncout; c++) y[(long)(co0 + c) * Tout + o] = acc[c];
}

// ---- LEVER PL-008: fused implicit-GEMM dilated conv1d (stride 1), all taps accumulated in
// registers (output written once). Y[co,t] = b[co] + sum_k sum_ci W[ci][k][co] * X[ci, t + k*dil - pad]
// (zero outside [0,T)). f32 throughout; summation order differs from the per-tap GEMM path
// (approximately lossless, not bitwise). Tile: 64 out-channels x 128 time steps, 128 threads, each
// 8 channels x 8 time steps (time interleaved by 16 -> conflict-free smem reads, coalesced stores).
#define IG_BM 64
#define IG_BN 128
#ifndef IG_BK
#define IG_BK 4
#endif
// STRIDED: input length T, output length Tout, output o reads x[o*stride + k*dil - pad]
// (non-strided instantiations compile to exactly the previous code: stride 1, Tout == T).
template <bool RES, bool STRIDED>
__device__ __forceinline__ void conv1d_igemm_body(const float* __restrict__ x, const float* __restrict__ w,
                                                  const float* __restrict__ b, float* __restrict__ y,
                                                  int Cin, int T, int Cout, int K, int dil, int pad,
                                                  int stride_, int Tout_) {
    extern __shared__ float smem[];
    const int stride = STRIDED ? stride_ : 1;
    const int Tout = STRIDED ? Tout_ : T;
    const int xw = STRIDED ? (IG_BN - 1) * stride + (K - 1) * dil + 1 : IG_BN + (K - 1) * dil;
    float* xs = smem;                   // [IG_BK][xw]
    float* ws = smem + IG_BK * xw;      // [IG_BK][K][IG_BM]
    const int t0 = blockIdx.x * IG_BN, co0 = blockIdx.y * IG_BM;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    for (int c0 = 0; c0 < Cin; c0 += IG_BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < IG_BK * xw; i += 128) {
            int c = i / xw, j = i - c * xw;
            int ci = c0 + c, t = t0 * stride - pad + j;
            xs[i] = (ci < Cin && t >= 0 && t < T) ? x[(long)ci * T + t] : 0.0f;
        }
        const int nw = IG_BK * K * IG_BM;
        for (int i = threadIdx.x; i < nw; i += 128) {
            int co = i & (IG_BM - 1), ck = i >> 6;
            int k = ck % K, c = ck / K;
            int ci = c0 + c;
            ws[i] = (ci < Cin && co0 + co < Cout) ? w[((long)ci * K + k) * Cout + co0 + co] : 0.0f;
        }
        __syncthreads();
        for (int c = 0; c < IG_BK; c++) {
            for (int k = 0; k < K; k++) {
                const float* wp = ws + (c * K + k) * IG_BM + ty * 8;
                const float* xp = xs + c * xw + k * dil + tx * stride;
                float a[8], bv[8];
#pragma unroll
                for (int i = 0; i < 8; i++) a[i] = wp[i];
#pragma unroll
                for (int j = 0; j < 8; j++) bv[j] = xp[16 * j * stride];
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[i][j] = __fmaf_rn(a[i], bv[j], acc[i][j]);  // explicit FMA, like the cuBLAS path it replaces
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int co = co0 + ty * 8 + i;
        if (co >= Cout) continue;
        float bias = b ? b[co] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int t = t0 + tx + 16 * j;
            if (t < Tout) {
                long o = (long)co * Tout + t;
                float v = acc[i][j] + bias;
                if (RES) y[o] = y[o] + v;  // residual in place: same order as add_inplace(res, conv)
                else y[o] = v;
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) conv1d_igemm(const float* __restrict__ x, const float* __restrict__ w,
                                                               const float* __restrict__ b, float* __restrict__ y,
                                                               int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_igemm_body<false, false>(x, w, b, y, Cin, T, Cout, K, dil, pad, 1, T);
}
// LEVER PL-012: strided variant (e.g. the generator's noise conv, stride 6).
extern "C" __global__ void __launch_bounds__(128) conv1d_igemm_s(const float* __restrict__ x, const float* __restrict__ w,
                                                                 const float* __restrict__ b, float* __restrict__ y,
                                                                 int Cin, int T, int Cout, int K, int dil, int pad,
                                                                 int stride, int Tout) {
    conv1d_igemm_body<false, true>(x, w, b, y, Cin, T, Cout, K, dil, pad, stride, Tout);
}
// LEVER PL-009: residual epilogue variant (y += conv), same order as add_inplace(res, conv).
extern "C" __global__ void __launch_bounds__(128) conv1d_igemm_res(const float* __restrict__ x, const float* __restrict__ w,
                                                                   const float* __restrict__ b, float* __restrict__ y,
                                                                   int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_igemm_body<true, false>(x, w, b, y, Cin, T, Cout, K, dil, pad, 1, T);
}

// LEVER PL-014 candidate: single-pass per-item channel statistics (sum and sum of squares in
// double, one read of the segment instead of two). Approximately lossless: the variance is
// E[x^2] - mean^2 in double instead of the mean of squared float deviations from the float mean.
extern "C" __global__ void chan_stats_seg1(const float* x, int L, const int* seg_start, const int* seg_len,
                                           float eps, float* mean_out, float* rstd_out, int C) {
    __shared__ double sh[256];
    __shared__ double sq[256];
    int c = blockIdx.x, b = blockIdx.y;
    const float* row = x + (long)c * L + seg_start[b];
    int T = seg_len[b];
    double s = 0.0, s2 = 0.0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        double v = (double)row[t];
        s += v;
        s2 += v * v;
    }
    sh[threadIdx.x] = s;
    sq[threadIdx.x] = s2;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            sh[threadIdx.x] += sh[threadIdx.x + k];
            sq[threadIdx.x] += sq[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double m = sh[0] / (double)T;
        double var = sq[0] / (double)T - m * m;
        if (var < 0.0) var = 0.0;
        mean_out[b * C + c] = (float)m;
        rstd_out[b * C + c] = 1.0f / sqrtf((float)var + eps);
    }
}

// ---- LEVER PL-016 candidate: sliding-window fused conv, templated on (K, DIL) for the generator's
// Snake-block convs. Thread = 8 out-channels x 8 CONTIGUOUS time steps; per input channel the thread
// loads its window x[8tx .. 8tx + 8 + (K-1)DIL) once into registers and reuses it for all K taps.
// Accumulation order per output is unchanged (input-channel-major, tap-minor) -> bitwise identical
// to conv1d_igemm. smem row stride padded to a multiple of 4 floats.
template <int K, int DIL, bool RES>
__device__ __forceinline__ void conv1d_sw_body(const float* __restrict__ x, const float* __restrict__ w,
                                               const float* __restrict__ b, float* __restrict__ y,
                                               int Cin, int T, int Cout, int pad) {
    constexpr int SPAN = 8 + (K - 1) * DIL;
    constexpr int XW = ((IG_BN + (K - 1) * DIL) + 3) & ~3;
    extern __shared__ float smem[];
    float* xs = smem;                  // [IG_BK][XW]
    float* ws = smem + IG_BK * XW;     // [IG_BK][K][IG_BM]
    const int t0 = blockIdx.x * IG_BN, co0 = blockIdx.y * IG_BM;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    for (int c0 = 0; c0 < Cin; c0 += IG_BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < IG_BK * XW; i += 128) {
            int c = i / XW, j = i - c * XW;
            int ci = c0 + c, t = t0 - pad + j;
            xs[i] = (ci < Cin && t >= 0 && t < T && j < IG_BN + (K - 1) * DIL) ? x[(long)ci * T + t] : 0.0f;
        }
        for (int i = threadIdx.x; i < IG_BK * K * IG_BM; i += 128) {
            int co = i & (IG_BM - 1), ck = i >> 6;
            int k = ck % K, c = ck / K;
            int ci = c0 + c;
            ws[i] = (ci < Cin && co0 + co < Cout) ? w[((long)ci * K + k) * Cout + co0 + co] : 0.0f;
        }
        __syncthreads();
#pragma unroll 1
        for (int c = 0; c < IG_BK; c++) {
            float win[SPAN];
            const float* xp = xs + c * XW + 8 * tx;
#pragma unroll
            for (int j = 0; j < SPAN; j++) win[j] = xp[j];
#pragma unroll
            for (int k = 0; k < K; k++) {
                const float* wp = ws + (c * K + k) * IG_BM + ty * 8;
                float a[8];
#pragma unroll
                for (int i = 0; i < 8; i++) a[i] = wp[i];
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[i][j] = __fmaf_rn(a[i], win[k * DIL + j], acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int co = co0 + ty * 8 + i;
        if (co >= Cout) continue;
        float bias = b ? b[co] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int t = t0 + 8 * tx + j;
            if (t < T) {
                long o = (long)co * T + t;
                float v = acc[i][j] + bias;
                if (RES) y[o] = y[o] + v;
                else y[o] = v;
            }
        }
    }
}
#define SW_KERNEL(K, D)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) conv1d_sw_k##K##d##D(const float* __restrict__ x,     \
        const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y, int Cin, int T,    \
        int Cout, int pad) { conv1d_sw_body<K, D, false>(x, w, b, y, Cin, T, Cout, pad); }                  \
    extern "C" __global__ void __launch_bounds__(128) conv1d_sw_res_k##K##d##D(const float* __restrict__ x, \
        const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y, int Cin, int T,    \
        int Cout, int pad) { conv1d_sw_body<K, D, true>(x, w, b, y, Cin, T, Cout, pad); }
SW_KERNEL(3, 1) SW_KERNEL(3, 3) SW_KERNEL(3, 5)
SW_KERNEL(7, 1) SW_KERNEL(7, 3) SW_KERNEL(7, 5)
SW_KERNEL(11, 1) SW_KERNEL(11, 3) SW_KERNEL(11, 5)

// ======================= PHASE 2 (experiment/reduced-precision branch) =======================
// Reduced-precision helpers. f32 [C][T] activations -> transposed low-precision [T][C] operands
// (per-tap GEMM offsets then become multiples of C: aligned TN tensor-core GEMMs).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void lp_transpose_f16(const float* __restrict__ x, __half* __restrict__ y, int C, int T) {
    __shared__ float tile[32][33];
    int t = blockIdx.x * 32 + threadIdx.x, c = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        int cc = c + k;
        tile[threadIdx.y + k][threadIdx.x] = (cc < C && t < T) ? x[(long)cc * T + t] : 0.0f;
    }
    __syncthreads();
    int tt = blockIdx.x * 32 + threadIdx.y, c2 = blockIdx.y * 32 + threadIdx.x;
    for (int k = 0; k < 32; k += 8) {
        int t3 = tt + k;
        if (t3 < T && c2 < C) y[(long)t3 * C + c2] = __float2half_rn(tile[threadIdx.x][threadIdx.y + k]);
    }
}
extern "C" __global__ void lp_transpose_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int C, int T) {
    __shared__ float tile[32][33];
    int t = blockIdx.x * 32 + threadIdx.x, c = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        int cc = c + k;
        tile[threadIdx.y + k][threadIdx.x] = (cc < C && t < T) ? x[(long)cc * T + t] : 0.0f;
    }
    __syncthreads();
    int tt = blockIdx.x * 32 + threadIdx.y, c2 = blockIdx.y * 32 + threadIdx.x;
    for (int k = 0; k < 32; k += 8) {
        int t3 = tt + k;
        if (t3 < T && c2 < C) y[(long)t3 * C + c2] = __float2bfloat16_rn(tile[threadIdx.x][threadIdx.y + k]);
    }
}
// Plain element-wise conversions (weights at load; row-major operands for linear layers).
extern "C" __global__ void lp_cvt_f16(const float* __restrict__ x, __half* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
extern "C" __global__ void lp_cvt_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}
// |x| max over n elements into *out (one block per call; grid-stride; deterministic: fixed order
// per thread + fixed tree).
extern "C" __global__ void lp_absmax(const float* __restrict__ x, long n, float* __restrict__ out) {
    __shared__ float sh[1024];
    float m = 0.0f;
    for (long i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, fabsf(x[i]));
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = sh[0];
}
// INT8 symmetric quantization with a device-resident per-tensor scale s = absmax/127 (s>0),
// transposing [C][T] f32 -> [T][C] int8; values rounded to nearest, clamped to [-127, 127].
extern "C" __global__ void lp_transpose_q8(const float* __restrict__ x, signed char* __restrict__ y, int C, int T,
                                           const float* __restrict__ absmax) {
    __shared__ float tile[32][33];
    float am = absmax[0];
    float inv = am > 0.0f ? 127.0f / am : 0.0f;
    int t = blockIdx.x * 32 + threadIdx.x, c = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        int cc = c + k;
        tile[threadIdx.y + k][threadIdx.x] = (cc < C && t < T) ? x[(long)cc * T + t] : 0.0f;
    }
    __syncthreads();
    int tt = blockIdx.x * 32 + threadIdx.y, c2 = blockIdx.y * 32 + threadIdx.x;
    for (int k = 0; k < 32; k += 8) {
        int t3 = tt + k;
        if (t3 < T && c2 < C) {
            float q = rintf(tile[threadIdx.x][threadIdx.y + k] * inv);
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            y[(long)t3 * C + c2] = (signed char)q;
        }
    }
}
// y[co][t] = bias[co] + acc[co][t] * (absmax/127) * wscale[co]   (acc int32, row stride ldc)
extern "C" __global__ void lp_dequant(const int* __restrict__ acc, int ldc, const float* __restrict__ wscale,
                                      const float* __restrict__ absmax, const float* __restrict__ b,
                                      float* __restrict__ y, int Cout, int T) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)Cout * T) return;
    int co = (int)(i / T), t = (int)(i - (long)co * T);
    float sx = absmax[0] / 127.0f;
    float v = (float)acc[(long)co * ldc + t] * (sx * wscale[co]);
    y[i] = (b ? b[co] : 0.0f) + v;
}

// ---- PHASE 2: fused tensor-core (WMMA) conv1d, stride 1, any K / dil. Tile 64 out-ch x 64 time,
// 4 warps (warp w: out-ch rows [16w, 16w+16), 4 accumulator fragments over time).
// Input f32 [Cin][T] is converted on the fly into a TRANSPOSED smem window [TW][XLD] (time rows,
// input-channel columns) so each tap's B operand is a row offset (32-byte aligned for any k*dil).
// Weights: low-precision [K][Cout][Cin] staged per 16-channel chunk as [K][64][16].
// HT = __half | __nv_bfloat16 (f32 accumulate) | signed char (int32 accumulate, dynamic per-tensor
// activation scale *absmax/127, per-out-channel weight scale wscale[co]).
#include <mma.h>
template <typename HT> struct WmmaAcc { typedef float T; };
template <> struct WmmaAcc<signed char> { typedef int T; };
template <typename HT> __device__ __forceinline__ HT to_lp(float v, float inv);
template <> __device__ __forceinline__ __half to_lp<__half>(float v, float) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 to_lp<__nv_bfloat16>(float v, float) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ signed char to_lp<signed char>(float v, float inv) {
    float q = rintf(v * inv);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return (signed char)q;
}
#ifndef WMMA_TN
#define WMMA_TN 64
#endif
template <typename HT, bool RES = false>
__device__ __forceinline__ void conv1d_wmma_body(const float* __restrict__ x, const HT* __restrict__ w,
                                                 const float* __restrict__ b, float* __restrict__ y,
                                                 int Cin, int T, int Cout, int K, int dil, int pad,
                                                 const float* __restrict__ absmax, const float* __restrict__ wscale) {
    using namespace nvcuda;
    typedef typename WmmaAcc<HT>::T AT;
    constexpr int XLD = sizeof(HT) == 1 ? 32 : 16;  // smem row stride (elements): 32-byte rows
    extern __shared__ __align__(128) unsigned char smem_raw[];
    constexpr int TN = WMMA_TN, NF = TN / 16;
    const int TW = TN + (K - 1) * dil;
    HT* xs = (HT*)smem_raw;                    // [TW][XLD]
    HT* ws = xs + TW * XLD;                    // [K][64][16]
    const int t0 = blockIdx.x * TN, co0 = blockIdx.y * 64, warp = threadIdx.x >> 5;
    float inv = 0.0f;
    if (absmax) {
        float am = absmax[0];
        inv = am > 0.0f ? 127.0f / am : 0.0f;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, AT> acc[NF];
#pragma unroll
    for (int f = 0; f < NF; f++) wmma::fill_fragment(acc[f], (AT)0);
    for (int c0 = 0; c0 < Cin; c0 += 16) {
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * TW; i += 128) {
            int ci = i / TW, tt = i - ci * TW;
            int t = t0 - pad + tt;
            float v = (c0 + ci < Cin && t >= 0 && t < T) ? x[(long)(c0 + ci) * T + t] : 0.0f;
            xs[tt * XLD + ci] = to_lp<HT>(v, inv);
        }
        for (int i = threadIdx.x; i < K * 64 * 16; i += 128) {
            int ci = i & 15, co = (i >> 4) & 63, k = i >> 10;
            ws[i] = (co0 + co < Cout && c0 + ci < Cin) ? w[((long)k * Cout + co0 + co) * Cin + c0 + ci] : (HT)0;
        }
        __syncthreads();
        for (int k = 0; k < K; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HT, wmma::row_major> a;
            wmma::load_matrix_sync(a, ws + (k * 64 + warp * 16) * 16, 16);
#pragma unroll
            for (int f = 0; f < NF; f++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HT, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, xs + (f * 16 + k * dil) * XLD, XLD);
                wmma::mma_sync(acc[f], a, bf, acc[f]);
            }
        }
    }
    __syncthreads();
    AT* cs = (AT*)smem_raw;  // [64][TN] staging (reuses the operand smem)
#pragma unroll
    for (int f = 0; f < NF; f++) wmma::store_matrix_sync(cs + (warp * 16) * TN + f * 16, acc[f], TN, wmma::mem_row_major);
    __syncthreads();
    float sx = absmax ? absmax[0] / 127.0f : 1.0f;
    for (int i = threadIdx.x; i < 64 * TN; i += 128) {
        int co = co0 + i / TN, t = t0 + i % TN;
        if (co < Cout && t < T) {
            float v = sizeof(HT) == 1 ? (float)cs[i] * (sx * wscale[co]) : (float)cs[i];
            long o = (long)co * T + t;
            float r = (b ? b[co] : 0.0f) + v;
            if (RES) y[o] = y[o] + r;  // residual in place (y already holds the residual branch input)
            else y[o] = r;
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_f16(const float* x, const __half* w, const float* b, float* y,
                                                                  int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_wmma_body<__half>(x, w, b, y, Cin, T, Cout, K, dil, pad, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_bf16(const float* x, const __nv_bfloat16* w, const float* b, float* y,
                                                                   int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_wmma_body<__nv_bfloat16>(x, w, b, y, Cin, T, Cout, K, dil, pad, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_s8(const float* x, const signed char* w, const float* b, float* y,
                                                                 int Cin, int T, int Cout, int K, int dil, int pad,
                                                                 const float* absmax, const float* wscale) {
    conv1d_wmma_body<signed char>(x, w, b, y, Cin, T, Cout, K, dil, pad, absmax, wscale);
}
// Multi-block |x| max (out must be zeroed first): block maxima combined with atomicMax on the float
// bit pattern (monotone for non-negative floats) -> deterministic result.
extern "C" __global__ void lp_absmax_mb(const float* __restrict__ x, long n, float* __restrict__ out) {
    __shared__ float sh[256];
    float m = 0.0f;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) m = fmaxf(m, fabsf(x[i]));
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + k]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((int*)out, __float_as_int(sh[0]));
}

// PHASE 2: residual-epilogue variants (y += conv), used by the Snake blocks (mask not needed: the
// AdaIN output already has zero gaps).
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_f16_res(const float* x, const __half* w, const float* b, float* y,
                                                                      int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_wmma_body<__half, true>(x, w, b, y, Cin, T, Cout, K, dil, pad, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_bf16_res(const float* x, const __nv_bfloat16* w, const float* b, float* y,
                                                                       int Cin, int T, int Cout, int K, int dil, int pad) {
    conv1d_wmma_body<__nv_bfloat16, true>(x, w, b, y, Cin, T, Cout, K, dil, pad, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) conv1d_wmma_s8_res(const float* x, const signed char* w, const float* b, float* y,
                                                                     int Cin, int T, int Cout, int K, int dil, int pad,
                                                                     const float* absmax, const float* wscale) {
    conv1d_wmma_body<signed char, true>(x, w, b, y, Cin, T, Cout, K, dil, pad, absmax, wscale);
}
