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
