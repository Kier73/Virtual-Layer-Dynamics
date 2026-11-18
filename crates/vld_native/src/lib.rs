//! Optional C-ABI native accelerator kernels for Virtual Layer Dynamics (`vld.accel.native`).
//!
//! Exposes deterministic, zero-dependency Rust kernels:
//!   1. `vld_lcg_jump_u64`: 128-bit overflow-safe binary monoid affine jump in `O(log n)`.
//!   2. `vld_bilinear_periodic_matvec_f64`: sufficient-statistic Virtual Layer matrix-vector contraction.
//!   3. `vld_rms_norm_f32`: contiguous row-wise FP32 Root-Mean-Square Normalization.
//!   4. `vld_dequant_f8_e4m3_ue8m0_block_f32`: block-scaled OCP `F8_E4M3` + `F8_E8M0` (`UE8M0`) dequantizer.
//!   5. `vld_sinkhorn_birkhoff_f32`: Manifold-Constrained Hyper-Connections (`mHC`) Birkhoff Sinkhorn-Knopp kernel.

use std::slice;

/// Advance `x_{t+1} = (a * x_t + c) mod m` by `steps` coordinates in `O(log steps)` using `u128` arithmetic.
#[no_mangle]
pub extern "C" fn vld_lcg_jump_u64(a: u64, c: u64, modulus: u64, x0: u64, steps: u64) -> u64 {
    if modulus < 2 {
        return 0;
    }
    let m = modulus as u128;
    let mut acc_a: u128 = 1;
    let mut acc_b: u128 = 0;
    let mut cur_a: u128 = (a as u128) % m;
    let mut cur_b: u128 = (c as u128) % m;
    let mut n = steps;
    while n > 0 {
        if (n & 1) == 1 {
            acc_a = (cur_a * acc_a) % m;
            acc_b = (cur_a * acc_b + cur_b) % m;
        }
        cur_b = (cur_a * cur_b + cur_b) % m;
        cur_a = (cur_a * cur_a) % m;
        n >>= 1;
    }
    ((acc_a * ((x0 as u128) % m) + acc_b) % m) as u64
}

/// Evaluate `y_i = (alpha + beta*i)*S0 + (gamma + delta*i)*S1 + sum_{q=0..period-1} orbit[(q - stride*i) mod period] * folded_x[q]`
/// in `O(rows + cols + rows * period)` operations without materializing `rows * cols` matrix elements.
#[no_mangle]
pub unsafe extern "C" fn vld_bilinear_periodic_matvec_f64(
    rows: usize,
    cols: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    orbit_ptr: *const f64,
    period: usize,
    stride: usize,
    x_ptr: *const f64,
    out_ptr: *mut f64,
) -> i32 {
    if (cols > 0 && x_ptr.is_null()) || (rows > 0 && out_ptr.is_null()) {
        return -1;
    }
    if period > 0 && orbit_ptr.is_null() {
        return -2;
    }
    let x = if cols > 0 { slice::from_raw_parts(x_ptr, cols) } else { &[] };
    let out = if rows > 0 { slice::from_raw_parts_mut(out_ptr, rows) } else { &mut [] };
    let orbit = if period > 0 { slice::from_raw_parts(orbit_ptr, period) } else { &[] };

    let mut s0: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut folded = vec![0.0_f64; period];

    for (j, &xj) in x.iter().enumerate() {
        s0 += xj;
        s1 += (j as f64) * xj;
        if period > 0 {
            folded[j % period] += xj;
        }
    }

    for (i, slot) in out.iter_mut().enumerate() {
        let fi = i as f64;
        let mut acc = (alpha + beta * fi) * s0 + (gamma + delta * fi) * s1;
        if period > 0 {
            let p128 = period as u128;
            let shift = (((stride % period) as u128 * (i as u128)) % p128) as usize;
            for q in 0..period {
                let idx = (q + period - shift) % period;
                acc += orbit[idx] * folded[q];
            }
        }
        *slot = acc;
    }
    0
}

/// Evaluate row-wise FP32 RMSNorm: `out[r, c] = x[r, c] * rsqrt(mean_f64(x[r]^2) + eps) * weight[c]`.
#[no_mangle]
pub unsafe extern "C" fn vld_rms_norm_f32(
    rows: usize,
    cols: usize,
    x_ptr: *const f32,
    w_ptr: *const f32,
    eps: f32,
    out_ptr: *mut f32,
) -> i32 {
    if cols == 0 || (rows > 0 && (x_ptr.is_null() || out_ptr.is_null())) || w_ptr.is_null() {
        return -1;
    }
    if !eps.is_finite() || eps < 0.0 {
        return -2;
    }
    let x = if rows > 0 { slice::from_raw_parts(x_ptr, rows * cols) } else { &[] };
    let w = slice::from_raw_parts(w_ptr, cols);
    if w.iter().any(|v| !v.is_finite()) {
        return -2;
    }
    let out = if rows > 0 { slice::from_raw_parts_mut(out_ptr, rows * cols) } else { &mut [] };

    let inv_cols = 1.0_f64 / (cols as f64);
    let eps64 = eps as f64;
    for r in 0..rows {
        let row_in = &x[r * cols..(r + 1) * cols];
        let row_out = &mut out[r * cols..(r + 1) * cols];
        let mut sum_sq: f64 = 0.0;
        for &v in row_in {
            if !v.is_finite() {
                return -2;
            }
            let dv = v as f64;
            sum_sq += dv * dv;
        }
        let scale = 1.0_f64 / (sum_sq * inv_cols + eps64).sqrt();
        for c in 0..cols {
            let y = ((row_in[c] as f64) * scale * (w[c] as f64)) as f32;
            if !y.is_finite() {
                return -2;
            }
            row_out[c] = y;
        }
    }
    0
}

#[inline]
fn decode_f8_e4m3_byte(b: u8) -> Option<f64> {
    let sign = if (b & 0x80) != 0 { -1.0_f64 } else { 1.0_f64 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f64;
    if exp == 15 && (b & 0x07) == 7 {
        return None;
    }
    if exp == 0 {
        Some(sign * man * (2.0_f64).powi(-9))
    } else {
        Some(sign * (1.0_f64 + man * 0.125_f64) * (2.0_f64).powi(exp - 7))
    }
}

#[inline]
fn decode_ue8m0_byte(u: u8) -> Option<f64> {
    if u == 255 {
        return None;
    }
    Some(f64::from_bits(((u as u64) + 896) << 52))
}

/// Dequantize a 2D `F8_E4M3` weight matrix `[rows, cols]` with `F8_E8M0` (`UE8M0`) block scales `[scale_rows, scale_cols]`.
#[no_mangle]
pub unsafe extern "C" fn vld_dequant_f8_e4m3_ue8m0_block_f32(
    rows: usize,
    cols: usize,
    block_r: usize,
    block_c: usize,
    w_ptr: *const u8,
    s_ptr: *const u8,
    scale_cols: usize,
    out_ptr: *mut f32,
) -> i32 {
    if rows == 0 || cols == 0 || block_r == 0 || block_c == 0 || scale_cols == 0 {
        return -1;
    }
    if w_ptr.is_null() || s_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let scale_rows = (rows + block_r - 1) / block_r;
    let req_scale_cols = (cols + block_c - 1) / block_c;
    if scale_cols != req_scale_cols {
        return -1;
    }
    let w = slice::from_raw_parts(w_ptr, rows * cols);
    let s = slice::from_raw_parts(s_ptr, scale_rows * scale_cols);
    let out = slice::from_raw_parts_mut(out_ptr, rows * cols);

    let mut lut_e4m3 = [0.0_f64; 256];
    for idx in 0..256usize {
        if let Some(v) = decode_f8_e4m3_byte(idx as u8) {
            lut_e4m3[idx] = v;
        }
    }

    let mut scales_f64 = vec![0.0_f64; scale_rows * scale_cols];
    for (idx, &sb) in s.iter().enumerate() {
        match decode_ue8m0_byte(sb) {
            Some(sv) => scales_f64[idx] = sv,
            None => return -2,
        }
    }

    for r in 0..rows {
        let br = r / block_r;
        let row_w = &w[r * cols..(r + 1) * cols];
        let row_out = &mut out[r * cols..(r + 1) * cols];
        let row_s = &scales_f64[br * scale_cols..(br + 1) * scale_cols];
        for c in 0..cols {
            let wb = row_w[c];
            if (wb & 0x7F) == 0x7F {
                return -2;
            }
            let val = (lut_e4m3[wb as usize] * row_s[c / block_c]) as f32;
            if !val.is_finite() {
                return -2;
            }
            row_out[c] = val;
        }
    }
    0
}

/// Evaluate DeepSeek-V4.1 Manifold-Constrained Hyper-Connections (`mHC`) Birkhoff Sinkhorn-Knopp projection.
#[no_mangle]
pub unsafe extern "C" fn vld_sinkhorn_birkhoff_f32(
    n_tokens: usize,
    hc: usize,
    sinkhorn_iters: usize,
    eps: f32,
    mixes_ptr: *const f32,
    scale_ptr: *const f32,
    base_ptr: *const f32,
    pre_out_ptr: *mut f32,
    post_out_ptr: *mut f32,
    comb_out_ptr: *mut f32,
) -> i32 {
    if hc == 0 || sinkhorn_iters == 0 {
        return -1;
    }
    if scale_ptr.is_null() || base_ptr.is_null() {
        return -1;
    }
    if n_tokens > 0
        && (mixes_ptr.is_null() || pre_out_ptr.is_null() || post_out_ptr.is_null() || comb_out_ptr.is_null())
    {
        return -1;
    }
    if !eps.is_finite() || eps < 0.0 {
        return -2;
    }
    let mix_hc = (2 + hc) * hc;
    let mixes = if n_tokens > 0 { slice::from_raw_parts(mixes_ptr, n_tokens * mix_hc) } else { &[] };
    let scale = slice::from_raw_parts(scale_ptr, 3);
    let base = slice::from_raw_parts(base_ptr, mix_hc);
    if scale.iter().any(|v| !v.is_finite()) || base.iter().any(|v| !v.is_finite()) {
        return -2;
    }
    let pre_out = if n_tokens > 0 { slice::from_raw_parts_mut(pre_out_ptr, n_tokens * hc) } else { &mut [] };
    let post_out = if n_tokens > 0 { slice::from_raw_parts_mut(post_out_ptr, n_tokens * hc) } else { &mut [] };
    let comb_out = if n_tokens > 0 {
        slice::from_raw_parts_mut(comb_out_ptr, n_tokens * hc * hc)
    } else {
        &mut []
    };

    let s0 = scale[0] as f64;
    let s1 = scale[1] as f64;
    let s2 = scale[2] as f64;
    let eps64 = eps as f64;
    let mut comb = vec![0.0_f64; hc * hc];
    let mut col_sum = vec![0.0_f64; hc];

    for i in 0..n_tokens {
        let m_row = &mixes[i * mix_hc..(i + 1) * mix_hc];
        if m_row.iter().any(|v| !v.is_finite()) {
            return -2;
        }
        for j in 0..hc {
            let z0 = (m_row[j] as f64) * s0 + (base[j] as f64);
            let sig0 = 1.0_f64 / (1.0_f64 + (-z0).exp());
            pre_out[i * hc + j] = (sig0 + eps64) as f32;

            let z1 = (m_row[hc + j] as f64) * s1 + (base[hc + j] as f64);
            let sig1 = 1.0_f64 / (1.0_f64 + (-z1).exp());
            post_out[i * hc + j] = (2.0_f64 * sig1) as f32;
        }

        for j in 0..hc {
            let mut rmax = f64::NEG_INFINITY;
            for k in 0..hc {
                let idx = 2 * hc + j * hc + k;
                let z = (m_row[idx] as f64) * s2 + (base[idx] as f64);
                comb[j * hc + k] = z;
                if z > rmax {
                    rmax = z;
                }
            }
            let mut rsum = 0.0_f64;
            for k in 0..hc {
                let ev = (comb[j * hc + k] - rmax).exp();
                comb[j * hc + k] = ev;
                rsum += ev;
            }
            for k in 0..hc {
                comb[j * hc + k] = comb[j * hc + k] / rsum + eps64;
            }
        }

        for k in 0..hc {
            let mut csum = 0.0_f64;
            for j in 0..hc {
                csum += comb[j * hc + k];
            }
            col_sum[k] = csum + eps64;
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= col_sum[k];
            }
        }

        for _ in 0..sinkhorn_iters.saturating_sub(1) {
            for j in 0..hc {
                let mut rsum = 0.0_f64;
                for k in 0..hc {
                    rsum += comb[j * hc + k];
                }
                let denom = rsum + eps64;
                for k in 0..hc {
                    comb[j * hc + k] /= denom;
                }
            }
            for k in 0..hc {
                let mut csum = 0.0_f64;
                for j in 0..hc {
                    csum += comb[j * hc + k];
                }
                col_sum[k] = csum + eps64;
            }
            for j in 0..hc {
                for k in 0..hc {
                    comb[j * hc + k] /= col_sum[k];
                }
            }
        }

        for idx in 0..(hc * hc) {
            comb_out[i * hc * hc + idx] = comb[idx] as f32;
        }
    }
    0
}
