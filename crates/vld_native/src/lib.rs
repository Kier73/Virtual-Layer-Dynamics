//! Optional C-ABI native accelerator kernels for Virtual Layer Dynamics (`vld.accel.native`).
//!
//! Exposes deterministic, zero-dependency Rust kernels:
//!   1. `vld_lcg_jump_u64`: 128-bit overflow-safe binary monoid affine jump in `O(log n)`.
//!   2. `vld_bilinear_periodic_matvec_f64`: sufficient-statistic Virtual Layer matrix-vector contraction.
//!   3. `vld_rms_norm_f32`: contiguous row-wise FP32 Root-Mean-Square Normalization.

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

/// Evaluate row-wise FP32 RMSNorm: `out[r, c] = x[r, c] * rsqrt(mean(x[r]^2) + eps) * weight[c]`.
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
    let x = if rows > 0 { slice::from_raw_parts(x_ptr, rows * cols) } else { &[] };
    let w = slice::from_raw_parts(w_ptr, cols);
    let out = if rows > 0 { slice::from_raw_parts_mut(out_ptr, rows * cols) } else { &mut [] };

    let inv_cols = 1.0_f32 / (cols as f32);
    for r in 0..rows {
        let row_in = &x[r * cols..(r + 1) * cols];
        let row_out = &mut out[r * cols..(r + 1) * cols];
        let mut sum_sq: f32 = 0.0;
        for &v in row_in {
            sum_sq += v * v;
        }
        let scale = 1.0_f32 / (sum_sq * inv_cols + eps).sqrt();
        for c in 0..cols {
            row_out[c] = row_in[c] * scale * w[c];
        }
    }
    0
}
