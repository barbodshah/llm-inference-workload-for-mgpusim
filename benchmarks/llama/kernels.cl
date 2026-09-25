// SPDX-License-Identifier: GPL-3.0-only
// Adapted from Uzair-90/llama-inference-hip. See NOTICE and LICENSE.
/*
 * hip_kernels.cpp
 * Implementation of all HIP GPU kernels for Llama2 transformer hot-paths.
 * Ported from karpathy/llama2.c (https://github.com/karpathy/llama2.c)
 */






// ---------------------------------------------------------------------------
// matmul: W(d,n) @ x(n) -> xout(d)
// Each GPU thread computes one row of the output.
// This is the single hottest function in the transformer, accounting for
// the majority of FLOPs. With HBM-backed wavefronts the memory bandwidth
// utilization is dramatically higher than single-core DRAM.
// ---------------------------------------------------------------------------
__kernel void kernel_matmul(__global float* xout, __global const float* x, __global const float* w, int n, int d) {
    int i = get_global_id(0);
    if (i >= d) return;

    float val = 0.0f;
    __global const float* row = w + i * n;
    for (int j = 0; j < n; j++) {
        val += row[j] * x[j];
    }
    xout[i] = val;
}

// ---------------------------------------------------------------------------
// RMSNorm: o[j] = weight[j] * ( x[j] / sqrt(mean(x^2) + eps) )
// One block, threads cooperate to accumulate the sum-of-squares via shared mem.
// ---------------------------------------------------------------------------
__kernel void kernel_rmsnorm(__global float* o, __global const float* x, __global const float* weight, int size) {
    __local float sdata[256];

    int tid = get_local_id(0);
    float partial = 0.0f;
    for (int i = tid; i < size; i += get_local_size(0)) {
        partial += x[i] * x[i];
    }
    sdata[tid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Tree reduction
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    float ss = sdata[0] / size + 1e-5f;
    float inv_rms = 1.0f / sqrt(ss);

    for (int i = tid; i < size; i += get_local_size(0)) {
        o[i] = weight[i] * (inv_rms * x[i]);
    }
}

// ---------------------------------------------------------------------------
// Softmax in-place
// Two-pass: find max (for numerical stability), then exp + normalize.
// ---------------------------------------------------------------------------
__kernel void kernel_softmax(__global float* x, int size) {
    __local float sdata[256];

    int tid = get_local_id(0);

    // Pass 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < size; i += get_local_size(0)) {
        local_max = fmax(local_max, x[i]);
    }
    sdata[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    float max_val = sdata[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += get_local_size(0)) {
        x[i] = exp(x[i] - max_val);
        local_sum += x[i];
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    float sum = sdata[0];

    for (int i = tid; i < size; i += get_local_size(0)) {
        x[i] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// Each thread rotates one (real, imag) pair in the query and/or key.
// ---------------------------------------------------------------------------
__kernel void kernel_rope(__global float* q, __global float* k, int dim, int kv_dim, int head_size, int pos) {
    int i = (get_global_id(0)) * 2;
    if (i >= dim) return;

    int head_dim = i % head_size;
    float freq = 1.0f / pow(10000.0f, (float)head_dim / (float)head_size);
    float val   = pos * freq;
    float fcr   = cos(val);
    float fci   = sin(val);

    // Rotate query
    float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // Rotate key only for positions within the kv dimension
    if (i < kv_dim) {
        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

// ---------------------------------------------------------------------------
// SwiGLU non-linearity: hb[i] = silu(hb[i]) * hb2[i]
// Trivially parallel over hidden_dim.
// ---------------------------------------------------------------------------
__kernel void kernel_swiglu(__global float* hb, __global const float* hb2, int hidden_dim) {
    int i = get_global_id(0);
    if (i >= hidden_dim) return;
    float v = hb[i];
    v *= (1.0f / (1.0f + exp(-v)));  // silu
    hb[i] = v * hb2[i];
}

// ---------------------------------------------------------------------------
// Element-wise accumulation: x[i] += y[i]
// ---------------------------------------------------------------------------
__kernel void kernel_add(__global float* x, __global const float* y, int size) {
    int i = get_global_id(0);
    if (i >= size) return;
    x[i] += y[i];
}

// ---------------------------------------------------------------------------
// Multi-head attention
// Block layout: one block per attention head.
// Each block:
//   1. Computes all QK dot-products (scores) for the current head.
//   2. Applies softmax over the [0..pos] scores.
//   3. Accumulates the value-weighted output into xb.
// ---------------------------------------------------------------------------
__kernel void kernel_attention(
    __global float*       xb,
    __global const float* q,
    __global const float* key_cache,
    __global const float* value_cache,
    __global float*       att,
    int          head_size,
    int          kv_dim,
    int          kv_mul,
    int          seq_len,
    int          pos,
    int          loff)
{
    int h = get_group_id(0);            // each block handles one head
    __global const float* qh = q + h * head_size;
    __global float* atth = att + h * seq_len;

    // --- Step 1: compute attention scores ---
    for (int t = get_local_id(0); t <= pos; t += get_local_size(0)) {
        __global const float* kh = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += qh[i] * kh[i];
        }
        atth[t] = score / sqrt((float)head_size);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // --- Step 2: in-block softmax over atth[0..pos] ---
    // Find max
    __local float sdata[128];
    float local_max = -1e30f;
    for (int t = get_local_id(0); t <= pos; t += get_local_size(0)) {
        local_max = fmax(local_max, atth[t]);
    }
    sdata[get_local_id(0)] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (get_local_id(0) < stride) sdata[get_local_id(0)] = fmax(sdata[get_local_id(0)], sdata[get_local_id(0) + stride]);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    float max_val = sdata[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_sum = 0.0f;
    for (int t = get_local_id(0); t <= pos; t += get_local_size(0)) {
        atth[t] = exp(atth[t] - max_val);
        local_sum += atth[t];
    }
    sdata[get_local_id(0)] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (get_local_id(0) < stride) sdata[get_local_id(0)] += sdata[get_local_id(0) + stride];
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / sdata[0];
    for (int t = get_local_id(0); t <= pos; t += get_local_size(0)) {
        atth[t] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // --- Step 3: weighted sum of values ---
    __global float* xbh = xb + h * head_size;
    for (int i = get_local_id(0); i < head_size; i += get_local_size(0)) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            __global const float* vh = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            acc += atth[t] * vh[i];
        }
        xbh[i] = acc;
    }
}


