// SPDX-License-Identifier: MIT
// Portable GCN3 research baseline: query tiles share an 8-token K/V tile.
// FP32 online softmax; no global attention-score matrix. Not FlashAttention.
#define KEY_TILE 8
#define DIM_TILE 64

#define ATTENTION(NAME, QT) \
__attribute__((reqd_work_group_size(QT * 64, 1, 1))) \
__kernel void NAME(__global float *out, __global const float *q, \
    __global const float *keys, __global const float *values, \
    __global const int *lengths, int queries, int heads, int kv_heads, \
    int dim, int stride) { \
    int lid = get_local_id(0), lane = lid % 64, row = lid / 64; \
    int tiles = (queries + QT - 1) / QT; \
    int group = get_group_id(0), tile = group % tiles; \
    int head = (group / tiles) % heads, request = group / (tiles * heads); \
    int query = tile * QT + row, length = lengths[request]; \
    int limit = length - queries + query + 1; \
    int kv_head = head / (heads / kv_heads); \
    __local volatile float tq[QT * DIM_TILE], tk[KEY_TILE * DIM_TILE]; \
    __local volatile float tv[KEY_TILE * DIM_TILE], scores[QT * KEY_TILE]; \
    __local volatile float alpha[QT], denom[QT]; \
    tq[lid] = query < queries && lane < dim ? \
        q[((request * queries + query) * heads + head) * dim + lane] : 0.0f; \
    float acc = 0.0f, maximum = -INFINITY, total = 0.0f; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int visible_end = min(length, length - queries + (tile + 1) * QT); \
    for (int base = 0; base < visible_end; base += KEY_TILE) { \
        for (int i = lid; i < KEY_TILE * DIM_TILE; i += QT * 64) { \
            int token = base + i / DIM_TILE, d = i % DIM_TILE; \
            int index = ((request * stride + token) * kv_heads + kv_head) * dim + d; \
            tk[i] = token < length && d < dim ? keys[index] : 0.0f; \
            tv[i] = token < length && d < dim ? values[index] : 0.0f; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (lane < KEY_TILE) { \
            float dot = 0.0f; \
            for (int d = 0; d < dim; d++) dot += tq[row * DIM_TILE + d] * tk[lane * DIM_TILE + d]; \
            scores[row * KEY_TILE + lane] = base + lane < limit && base + lane < length && query < queries ? dot * native_rsqrt((float)dim) : -INFINITY; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (lane == 0) { \
            float tile_max = -INFINITY; \
            for (int j = 0; j < KEY_TILE; j++) tile_max = fmax(tile_max, scores[row * KEY_TILE + j]); \
            float next_max = fmax(maximum, tile_max); \
            float a = maximum > -INFINITY ? native_exp(maximum - next_max) : 0.0f; \
            float sum = 0.0f; \
            for (int j = 0; j < KEY_TILE; j++) { \
                float s = scores[row * KEY_TILE + j]; \
                float w = s > -INFINITY ? native_exp(s - next_max) : 0.0f; \
                scores[row * KEY_TILE + j] = w; sum += w; \
            } \
            total = total * a + sum; maximum = next_max; \
            alpha[row] = a; denom[row] = total; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        acc *= alpha[row]; \
        for (int j = 0; j < KEY_TILE; j++) acc += scores[row * KEY_TILE + j] * tv[j * DIM_TILE + lane]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (query < queries && lane < dim) \
        out[((request * queries + query) * heads + head) * dim + lane] = acc * native_recip(denom[row]); \
}

ATTENTION(attention_prefill, 4)
ATTENTION(attention_decode, 1)

// Two independent instances of this kernel exercise existing queue/dispatch
// concurrency. Unsigned wraparound has an exact CPU reference.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void concurrency_probe(__global uint *out, uint rounds, uint seed) {
    int index = get_group_id(0) * 64 + get_local_id(0);
    uint x = seed + (uint)index;
    for (uint i = 0; i < rounds; i++) x = x * 1664525u + 1013904223u;
    out[index] = x;
}
