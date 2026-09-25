// SPDX-License-Identifier: GPL-3.0-only
// Compile with -DLLAMA_REFERENCE_SOURCE='"/absolute/path/to/run.c"'.
#ifndef LLAMA_REFERENCE_SOURCE
#error Define LLAMA_REFERENCE_SOURCE to the upstream llama2.c run.c file
#endif
#define TESTING
#include LLAMA_REFERENCE_SOURCE

// Emit raw FP32 logits for a fixed sequence; no tokenizer or sampler involved.
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: reference checkpoint token...\n"); return 2; }
    Transformer model = {0};
    build_transformer(&model, argv[1]);
    if (argc - 2 > model.config.seq_len) return 2;
    for (int i = 2; i < argc; i++) {
        char *end;
        long token = strtol(argv[i], &end, 10);
        if (*end || token < 0 || token >= model.config.vocab_size) return 2;
        float *logits = forward(&model, (int)token, i - 2);
        if (fwrite(logits, sizeof(float), model.config.vocab_size, stdout) !=
            (size_t)model.config.vocab_size) return 1;
    }
    free_transformer(&model);
    return 0;
}
