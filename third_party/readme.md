# Third-Party Dependencies

This directory hosts external dependencies used by `zerostride-core`.

- **`ggml/`** (populated in Phase 1): Tensor algebra backend used strictly for low-level compute kernels (AVX2 / ARM NEON / Metal). Memory allocation and expert I/O orchestration are managed by `zerostride-core`.
