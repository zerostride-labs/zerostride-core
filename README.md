# ZeroStride

ZeroStride is an experimental C++ inference runtime focused on executing Mixture-of-Experts (MoE) models on hardware with physical memory constraints.

The goal of the project is to investigate and evaluate techniques to mitigate token generation latency jitter caused by OS paging (*page faults*), exploring:

- Static memory allocation and zero dynamic allocations within the inference hot loop.
- Predictive and asynchronous streaming of expert weights from secondary storage.
- Bounded and deterministic memory consumption during decoding.

Inspired by open ecosystem work such as `llama.cpp` and `strata`.

## Status

Early technical research and design phase.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
