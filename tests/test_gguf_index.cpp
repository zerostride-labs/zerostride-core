#include "format/gguf_index.hpp"
#include "memory/alloc_guard.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>

namespace {

int test_synthetic_gguf_indexing_and_offsets() {
    const char* tmp_path = "/tmp/zerostride_phase1_test.gguf";

    zerostride::format::SyntheticMoESpec spec{};
    spec.num_layers = 4;
    spec.num_experts = 16;
    spec.active_experts_k = 4;
    spec.hidden_dim = 256;
    spec.ffn_dim = 512;

    zerostride::Status st =
        zerostride::format::GgufIndex::write_synthetic_moe_gguf(tmp_path, spec);
    if (!st.ok()) {
        std::cerr << "[FAIL] write_synthetic_moe_gguf failed: " << st.message << "\n";
        return 1;
    }

    // Place GgufIndex in static storage (.bss) to respect bounded stack frames
    static zerostride::format::GgufIndex idx;
    {
        zerostride::memory::AllocGuard guard;
        st = idx.index_file(tmp_path);
        if (!st.ok()) {
            std::cerr << "[FAIL] GgufIndex::index_file failed: " << st.message << "\n";
            std::remove(tmp_path);
            return 1;
        }
        if (!guard.verify_zero_allocations().ok()) {
            std::cerr << "[FAIL] GgufIndex::index_file performed dynamic heap allocations\n";
            std::remove(tmp_path);
            return 1;
        }
    }

    std::remove(tmp_path);

    const auto& meta = idx.metadata();
    if (meta.gguf_version != 3 || meta.block_count != 4 || meta.expert_count != 16 ||
        meta.expert_used_count != 4 || meta.embedding_length != 256) {
        std::cerr << "[FAIL] Extracted GGUF metadata mismatch\n";
        return 1;
    }

    // 1 global + 4 layers * 5 tensors = 21 tensors
    if (idx.tensors().size() != 21) {
        std::cerr << "[FAIL] Expected 21 indexed tensors, got " << idx.tensors().size() << "\n";
        return 1;
    }

    // Verify classification of ResidentDense, ResidentRouter, and StreamableMoEExpert
    const auto* embd = idx.find_tensor("token_embd.weight");
    const auto* router0 = idx.find_tensor("blk.0.ffn_gate_inp.weight");
    const auto* exps0 = idx.find_tensor("blk.0.ffn_gate_exps.weight");

    if (embd == nullptr ||
        embd->storage_class != zerostride::format::TensorStorageClass::kResidentDense) {
        std::cerr << "[FAIL] token_embd.weight missing or misclassified\n";
        return 1;
    }
    if (router0 == nullptr ||
        router0->storage_class != zerostride::format::TensorStorageClass::kResidentRouter ||
        router0->layer_index != 0) {
        std::cerr << "[FAIL] blk.0.ffn_gate_inp.weight missing or misclassified\n";
        return 1;
    }
    if (exps0 == nullptr ||
        exps0->storage_class != zerostride::format::TensorStorageClass::kStreamableMoEExpert ||
        exps0->expert_count != 16) {
        std::cerr << "[FAIL] blk.0.ffn_gate_exps.weight missing or misclassified\n";
        return 1;
    }

    // Verify 3D expert slice offset arithmetic:
    // Each expert is [256, 512] in Q4_K (256 elements = 1 block of 144 bytes -> 512 blocks * 144B = 73,728 bytes)
    const std::uint64_t expected_slice_bytes = 512ULL * 144ULL;
    if (exps0->expert_slice_bytes != expected_slice_bytes) {
        std::cerr << "[FAIL] Expected expert_slice_bytes == " << expected_slice_bytes << ", got "
                  << exps0->expert_slice_bytes << "\n";
        return 1;
    }
    if (exps0->expert_file_offset(3) != exps0->file_offset + 3ULL * expected_slice_bytes) {
        std::cerr << "[FAIL] Incorrect expert_file_offset(3) calculation\n";
        return 1;
    }

    // Verify WorkingSetReport math
    const auto report = idx.compute_working_set();
    // 1 single expert across gate_exps + up_exps + down_exps = 3 * 73,728 = 221,184 bytes
    // Top-4 active experts per layer = 4 * 221,184 = 884,736 bytes
    if (report.single_layer_top_k_bytes != 4ULL * 3ULL * expected_slice_bytes) {
        std::cerr << "[FAIL] Unexpected single_layer_top_k_bytes: "
                  << report.single_layer_top_k_bytes << "\n";
        return 1;
    }
    if (report.total_resident_Arena_bytes >= report.total_file_bytes) {
        std::cerr << "[FAIL] Expected resident arena footprint to be smaller than total MoE file\n";
        return 1;
    }

    std::ostringstream oss;
    idx.print_summary(oss, 64ULL * 1024 * 1024);
    if (oss.str().find("[FITS IN CEILING]") == std::string::npos) {
        std::cerr << "[FAIL] Summary output missing [FITS IN CEILING]\n";
        return 1;
    }

    return 0;
}

} // namespace

int main() {
    if (test_synthetic_gguf_indexing_and_offsets() != 0) {
        return 1;
    }

    std::cout << "[PASS] All ZeroStride Phase 1 GGUF Zero-Load Indexer unit tests succeeded.\n";
    return 0;
}
