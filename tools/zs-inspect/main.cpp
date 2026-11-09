#include "format/gguf_index.hpp"
#include "zerostride/config.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " <model.gguf> [options]\n"
              << "  " << prog << " --generate-synthetic <out.gguf> [options]\n\n"
              << "Options:\n"
              << "  --ceiling-mb <MiB>         Target RAM ceiling in MiB (default: 3584)\n"
              << "  --top-k <K>                Override active experts K per token\n"
              << "  --list-tensors             Print offset table for all indexed tensors\n"
              << "  --layers <L>               Layers for --generate-synthetic (default: 4)\n"
              << "  --experts <E>              Total experts per layer for --generate-synthetic (default: 16)\n"
              << "  --help                     Show this help message\n";
}

const char* storage_class_label(zerostride::format::TensorStorageClass sc) noexcept {
    switch (sc) {
        case zerostride::format::TensorStorageClass::kResidentDense:
            return "RESIDENT_DENSE";
        case zerostride::format::TensorStorageClass::kResidentRouter:
            return "RESIDENT_ROUTER";
        case zerostride::format::TensorStorageClass::kStreamableMoEExpert:
            return "STREAM_MOE_EXPS";
    }
    return "UNKNOWN";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* input_path = nullptr;
    const char* generate_path = nullptr;
    std::size_t ceiling_bytes = zerostride::kDefaultMemoryCeilingBytes;
    std::uint32_t top_k_override = 0;
    bool list_tensors = false;
    zerostride::format::SyntheticMoESpec synth_spec{};

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--generate-synthetic" && i + 1 < argc) {
            generate_path = argv[++i];
        } else if (arg == "--ceiling-mb" && i + 1 < argc) {
            ceiling_bytes =
                static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10)) * 1024ULL * 1024ULL;
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k_override = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            synth_spec.active_experts_k = top_k_override;
        } else if (arg == "--layers" && i + 1 < argc) {
            synth_spec.num_layers =
                static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--experts" && i + 1 < argc) {
            synth_spec.num_experts =
                static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--list-tensors") {
            list_tensors = true;
        } else if (!arg.empty() && arg[0] != '-') {
            input_path = argv[i];
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (generate_path != nullptr) {
        const zerostride::Status st =
            zerostride::format::GgufIndex::write_synthetic_moe_gguf(generate_path, synth_spec);
        if (!st.ok()) {
            std::cerr << "Failed to generate synthetic GGUF: " << st.message << "\n";
            return 1;
        }
        input_path = generate_path;
    }

    if (input_path == nullptr) {
        std::cerr << "Error: missing GGUF file path.\n";
        print_usage(argv[0]);
        return 1;
    }

    static zerostride::format::GgufIndex index;
    const zerostride::Status st = index.index_file(input_path);
    if (!st.ok()) {
        std::cerr << "Error indexing '" << input_path << "': " << st.message << "\n";
        return 1;
    }

    index.print_summary(std::cout, ceiling_bytes, top_k_override);

    if (list_tensors) {
        std::cout << "\nIndexed Tensor Directory:\n";
        std::cout << std::left << std::setw(32) << "Tensor Name" << std::setw(8) << "Type"
                  << std::setw(18) << "Class" << std::right << std::setw(14) << "File Offset"
                  << std::setw(14) << "Size (B)" << std::setw(14) << "Slice (B)\n";
        std::cout << "----------------------------------------------------------------------------------------------------\n";
        for (const auto& t : index.tensors()) {
            const auto traits = zerostride::format::get_ggml_type_traits(t.type);
            std::cout << std::left << std::setw(32) << t.name_view() << std::setw(8) << traits.name
                      << std::setw(18) << storage_class_label(t.storage_class) << std::right
                      << std::setw(14) << t.file_offset << std::setw(14) << t.size_bytes
                      << std::setw(14) << t.expert_slice_bytes << "\n";
        }
    }

    return 0;
}
