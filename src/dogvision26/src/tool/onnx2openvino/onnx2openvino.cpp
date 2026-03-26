#include <filesystem>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/pass/serialize.hpp>
#include <string>

namespace {

void print_usage(const char* exec_name) {
    std::cout << "Usage:\n"
              << "  " << exec_name << " <input.onnx> <output.xml | output_dir>\n\n"
              << "Examples:\n"
              << "  " << exec_name << " model.onnx model.xml\n"
              << "  " << exec_name << " model.onnx ./openvino_model\n";
}

std::pair<std::filesystem::path, std::filesystem::path> resolve_output_paths(
    const std::filesystem::path& onnx_path,
    const std::filesystem::path& output_arg) {
    if (output_arg.extension() == ".xml") {
        const auto xml_path = output_arg;
        const auto bin_path = xml_path.parent_path() / (xml_path.stem().string() + ".bin");
        return {xml_path, bin_path};
    }

    const auto model_name = onnx_path.stem().string();
    const auto xml_path = output_arg / (model_name + ".xml");
    const auto bin_path = output_arg / (model_name + ".bin");
    return {xml_path, bin_path};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path onnx_path = argv[1];
        const std::filesystem::path output_arg = argv[2];

        if (!std::filesystem::exists(onnx_path)) {
            std::cerr << "Error: ONNX file does not exist: " << onnx_path << std::endl;
            return 2;
        }

        auto [xml_path, bin_path] = resolve_output_paths(onnx_path, output_arg);

        if (!xml_path.parent_path().empty()) {
            std::filesystem::create_directories(xml_path.parent_path());
        }
        if (!bin_path.parent_path().empty()) {
            std::filesystem::create_directories(bin_path.parent_path());
        }

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(onnx_path.string());
        ov::pass::Serialize(xml_path.string(), bin_path.string()).run_on_model(model);

        std::cout << "Convert success." << std::endl;
        std::cout << "xml: " << xml_path << std::endl;
        std::cout << "bin: " << bin_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Convert failed: " << e.what() << std::endl;
        return 3;
    }

    return 0;
}