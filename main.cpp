#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("./model/LunarLander-v2_2550.pt");
  }
  catch (const c10::Error& e) {
        std::cout << "failed" << std::endl;
        std::cerr << "error loading the model\n";
        return -1;
  }

  // Create a vector of inputs.
  torch::Device device(torch::kCUDA);
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> inputs2;
  inputs.push_back(torch::zeros({1, 8}, device));
  inputs2.push_back(torch::ones({1, 8}, device));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  at::Tensor output2 = module.forward(inputs2).toTensor();
  std::cout << output2.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}
