#include "nvml_control/allocator.hpp"

namespace nvml {

namespace {
const unsigned int A100_N_SLICES = 7;
}  // anonymous namespace

Allocator::Allocator(GPU &device) : device_(device) {
    // hard-coded for A100 MIG, these values could be different on other GPUs
    unsigned int gpu_instances = device_.remaining_gpu_instance_capacity(1);
    if (gpu_instances != A100_N_SLICES) {
        throw std::runtime_error("Not enough Compute Instance capacity for allocator. Is GPU in use?");
    }
}

}  // namespace nvml
