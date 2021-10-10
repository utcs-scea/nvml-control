#include "error.hpp"
#include "nvml_control/allocator.hpp"

#include <thread>

namespace nvml {

namespace {
const unsigned int A100_N_SLICES = 7;
}  // anonymous namespace

ComputeInstance IsolatedGIAllocator::allocate(unsigned short n_slices) {
    if (n_slices == 0 || n_slices > A100_N_SLICES) {
        // not possibly to satisfy invalid requests
        throw std::invalid_argument("n_slices is out of range");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    GPUInstance gpu_instance(device_, n_slices);
    ComputeInstance compute_instance(std::move(gpu_instance), n_slices);
    return compute_instance;
}

unsigned int
IsolatedGIAllocator::remaining(unsigned short n_slices) const noexcept {
    return device_.remaining_gpu_instance_capacity(n_slices);
}

void IsolatedGIAllocator::free(ComputeInstance &&instance) {
    std::unique_lock<std::mutex> lock(mutex_);
    { ComputeInstance free_on_scope_exit = std::move(instance); }
    cv_.notify_all();
}

}  // namespace nvml
