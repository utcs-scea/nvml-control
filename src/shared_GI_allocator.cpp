#include "error.hpp"
#include "nvml_control/allocator.hpp"

#include <algorithm>  // std::count_if
#include <thread>

namespace nvml {

namespace {
const unsigned int A100_N_SLICES = 7;
}  // anonymous namespace

SharedGIAllocator::SharedGIAllocator(GPU &device)
    : Allocator(device), gpu_instance_(device_, A100_N_SLICES) {
}

unsigned int SharedGIAllocator::remaining(unsigned short n_slices) const noexcept {
    return gpu_instance_.remaining_compute_instance_capacity(n_slices);
}

ComputeInstance SharedGIAllocator::allocate(unsigned short n_slices) {
    if (n_slices == 0 || n_slices > A100_N_SLICES) {
        // not possibly to satisfy invalid requests
        throw std::invalid_argument("n_slices is out of range");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    ComputeInstance compute_instance(gpu_instance_, n_slices);
    return compute_instance;
}

void SharedGIAllocator::free(ComputeInstance &&instance) {
    std::unique_lock<std::mutex> lock(mutex_);
    { ComputeInstance free_on_scope_exit = std::move(instance); }
    cv_.notify_all();
}

}  // namespace nvml
