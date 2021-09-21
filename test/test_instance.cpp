#include "nvml_control/instance.hpp"
#include "gtest/gtest.h"

#include <algorithm>

namespace mut = nvml;

namespace {
// hard-coded for an A100 GPU
constexpr int TEST_GPU_ID = 1;
constexpr unsigned int A100_N_SLICES = 7;
}  // namespace

TEST(NvmlControl, GPU_ConstructDestruct) {
    mut::GPU gpu(TEST_GPU_ID);
}

class NvmlControlGPU : public ::testing::Test {
public:
    mut::GPU gpu_;
    NvmlControlGPU() : gpu_(TEST_GPU_ID) {}
};

TEST_F(NvmlControlGPU, InstanceCapacityEmpty) {
    ASSERT_EQ(A100_N_SLICES, gpu_.remaining_gpu_instance_capacity(1));
}

TEST_F(NvmlControlGPU, GPUInstance_ConstructDestructSize1) {
    ASSERT_NO_THROW(mut::GPUInstance gi(gpu_, 1));
}

TEST_F(NvmlControlGPU, GPUInstance_ConstructDestructSizeMax) {
    mut::GPUInstance gi(gpu_, A100_N_SLICES);
}

TEST_F(NvmlControlGPU, GPUInstance_ConstructDestructAllSizes) {
    std::vector<unsigned short> invalid_slice_sizes = {5, 6};
    for (unsigned int slices = A100_N_SLICES; slices >= 1; slices--) {
        if (std::find(invalid_slice_sizes.cbegin(), invalid_slice_sizes.cend(),
                      slices) != invalid_slice_sizes.cend()) {
            EXPECT_THROW(mut::GPUInstance gi(gpu_, slices),
                         std::invalid_argument);
        } else {
            EXPECT_NO_THROW(mut::GPUInstance gi(gpu_, slices));
        }
    }
}

TEST_F(NvmlControlGPU, GPUInstance_ConstructDestructMoreThanCapacity) {
    // If we do something wrong in destruction this will catch it
    for (unsigned int i = 0; i < A100_N_SLICES * 2; i++) {
        mut::GPUInstance gi(gpu_, 1);
    }
}

TEST_F(NvmlControlGPU, GPUInstance_ConstructOverCapacity) {
    // the biggest GPU Instance profile ID is 0
    mut::GPUInstance gi(gpu_, 7);
    // now try to construct one more, this should throw
    ASSERT_THROW(mut::GPUInstance(gpu_, 1), std::runtime_error);
}

class NvmlControlGPUInstance : public NvmlControlGPU {
public:
    mut::GPUInstance gpu_instance_;
    // create a GPU Instance that occupies the whole GPU
    NvmlControlGPUInstance() : NvmlControlGPU(), gpu_instance_(gpu_, 7) {}
};

TEST_F(NvmlControlGPUInstance, ComputeInstance_ConstructDestruct) {
    mut::ComputeInstance compute_instance{gpu_instance_, 1};
}

TEST_F(NvmlControlGPUInstance,
       ComputeInstance_ConstructDestructMoreThanCapacity) {
    for (unsigned int i = 0; i < A100_N_SLICES * 2; i++) {
        mut::ComputeInstance ci(gpu_instance_, 1);
    }
}

TEST_F(NvmlControlGPUInstance, ComputeInstance_ConstructOverCapacity) {
    mut::ComputeInstance ci(gpu_instance_, 7);
    ASSERT_THROW(mut::ComputeInstance(gpu_instance_, 1), std::runtime_error);
}

TEST_F(NvmlControlGPUInstance, ComputeInstance_DifferentCudaVisibleDevicesStrings) {
    mut::ComputeInstance ci1(gpu_instance_, 3);
    mut::ComputeInstance ci2(gpu_instance_, 4);
    ASSERT_NE(ci1.get_cuda_visible_devices_string(),
                 ci2.get_cuda_visible_devices_string());
}
