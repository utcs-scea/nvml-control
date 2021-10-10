#include "nvml_control/instance.hpp"
#include "error.hpp"
#include <iostream>
#include <sstream>

namespace nvml {

GPU::GPU(int device) noexcept : device_id_(device) {
    CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(device, &device_));
}

unsigned int
GPU::remaining_gpu_instance_capacity(unsigned short n_slices) const noexcept {
    unsigned int ret{0};
    CHECK_NVML(nvmlDeviceGetGpuInstanceRemainingCapacity(
        device_, look_up_gpu_instance_profile_id(n_slices), &ret));
    return ret;
}

unsigned int
GPU::look_up_gpu_instance_profile_id(unsigned short n_slices) const {
    // hard-coded for A100 MIG, these values could be different on other GPUs
    // TODO use nvml API
    switch (n_slices) {
    case 1: return 19;
    case 2: return 14;
    case 3: return 9;
    case 4: return 5;
    case 7: return 0;
    default:
        throw std::invalid_argument(
            "n_slices does not correspond to a profile_id for this GPU");
    }
}

unsigned int
GPU::look_up_compute_instance_profile_id(unsigned short n_slices) const {
    // TODO use nvml API
    switch (n_slices) {
    case 1: return 0;
    case 2: return 1;
    case 3: return 2;
    case 4: return 3;
    case 7: return 4;
    default:
        throw std::invalid_argument(
            "n_slices does not correspond to a profile_id for this GPU");
    }
}

GPUInstance::GPUInstance(GPU &gpu, unsigned short size) : gpu_(&gpu) {
    // look up profile_id (TODO use nvml API)
    unsigned int profile_id = gpu.look_up_gpu_instance_profile_id(size);

    // we would like to use this API, but it doesnt seem to work and isn't
    // publicly documented.
    // THROW_NVML(nvmlDeviceCreateGpuInstanceWithPlacement( gpu.device_,
    // profile_id,
    // &placement, &instance_));
    THROW_NVML(
        nvmlDeviceCreateGpuInstance(gpu.device_, profile_id, &instance_));
    valid_ = true;

    /* TODO move to allocator
    constexpr unsigned int A100_MAX_PLACEMENTS = 7;
    nvmlGpuInstancePlacement_t placements[A100_MAX_PLACEMENTS];
    unsigned int count;
    THROW_NVML(nvmlDeviceGetGpuInstancePossiblePlacements(
        gpu.device_, profile_id, placements, &count));
    bool valid = false;
    for (unsigned int i = 0; i < count; i++)
    {
        if (placements[i].start == placement.start && placements[i].size ==
   placement.size)
        {
            valid = true;
            break;
        }
    }
    if (!valid)
    {
        throw std::runtime_error("No available placements for requested
   profile");
    }
    std::stack<GPUInstance> instances;
    do {
        // create an instance
        instances.emplace(gpu_, placement.size);
        // check the placement
    } while (instances.top().placement())
        // repeat until the instance is at the requested placement.
*/
}

GPUInstance::GPUInstance(GPUInstance &&rhs) noexcept
    : valid_(rhs.valid_), gpu_(rhs.gpu_), instance_(rhs.instance_) {
    rhs.valid_ = false;
    rhs.gpu_ = NULL;
    rhs.instance_ = NULL;
}

GPUInstance::~GPUInstance() noexcept {
    if (valid_) {
        CHECK_NVML(nvmlGpuInstanceDestroy(instance_));
    }
}

GPUInstance &GPUInstance::operator=(GPUInstance &&rhs) noexcept {
    valid_ = rhs.valid_;
    gpu_ = rhs.gpu_;
    instance_ = rhs.instance_;
    rhs.valid_ = false;
    rhs.gpu_ = NULL;
    rhs.instance_ = NULL;
    return *this;
}

unsigned int GPUInstance::remaining_compute_instance_capacity(
    unsigned short n_slices) const noexcept {
    unsigned int count;
    CHECK_NVML(nvmlGpuInstanceGetComputeInstanceRemainingCapacity(
        instance_, gpu_->look_up_compute_instance_profile_id(n_slices),
        &count));
    return count;
}

nvmlGpuInstancePlacement_t GPUInstance::get_placement() const noexcept {
    nvmlGpuInstanceInfo_t info;
    CHECK_NVML(nvmlGpuInstanceGetInfo(instance_, &info));
    return info.placement;
}

ComputeInstance::ComputeInstance(GPUInstance &&gpu_instance,
                                 unsigned int n_slices)
    : valid_(true), managed_(std::move(gpu_instance)) {
    THROW_NVML(nvmlGpuInstanceCreateComputeInstance(
        managed_.instance_,
        gpu_instance.gpu_->look_up_compute_instance_profile_id(n_slices),
        &instance_));
}

ComputeInstance::ComputeInstance(GPUInstance &gpu_instance,
                                 unsigned int n_slices)
    : valid_(true) {
    THROW_NVML(nvmlGpuInstanceCreateComputeInstance(
        gpu_instance.instance_,
        gpu_instance.gpu_->look_up_compute_instance_profile_id(n_slices),
        &instance_));
}

ComputeInstance::ComputeInstance(ComputeInstance &&rhs) noexcept
    : valid_(rhs.valid_), instance_(rhs.instance_),
      managed_(std::move(rhs.managed_)) {
    rhs.valid_ = false;
    rhs.instance_ = NULL;
}

ComputeInstance::~ComputeInstance() noexcept {
    if (valid_) {
        CHECK_NVML(nvmlComputeInstanceDestroy(instance_));
    }
}

ComputeInstance &ComputeInstance::operator=(ComputeInstance &&rhs) noexcept {
    valid_ = rhs.valid_;
    instance_ = rhs.instance_;
    managed_ = std::move(rhs.managed_);
    rhs.valid_ = false;
    rhs.instance_ = NULL;
    return *this;
}

std::string ComputeInstance::get_cuda_visible_devices_string() const noexcept {
    nvmlComputeInstanceInfo_t compute_instance_info;
    try {
        THROW_NVML(
            nvmlComputeInstanceGetInfo(instance_, &compute_instance_info));
    } catch (const std::runtime_error &e) {
        // instance_ is not valid
        // return empty string
        return {};
    }
    nvmlGpuInstanceInfo_t gpu_instance_info;
    CHECK_NVML(nvmlGpuInstanceGetInfo(compute_instance_info.gpuInstance,
                                      &gpu_instance_info));
    char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetUUID(gpu_instance_info.device, uuid,
                                 NVML_DEVICE_UUID_V2_BUFFER_SIZE));
    std::stringstream visible_device;
    // Format documented online:
    // https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-gi
    visible_device << "MIG-" << uuid << "/" << gpu_instance_info.id << "/"
                   << compute_instance_info.id;
    return visible_device.str();
}

NVMLControl::NVMLControl() {
    CHECK_NVML(nvmlInit_v2());
}

NVMLControl::~NVMLControl() {
    CHECK_NVML(nvmlShutdown());
}

namespace {
/// Automatically initialize nvml when this library loads
NVMLControl g_nvml_control;
}  // anonymous namespace

}  // namespace nvml
