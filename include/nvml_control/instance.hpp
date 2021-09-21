#pragma once

#include <memory>
#include <nvml.h>
#include <stdexcept>
#include <string>

namespace nvml {

/**
 * @brief A class encapsulating a GPU device representation
 */
class GPU {
private:
    nvmlDevice_t device_;
    friend class GPUInstance;  // for access to device_
public:
    /// The GPU index passed to the constructor
    const int device_id_;

    /**
     * @brief Constructor
     * @param device The GPU device index number, such as the number passed to
     * CUDA_VISIBLE_DEVICES
     */
    GPU(int device) noexcept;

    /**
     * @brief Gets the number of concurrent GPU Instances 
     * that can be allocated
     * @param n_slices n_slices the number of slices in each GPU Instance
     */
    unsigned int remaining_gpu_instance_capacity(unsigned short n_slices) const noexcept;

private:
    /**
     * @brief Returns the GPU Instance profile ID corresponding to the n_slices.
     * @param n_slices The number of slices.
     * @returns the instance ID
     * @throws invalid_argument if n_slices is invalid.
     */
    unsigned int look_up_gpu_instance_profile_id(unsigned short n_slices) const;

    /**
     * @brief Returns the Compute Instance profile ID corresponding to the
     * n_slices.
     * @param n_slices The number of slices.
     * @returns the instance ID
     * @throws invalid_argument if n_slices is invalid.
     */
    unsigned int
    look_up_compute_instance_profile_id(unsigned short n_slices) const;
    friend class ComputeInstance;  // for access to look_up_compute_instance_profile_id
};

class GPUInstance {
private:
    bool valid_{false};
    friend class Allocator;  // for access to valid_
    GPU const * gpu_;
    friend class ComputeInstance;  // for access to gpu_ 
    nvmlGpuInstance_t instance_;
    friend class ComputeInstance;  // for access to instance_;

public:
    /**
     * @brief Create a GPUInstance
     * @param gpu on this GPU device
     * @param size occupying this many slices
     */
    GPUInstance(GPU &gpu, unsigned short size);
    GPUInstance(GPUInstance &&rhs) noexcept;
    GPUInstance() noexcept : valid_(false) {}
    ~GPUInstance() noexcept;

    GPUInstance &operator=(GPUInstance &&rhs) noexcept;

    /**
     * @brief Gets the number of remaining Compute Instances available of size n_slices a
     * on this GPU Instance.
     * @param n_slices The number of slices
     */
    unsigned int
    remaining_compute_instance_capacity(unsigned short n_slices) const noexcept;

    nvmlGpuInstancePlacement_t get_placement() const noexcept;
    bool is_valid() const noexcept { return valid_; }
};

class ComputeInstance {
private:
    bool valid_{false};
    nvmlComputeInstance_t instance_;
    friend class Allocator;  // for access to instance_ in
                             // allocated_slice_start_ map
    GPUInstance managed_;    // only set if this Compute Instance manges its own
                             // GPU Instance

public:
    /**
     * @brief Create a ComputeInstance
     * @param gpu on this GPUInstance
     * @param n_slices with this many slices 
     * This constructor takes ownership of a GPU instance and manages its
     * lifetime.
     */
    ComputeInstance(GPUInstance &&gpu_instance, unsigned int n_slices);

    /**
     * @brief Create a ComputeInstance
     * @param gpu on this GPUInstance
     * @param n_clies with this many slices 
     * This constructor refers to an existing GPU instance
     */
    ComputeInstance(GPUInstance &gpu_instance, unsigned int n_slices);
    ComputeInstance(ComputeInstance &&rhs) noexcept;
    ComputeInstance() noexcept : valid_(false) {}
    ~ComputeInstance() noexcept;
    ComputeInstance &operator=(ComputeInstance &&rhs) noexcept;
    std::string get_cuda_visible_devices_string() const noexcept;
    bool is_valid() const noexcept { return valid_; }
};

class NVMLControl {
private:
    friend class ComputeInstance;

public:
    NVMLControl();
    ~NVMLControl();
};

}  // namespace nvml
