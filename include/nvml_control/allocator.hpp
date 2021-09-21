#pragma once

#include "nvml_control/instance.hpp"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace nvml {

/**
 * @brief the Allocator creates Compute instances with isolated memory and
 * compute resources.
 */
class Allocator {
protected:
    GPU &device_;
    std::mutex mutex_;
    std::condition_variable cv_;

public:
    /**
     * @brief Constructs an Allocator for device.
     * @param device the GPU device on which to allocate the compute
     * @throws runtime_error if there is no available GPU Instance capacity on
     * the device
     */
    Allocator(GPU &device);

    virtual ~Allocator() = default;
    /**
     * @brief Allocate a ComputeInstance on the GPU.
     * This operation blocks until the placement can be satisfied
     * @param n_slices the number of slices to allocate
     * @returns The allocated ComputeInstance
     * @throws runtime_error if unable to allocate
     */
    virtual ComputeInstance allocate(unsigned short n_slices) = 0;

    /**
     * @brief Returns the number of remaining allocations for n_slices
     * @param n_slices the size of the allocation unit
     */
    virtual unsigned int remaining(unsigned short n_slices) const noexcept = 0;

    /**
     * @brief Free a ComputeInstance and make its range of slices available for
     * future allocations. This operation frees the GPU Instance and Compute
     * Instance of the ComputeInstance and notifies any waiters blocked in
     * allocate.
     * @param instance An instance to free. Should not be in use.
     * @throws runtime_error if unable to free the instance
     */
    virtual void free(ComputeInstance &&instance) = 0;
};

class SharedGIAllocator : public Allocator {
private:
    GPUInstance gpu_instance_;
public:
    SharedGIAllocator(GPU& device);
    ComputeInstance allocate(unsigned short n_slices) override;
    unsigned int remaining(unsigned short n_slices) const noexcept override;
    void free(ComputeInstance &&instance) override;
};

class IsolatedGIAllocator : public Allocator {
public:
    using Allocator::Allocator;
    ComputeInstance allocate(unsigned short n_slices) override;
    unsigned int remaining(unsigned short n_slices) const noexcept override;
    void free(ComputeInstance &&instance) override;
};

}  // namespace nvml
