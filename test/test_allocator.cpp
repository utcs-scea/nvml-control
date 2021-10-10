#include "multiset_sum.hpp"
#include "nvml_control/allocator.hpp"
#include "gtest/gtest.h"

#include <chrono>
#include <list>
#include <thread>

namespace mut = nvml;

namespace {
// hard-coded for an A100 GPU
constexpr int TEST_GPU_ID = 1;
}  // anonymous namespace

template <typename AllocatorType>
class Allocator : public ::testing::Test {
public:
    mut::GPU gpu_;
    AllocatorType allocator_;
    std::list<mut::ComputeInstance> allocated_;
    Allocator() : gpu_(TEST_GPU_ID), allocator_(gpu_) {}

    void TearDown() override {
        while (allocated_.size()) {
            allocator_.free(std::move(allocated_.front()));
            allocated_.pop_front();
        }
    }
};

typedef ::testing::Types<mut::IsolatedGIAllocator, mut::SharedGIAllocator>
    AllocatorTypes;
TYPED_TEST_CASE(Allocator, AllocatorTypes);

TYPED_TEST(Allocator, allocate_all) {
    constexpr unsigned int n_slices = 1;
    auto total = this->allocator_.remaining(n_slices);
    for (unsigned int i = 0; i < total; i++) {
        this->allocated_.push_back(this->allocator_.allocate(n_slices));
    }
}

TYPED_TEST(Allocator, allocate_full_throws) {
    constexpr unsigned int n_slices = 1;
    auto total = this->allocator_.remaining(n_slices);
    for (unsigned int i = 0; i < total; i++) {
        this->allocated_.push_back(this->allocator_.allocate(n_slices));
    }
    EXPECT_THROW(this->allocator_.allocate(1), std::runtime_error);
}

TYPED_TEST(Allocator, allocate_auto_location) {
    this->allocated_.push_back(this->allocator_.allocate(3));
    this->allocated_.push_back(this->allocator_.allocate(2));
}

namespace {
template <typename T>
std::string vector_to_string(const std::vector<T> &vec) {
    std::stringstream ss;
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); i++) {
        ss << vec.at(i);
        if (i != (vec.size() - 1)) {
            ss << ", ";
        }
    }
    return ss.str();
}

void allocation_order(mut::Allocator &allocator, unsigned int use_slices) {
    // hard-coded for A100 slice sizes
    const auto all_slice_groups = multiset_sum(use_slices, {1, 2, 3, 4, 7});
    for (auto slices : all_slice_groups) {
        std::vector<unsigned int> permutable_slices(slices.cbegin(),
                                                    slices.cend());
        std::cout << "Trying all permutations of: ["
                  << vector_to_string(permutable_slices) << "]" << std::endl;
        do {
            std::cout << "   [ ";
            // We wrap this procedure in a lambda so we can use it in a
            // googletest assert
            std::list<mut::ComputeInstance> instances;
            auto test_instances = [&] {
                // allocate Instances for all slices
                for (auto n_slices : permutable_slices) {
                    std::cout << n_slices << " " << std::flush;
                    instances.push_back(allocator.allocate(n_slices));
                }
            };
            try {
                test_instances();
                std::cout << "] succeeded";
            } catch (const std::runtime_error &error) {
                std::cout << "<-- failed ] ";
            }
            std::cout << std::endl;
            while (instances.size()) {
                allocator.free(std::move(instances.front()));
                instances.pop_front();
            }
        } while (std::next_permutation(permutable_slices.begin(),
                                       permutable_slices.end()));
    }
}
}  // namespace

TYPED_TEST(Allocator, AllocationOrder_Total7) {
    allocation_order(this->allocator_, 7);
}

TYPED_TEST(Allocator, AllocationOrder_Total6) {
    allocation_order(this->allocator_, 6);
}

TYPED_TEST(Allocator, AllocationOrder_Total5) {
    allocation_order(this->allocator_, 5);
}

TYPED_TEST(Allocator, AllocationOrder_Total4) {
    allocation_order(this->allocator_, 4);
}

TYPED_TEST(Allocator, AllocationOrder_Total3) {
    allocation_order(this->allocator_, 3);
}

TYPED_TEST(Allocator, AllocationOrder_Total2) {
    allocation_order(this->allocator_, 2);
}

TYPED_TEST(Allocator, AllocationOrder_Total1) {
    allocation_order(this->allocator_, 1);
}

TYPED_TEST(Allocator, DifferentCudaVisibleDevicesStrings) {
    this->allocated_.push_back(this->allocator_.allocate(3));
    this->allocated_.push_back(this->allocator_.allocate(2));
    ASSERT_NE(this->allocated_.front().get_cuda_visible_devices_string(),
              this->allocated_.back().get_cuda_visible_devices_string());
}
