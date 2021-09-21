#pragma once

#define THROW_NVML(expr)                         \
    do {                                         \
        nvmlReturn_t __ret;                      \
        if ((__ret = expr) != NVML_SUCCESS) {    \
            std::string msg = #expr " failed: "; \
            msg += nvmlErrorString(__ret);       \
            throw std::runtime_error(msg);       \
        }                                        \
    } while (false)

#define CHECK_NVML(expr)                                                \
    do {                                                                \
        nvmlReturn_t __ret;                                             \
        if ((__ret = expr) != NVML_SUCCESS) {                           \
            std::cerr << #expr << " failed: " << nvmlErrorString(__ret) \
                      << std::endl;                                     \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (false)
