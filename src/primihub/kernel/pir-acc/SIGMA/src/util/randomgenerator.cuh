#pragma once

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <cstdint>
#include "../kernelprovider.cuh"


namespace sigma::util {
    __global__
    void generate_random(curandStateXORWOW_t* states, uint64_t *arr, size_t size);

    __global__
    void initialize_generator(curandStateXORWOW_t* states, unsigned long long seed, size_t size);

    class RandomGenerator {

    public:

        void prepare_states(size_t size);

        void generate(uint64_t *destination, size_t size);

        void generate(uint64_t *destination, size_t size, cudaStream_t &stream);

    private:
        curandStateXORWOW_t *states_ = nullptr;
        size_t size_ = 0;

    };
}
