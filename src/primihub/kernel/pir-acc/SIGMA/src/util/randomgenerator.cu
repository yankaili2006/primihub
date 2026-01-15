
#include "randomgenerator.cuh"
#include "../kernelprovider.cuh"

namespace sigma::util {
    __global__
    void generate_random(curandStateXORWOW_t* states, uint64_t *arr, size_t size) {
        size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid >= size) return;

        auto destination = reinterpret_cast<uint32_t *>(arr + tid);
        destination[0] = curand(&states[tid]);
        destination[1] = curand(&states[tid]);
    }

    __global__
    void initialize_generator(curandStateXORWOW_t* states, unsigned long long seed, size_t size) {
        size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid >= size) return;

        curand_init(seed, tid, 0, &states[tid]);
    }

    void RandomGenerator::generate(uint64_t *destination, size_t size) {
        size_t thread_count = 256;
        size_t block_count = (size - 1) / thread_count + 1;
        generate_random<<<block_count, thread_count>>>(states_, destination, size);
    }

    void RandomGenerator::generate(uint64_t *destination, size_t size, cudaStream_t &stream) {
        size_t thread_count = 256;
        size_t block_count = (size - 1) / thread_count + 1;
        generate_random<<<block_count, thread_count, 0, stream>>>(states_, destination, size);
    }


    void RandomGenerator::prepare_states(size_t size) {
        if (states_ && size == size_) {
            return;
        }
        if (states_) {
            KernelProvider::free(states_);
        }
        states_ = KernelProvider::malloc<curandStateXORWOW_t>(size);
        size_ = size;

        size_t thread_count = 1024;
        size_t block_count = (size - 1) / thread_count + 1;
        initialize_generator<<<block_count, thread_count>>>(states_, time(nullptr), size);
    }

}
