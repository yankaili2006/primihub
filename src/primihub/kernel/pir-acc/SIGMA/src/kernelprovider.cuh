#pragma once

#include <stdexcept>
#include <iostream>
#include "cuda_runtime.h"

#define CHECK(call) \
do {\
    const cudaError_t error_code = call; \
    if (error_code != cudaSuccess) {\
        printf("CUDA Error:\n"); \
        printf(" File: %s\n",__FILE__); \
        printf(" Line: %d\n",__LINE__); \
        printf(" Errorcode:%d\n",error_code); \
        printf(" Errortext:%s\n",cudaGetErrorString(error_code));\
        exit(1); \
    } \
} while (0)

namespace sigma {

    class KernelProvider {

        static bool initialized;

    public:

        static void checkInitialized() {
            if (!initialized)
                throw std::invalid_argument("KernelProvider not initialized.");
        }

        static void initialize() {//cudasetdevice
            cudaSetDevice(0);
            initialized = true;
        }

        template<typename T>
        static T *malloc(size_t length) {//分配GPU上的内存
            checkInitialized();
            if (length == 0) return nullptr;
            T *ret;
            auto status = cudaMalloc((void **) &ret, length * sizeof(T));
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda Malloc failed.");
            return ret;
        }

        template<typename T>
        static void free(T *pointer) {//释放GPU上分配的内存
            checkInitialized();//free指针
            auto status = cudaFree(pointer);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda free failed.");
        }

        template<typename T>
        static void copy(T *deviceDestPtr, const T *hostFromPtr, size_t length) {//数据复制传输
            checkInitialized();
            if (length == 0) return;
            auto status = cudaMemcpy(deviceDestPtr, hostFromPtr, length * sizeof(T), cudaMemcpyHostToDevice);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda copy from host to device failed.");
        }

        template<typename T>
        static void copyAsync(T *deviceDestPtr, const T *hostFromPtr, size_t length, cudaStream_t &stream) {
            checkInitialized();
            if (length == 0) return;
            auto status = cudaMemcpyAsync(deviceDestPtr, hostFromPtr, length * sizeof(T), cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda copy from host to device failed.");
        }

        template<typename T>
        static void copyOnDevice(T *deviceDestPtr, const T *deviceFromPtr, size_t length) {//设备到设备的数据传输
            checkInitialized();
            if (length == 0) return;
            auto status = cudaMemcpy(deviceDestPtr, deviceFromPtr, length * sizeof(T), cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda copy on device failed.");
        }

        template<typename T>
        static void retrieve(T *hostDestPtr, const T *deviceFromPtr, size_t length) {
            checkInitialized();
            if (length == 0) return;
            auto status = cudaMemcpy(hostDestPtr, deviceFromPtr, length * sizeof(T), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda retrieve from device to host failed.");
        }

        template<typename T>
        static void retrieveAsync(T *hostDestPtr, const T *deviceFromPtr, size_t length, cudaStream_t &stream) {
            checkInitialized();
            if (length == 0) return;
            auto status = cudaMemcpyAsync(hostDestPtr, deviceFromPtr, length * sizeof(T), cudaMemcpyDeviceToHost, stream);
            if (status != cudaSuccess)
                throw std::runtime_error("Cuda retrieve from device to host failed.");
        }

        template<typename T>
        static void memsetZero(T *devicePtr, size_t length) {
            if (length == 0) return;
            cudaMemset(devicePtr, 0, sizeof(T) * length);
        }

    };

}
