#pragma once


#include "hostarray.h"
#include "pointer.h"
#include "../kernelprovider.cuh"
#include "cuda_runtime.h"
#include <vector>
#include <exception>


namespace sigma::util {

    enum MemoryType : uint
    {//存储type
        MemoryTypeNone   = 0,
        MemoryTypeHost   = 1 << 0,
        MemoryTypeDevice = 1 << 1,
        MemoryTypeAll    = UINT_MAX,
    };

    template<typename T>
    class DeviceArray {
        T *data_;
        size_t len_;
    public:
        DeviceArray() {
            data_ = nullptr;
            len_ = 0;
        }

        explicit DeviceArray(size_t cnt) {
            data_ = KernelProvider::malloc<T>(cnt);
            len_ = cnt;
        }

        SIGMA_NODISCARD size_t length() const {
            return len_;
        }

        SIGMA_NODISCARD size_t size() const {
            return len_;
        }

        // 保留bool allocate编译不通过以提醒
        DeviceArray(T *data, size_t length, bool allocate) {
            if (allocate) {
                len_ = length;
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copy(data_, data, len_);
            } else {
                data_ = data;
                len_ = length;
            }
        }

        DeviceArray(DeviceArray &&a) {
            data_ = a.data_;
            len_ = a.len_;
            a.data_ = nullptr;
            a.len_ = 0;
        }

        DeviceArray &operator=(DeviceArray &&a) {
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = a.data_;
            len_ = a.len_;
            a.data_ = nullptr;
            a.len_ = 0;
            return *this;
        }

        DeviceArray(const HostArray<T> &host) {
            len_ = host.length();
            data_ = KernelProvider::malloc<T>(len_);
            KernelProvider::copy(data_, host.get(), len_);
        }

        ~DeviceArray() {
            release();
        }

        DeviceArray copy() const {
            T *copied = KernelProvider::malloc<T>(len_);
            KernelProvider::copyOnDevice<T>(copied, data_, len_);
            return DeviceArray(copied, len_);
        }

        DeviceArray &operator=(const DeviceArray &r) {//运算符重载
            if (data_) {
                KernelProvider::free(data_);
            }
            len_ = r.len_;
            data_ = KernelProvider::malloc<T>(len_);
            KernelProvider::copyOnDevice<T>(data_, r.data_, len_);
            return *this;
        }

        DeviceArray(const DeviceArray &r) {
            len_ = r.len_;
            if (len_ > 0) {
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copyOnDevice<T>(data_, r.data_, len_);
            } else {
                data_ = nullptr;
            }
        }

        HostArray<T> toHost() const {
            T *ret = new T[len_];
            KernelProvider::retrieve(ret, data_, len_);
            return HostArray<T>(ret, len_);
        }

        __host__ __device__
        T *get() {
            return data_;
        }

        __host__ __device__
        const T *get() const {
            return data_;
        }

        inline void release() {
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = nullptr;
            len_ = 0;
        }

        void resize(size_t size) {//数据清空？重新分配大小
            if (len_ == size) {
                return;
            }
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = KernelProvider::malloc<T>(size);
            len_ = size;
        }

        /*DevicePointer<T> ensure(size_t size) {
            if (size > size_) resize(size);
            return asPointer();
        }*/

        void set_data(T *data, size_t length) {
            data_ = data;
            len_ = length;
        }

        void copy_device_data(const T *data, size_t length) {
            if (len_ == length) {
                KernelProvider::copyOnDevice<T>(data_, data, len_);
            } else {
                KernelProvider::free(data_);
                len_ = length;
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copyOnDevice<T>(data_, data, len_);
            }
        }

        void set_host_data(const T *data, size_t length) {//对device
            if (len_ == length) {
                KernelProvider::copy(data_, data, len_);
            } else {
                KernelProvider::free(data_);
                len_ = length;
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copy(data_, data, len_);
            }
        }

        T back() const {
            T ret;
            if (data_) KernelProvider::retrieve(&ret, data_ + len_ - 1, 1);
            return ret;
        }

        bool isNull() const {
            return data_ == nullptr;
        }

    };

}

