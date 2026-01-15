#pragma once

#include <vector>
#include <exception>
#include <cstring>
#include <limits>
#include "defines.h"

namespace sigma::util {//定义host 函数len

    template<typename T>
    class HostArray {

        T *data_;
        std::size_t len;

    public:

        SIGMA_NODISCARD std::size_t length() const {
            return len;
        }

        SIGMA_NODISCARD std::size_t size() const {
            return len;
        }

        HostArray() {
            data_ = nullptr;
            len = 0;
        }

        HostArray(std::size_t cnt) {
            if (cnt > 0) {
                data_ = new T[cnt];
                memset(data_, 0, sizeof(T) * cnt);
            } else {
                data_ = nullptr;
            }
            len = cnt;
        }

        HostArray(T *data, std::size_t cnt) : data_(data), len(cnt) {
        }

        HostArray(const T *copy_from, std::size_t cnt) {
            if (cnt == 0) {
                data_ = nullptr;
                len = 0;
                return;
            }
            data_ = new T[cnt];
            for (std::size_t i = 0; i < cnt; i++) data_[i] = copy_from[i];
            len = cnt;
        }

        HostArray(const std::vector<T> &a) {
            len = a.size();
            data_ = new T[len];
            for (std::size_t i = 0; i < len; i++) {
                data_[i] = a[i];
            }
        }

        HostArray(HostArray &&arr) {
            data_ = arr.data_;
            len = arr.len;
            arr.data_ = nullptr;
            arr.len = 0;
        }

        ~HostArray() {
            delete[] data_;
        }

        HostArray &operator=(const HostArray &r) = delete;

        HostArray &operator=(HostArray &&from) {
            delete[] data_;
            data_ = from.data_;
            len = from.len;
            from.data_ = nullptr;
            from.len = 0;
            return *this;
        }

        HostArray(const HostArray &r) = delete;

        HostArray<T> copy() const {
            // need to cast data into const pointer
            // to make sure the contents are copied.
            const T *const_ptr = static_cast<const T *>(data_);
            return HostArray<T>(const_ptr, len);
        }

        const T &operator[](std::size_t i) const {
            return data_[i];
        }

        T &operator[](std::size_t i) {
            return data_[i];
        }

        T *get() {
            return data_;
        }

        const T *get() const {
            return data_;
        }

    };

}
