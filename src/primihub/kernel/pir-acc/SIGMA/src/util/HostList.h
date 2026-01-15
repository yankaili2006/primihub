//
// Created by scwang on 2023/12/26.
//

#ifndef CUSEAL_HOSTLIST_H
#define CUSEAL_HOSTLIST_H

#include "defines.h"

namespace sigma::util {

    template<typename T>
    class HostList {
        T *data_;
        std::size_t len_;
        bool need_release_ = true;

    public:

        SIGMA_NODISCARD std::size_t

        length() const {
            return len_;
        }

        SIGMA_NODISCARD std::size_t

        size() const {
            return len_;
        }

        HostList() {
            data_ = nullptr;
            len_ = 0;
        }

        HostList(std::size_t cnt);

        HostList(T *data, std::size_t cnt) : data_(data), len_(cnt) {
            need_release_ = false;
        }

//        HostList(const T *copy_from, std::size_t cnt);

        HostList(const std::vector<T> &a);

        HostList(HostList &&arr) {
            data_ = arr.data_;
            len_ = arr.len_;
            arr.data_ = nullptr;
            arr.len_ = 0;
        }

        ~HostList();

//        HostList &operator=(const HostList &r) = delete;
//
//        HostList &operator=(HostList &&from) {
//            delete[] data_;
//            data_ = from.data_;
//            len_ = from.len_;
//            from.data_ = nullptr;
//            from.len_ = 0;
//            return *this;
//        }

        HostList(const HostList &r) = delete;

        HostList<T> copy() const {
            // need to cast data into const pointer
            // to make sure the contents are copied.
            const T *const_ptr = static_cast<const T *>(data_);
            return HostList<T>(const_ptr, len_);
        }

        const T &operator[](std::size_t i) const {
            return data_[i];
        }

        T &operator[](std::size_t i) {
            return data_[i];
        }

        void set_data(T *data, std::size_t cnt) {
            data_ = data;
            len_ = cnt;
            need_release_ = false;
        }

        T *get() {
            return data_;
        }

        const T *get() const {
            return data_;
        }
    };

    template<typename T>
    class HostGroup {

    public:

        HostGroup() = default;

        HostGroup(T *data, size_t row, size_t col) {
            size_ = row;
            lists_ = new HostList<T>[size_];
            auto p = data;
            for (int i = 0; i < size_; ++i) {
                lists_[i].set_data(p, col);
                p += col;
            }
        }

        ~HostGroup() {
            delete[] lists_;
        }

        void set_data(T *data, size_t row, size_t col) {
            size_ = col;
            lists_ = new HostList<T>[size_];
            auto p = data;
            for (int i = 0; i < size_; ++i) {
                lists_[i].set_data(p, row);
                p += row;
            }
        }

        HostList<T> *next_list() {
            if (offset_ >= size_) {
                return nullptr;
            }
            return lists_ + offset_++;
        }

    private:

        HostList<T> *lists_;
        size_t size_;
        size_t offset_ = 0;

    };

} // sigma
// util

#endif //CUSEAL_HOSTLIST_H
