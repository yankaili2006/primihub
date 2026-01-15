//
// Created by scwang on 2023/12/26.
//

#include <vector>
#include "HostList.h"


namespace sigma::util {
    template<typename T>
    HostList<T>::HostList(std::size_t cnt) {
        if (cnt > 0) {
            cudaMallocHost((void **)&data_, sizeof(T) * cnt);
            memset(data_, 0, sizeof(T) * cnt);
        } else {
            data_ = nullptr;
        }
        len_ = cnt;
    }

//    template<typename T>
//    HostList<T>::HostList(const T *copy_from, std::size_t cnt) {
//        if (cnt == 0) {
//            data_ = nullptr;
//            len_ = 0;
//            return;
//        }
//        cudaMallocHost(data_, cnt);
//        for (std::size_t i = 0; i < cnt; i++) data_[i] = copy_from[i];
//        len_ = cnt;
//    }

    template<typename T>
    HostList<T>::HostList(const std::vector<T> &a) {
        len_ = a.size();
        cudaMallocHost((void **)&data_, len_ * sizeof(T));
        for (std::size_t i = 0; i < len_; i++) {
            data_[i] = a[i];
        }
    }

    template<typename T>
    HostList<T>::~HostList() {
        if (need_release_) {
            cudaFreeHost(data_);
        }
    }


} // sigma
// util