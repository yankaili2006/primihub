//
// Created by scwang on 2023/11/12.
//

#include <string>
#include <vector>
#include "util/HostList.h"

#ifndef CUSEAL_VECTORUTIL_H
#define CUSEAL_VECTORUTIL_H

namespace util {

    std::vector<std::vector<float>> read_npy_data(const std::string &npy_name);

    std::vector<std::vector<double>> batch_read_npy_data(const std::string &npy_name, size_t batch_size, double scale);

    float *read_formatted_npy_data(const std::string &npy_name, size_t slots, float scale, size_t &size);

    class NPYReader {

    public:

        NPYReader(const std::string &npy_name, size_t slots);

        ~NPYReader();

        sigma::util::HostGroup<float> *read_data(float scale);

    private:

        FILE *fp_;
        size_t row_;
        size_t col_;
        size_t word_size_;
        float *temp_;

    };

} // util

#endif //CUSEAL_VECTORUTIL_H
