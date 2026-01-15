//
// Created by scwang on 2023/11/12.
//

#include "vectorutil.h"
#include "../extern/cnpy/cnpy.h"
#include "util/HostList.h"
#include <cmath>
#include <vector>

namespace util {

    double calculateMagnitude(const std::vector<float> &vector) {
        double sum = 0.0;
        for (double value: vector) {
            sum += value * value;
        }
        return std::sqrt(sum);
    }

    void normalizeVector(std::vector<float> &vector) {
        float magnitude = calculateMagnitude(vector);
        if (magnitude > 0.0) {
            for (float &value: vector) {
                value /= magnitude;
            }
        }
    }

    std::vector<std::vector<float>> read_npy_data(const std::string &npy_name) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t numRows = shape[0];
        size_t numCols = shape[1];
        auto *data = arr.data<float>();
        std::vector<std::vector<float>> matrix;
        matrix.reserve(numRows);
        for (size_t i = 0; i < numRows; ++i) {
            std::vector<float> row;
            row.reserve(numCols);
            for (size_t j = 0; j < numCols; ++j) {
                auto value = data[i * numCols + j];
                row.push_back(value);
            }
            normalizeVector(row);
            matrix.push_back(row);
        }
        return matrix;
    }

    std::vector<std::vector<double>> batch_read_npy_data(const std::string &npy_name, size_t batch_size, double scale) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t numRows = shape[0] / batch_size; // TODO: 未读取完整 @wangshuchao
        size_t numCols = shape[1] * batch_size;
        auto *data = arr.data<float>();
        std::vector<std::vector<double>> matrix;
        matrix.reserve(numRows);
        for (size_t i = 0; i < numRows; ++i) {
            std::vector<double> row;
            row.reserve(numCols);
            for (size_t j = 0; j < numCols; ++j) {
                auto value = (double) data[i * numCols + j] * scale;
                row.push_back(value);
            }
//            normalizeVector(row);
            matrix.push_back(row);
        }
        return matrix;
    }

    float *read_formatted_npy_data(const std::string &npy_name, size_t slots, float scale, size_t &size) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t single_size = 512;
        auto batch_size = slots / single_size;
        size_t numRows = shape[0] / batch_size; // TODO: 未读取完整 @wangshuchao
        size_t numCols = shape[1] * batch_size;
        size = numRows;

        auto data = arr.data<float>();
        auto matrix = new float[numRows * numCols];

//        01  02  11  12             01  11  21  31
//        21  22  31  32    ---->    02  12  22  32
//        41  42  51  52    ---->    41  51  61  71
//        61  62  71  72             42  52  62  72
        for (size_t offset = 0; offset < numRows - single_size; offset += single_size) {
            auto matrix_start = matrix + (offset * slots);
            auto data_start = data + (offset * slots);
            for (size_t i = 0; i < single_size; i++) {
                auto line_ptr = matrix_start + (i * slots);
                auto data_ptr = data_start + i;
                for (size_t j = 0; j < slots; j++) {
                    *(line_ptr + j) = *(data_ptr + j * single_size) * scale;
                }
            }
        }

        return matrix;
    }

    FILE *read_npy_header(const std::string &npy_name, size_t slots, double scale, size_t &size) {
        FILE *fp = fopen(npy_name.c_str(), "rb");
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

        cnpy::NpyArray arr(shape, word_size, fortran_order);
        size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
        if (nread != arr.num_bytes())
            throw std::runtime_error("load_the_npy_file: failed fread");

        return fp;
    }

    NPYReader::NPYReader(const std::string &npy_name, size_t slots) {
        fp_ = fopen(npy_name.c_str(), "rb");
        std::vector<size_t> shape;
        bool fortran_order;
        cnpy::parse_npy_header(fp_, word_size_, shape, fortran_order);
        size_t single_size = 512;
        auto batch_size = slots / 512;
        row_ = shape[0] / batch_size;
        col_ = slots;

        temp_ = new float[slots * 512];
    }

    sigma::util::HostGroup<float> *NPYReader::read_data(float scale) {
        size_t read_size = col_ * 4 * 512;
        size_t n_read = fread(temp_, 1, read_size, fp_);
        if (read_size != n_read) {
            return nullptr;
        }
        float *data = nullptr;
        cudaMallocHost((void **)&data, col_ * 512 * sizeof(float));

        auto data_start = data;
        auto temp_start = temp_;
        for (size_t i = 0; i < 512; i++) {
            auto line_ptr = data_start + (i * col_);
            auto data_ptr = temp_start + i;
            for (size_t j = 0; j < col_; j++) {
                *(line_ptr + j) = *(data_ptr + j * 512) * scale;
            }
        }

        return new sigma::util::HostGroup<float>(data, 512, col_);
    }

    NPYReader::~NPYReader() {
        delete[] temp_;
    }

} // util