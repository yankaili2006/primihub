
#include <iostream>
#include <fstream>
#include <vector>
#include <sigma.h>
#include <queue>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string encrypted_c1_data_path = "../data/ip_results/encrypted_c1.dat";
const std::string results_data_path = "../data/ip_results/top_ip_results.json";

std::string ip_results_path(size_t index) {
    return "../data/ip_results/probe_" + std::to_string(index) + "_results.dat";
}

class TopNPairs {

private:

    size_t n_;
    std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> pq_;

public:

    explicit TopNPairs(size_t n) : n_(n) {}

    void add(const std::pair<double, size_t> &value) {
        if (pq_.size() < n_) {
            pq_.push(value);
        } else {
            if (value.first > pq_.top().first) {
                pq_.pop();
                pq_.push(value);
            }
        }
    }

    std::vector<std::pair<double, size_t>> getData() {
        std::vector<std::pair<double, size_t>> results;
        while (!pq_.empty()) {
            results.push_back(pq_.top());
            pq_.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

};

int main() {

    TIMER_START;

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
//    params.setup_device_params(); // 初始化device相关参数
    sigma::SIGMAContext context(params);
//    context.setup_device_params(); // 初始化device相关参数

    sigma::CKKSEncoder encoder(context);

    sigma::SecretKey secret_key;
    util::load_secret_key(context, secret_key, secret_key_data_path);
    sigma::Decryptor decryptor(context, secret_key);

    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    double customized_scale = pow(2.0, customized_scale_power);

//    std::ifstream c1_ifs(encrypted_c1_data_path, std::ios::binary);

    Json::Value root;
    for (size_t i = 0;; i++) {
        std::ifstream ifs(ip_results_path(i), std::ios::binary);
        if (!ifs.good()) {
            break;
        }

        sigma::Ciphertext c1;
        c1.use_half_data() = true;
        c1.load(context, ifs);

        TopNPairs pairs(5);
        size_t idx = 0;
        while (!ifs.eof()) {
            sigma::Ciphertext encrypted_vec;
            encrypted_vec.use_half_data() = true;
            try {
                encrypted_vec.load(context, ifs);
            } catch (const std::exception &e) {
                break;
            }
            sigma::Plaintext plaintext;
            decryptor.ckks_decrypt(encrypted_vec, c1, plaintext);
            std::vector<double> dest;
            encoder.decode(plaintext, dest);
            for (auto value: dest) {
                pairs.add(std::pair(value / customized_scale, idx++));
            }
//            std::cout << "Decrypt end " << idx << std::endl;
        }
        ifs.close();
        Json::Value ips;
        auto data = pairs.getData();
        for (auto pair: data) {
            Json::Value pairValue;
            pairValue["inner_product"] = pair.first;
            pairValue["index"] = pair.second;
            ips.append(pairValue);
        }
        root.append(ips);
    }

    std::ofstream outputFile(results_data_path);
    if (outputFile.is_open()) {
        Json::StreamWriterBuilder writerBuilder;
        writerBuilder["indentation"] = "    ";
        std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());
        writer->write(root, &outputFile);
        outputFile.close();
        std::cout << "Data successfully written to disk." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    TIMER_PRINT_NOW(Decrypt_and_decode);

    return 0;
}
