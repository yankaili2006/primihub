
#include <iostream>
#include <fstream>
#include <sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const std::string public_key_data_path = "../data/public_key.dat";
const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string encrypted_data_path = "../data/gallery.dat";
const std::string encrypted_c1_data_path = "../data/encrypted_c1.dat";
const static std::string FILE_STORE_PATH = "../vectors/";

int main() {

//    std::cout << "Encode and encrypt start" << std::endl;
//    auto time_start = std::chrono::high_resolution_clock::now();

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    float customized_scale = pow(2.0, float(customized_scale_power));

    auto slots = poly_modulus_degree / 2;

    size_t gallery_size = 0;
    auto gallery_ptr = util::read_formatted_npy_data(FILE_STORE_PATH + "gallery_x.npy", slots, customized_scale, gallery_size);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
//    params.setup_device_params(); // 初始化device相关参数
    sigma::SIGMAContext context(params);
//    context.setup_device_params(); // 初始化device相关参数

//    sigma::PublicKey public_key;
    sigma::SecretKey secret_key;
//    util::load_public_key(context, public_key, public_key_data_path);
    util::load_secret_key(context, secret_key, secret_key_data_path);

    secret_key.copy_to_device();

    sigma::CKKSEncoder encoder(context);
    sigma::Encryptor encryptor(context, secret_key);

    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    encryptor.sample_symmetric_ckks_c1(c1);
    std::ofstream c1_ofs(encrypted_c1_data_path, std::ios::binary);
    c1.save(c1_ofs);
    c1_ofs.close();

    c1.copy_to_device();

    std::ofstream ofs(encrypted_data_path, std::ios::binary);

    std::cout << "Encode and encrypt start" << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    sigma::Plaintext plain_vec;
    sigma::Ciphertext ciphertext;
    for (int i = 0; i < gallery_size; ++i) {
        auto vec = gallery_ptr + (i * slots);

        auto time_start0 = std::chrono::high_resolution_clock::now();

        encoder.encode_float(vec, slots, scale, plain_vec);

//        auto time_end0 = std::chrono::high_resolution_clock::now();
//        auto time_diff0 = std::chrono::duration_cast<std::chrono::microseconds >(time_end0 - time_start0);
//        std::cout << "encrypt file end [" << time_diff0.count() << " microseconds]" << std::endl;

//        auto time_start1 = std::chrono::high_resolution_clock::now();

        ciphertext.use_half_data() = true;
        encryptor.encrypt_symmetric_ckks(plain_vec, ciphertext, c1);

        ciphertext.retrieve_to_host();

//        auto time_end1 = std::chrono::high_resolution_clock::now();
//        auto time_diff1 = std::chrono::duration_cast<std::chrono::microseconds >(time_end1 - time_start1);
//        std::cout << "encrypt file end [" << time_diff1.count() << " microseconds]" << std::endl;
//        std::cout << std::endl << std::endl;

        ciphertext.save(ofs);
//        std::cout << "encrypt end " << i << std::endl;  // TODO: remove @wangshuchao
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;

    c1.release_device_data();

    ofs.close();

    delete[] gallery_ptr;

    return 0;
}
