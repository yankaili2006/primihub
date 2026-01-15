//
// Created by scwang on 2023/11/12.
//

#include <sigma.h>

#ifndef CUSEAL_KEYUTIL_H
#define CUSEAL_KEYUTIL_H

#define TIMER_START auto __timer_start = std::chrono::high_resolution_clock::now()
#define TIMER_PRINT_NOW(name) \
    auto name = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - __timer_start); \
    std::cout << #name << " [" << name.count() << " milliseconds]" << std::endl

namespace util {

    void save_public_key(const sigma::PublicKey& publicKey, const std::string& path);
    void save_secret_key(const sigma::SecretKey& secretKey, const std::string& path);
    void save_galois_keys(const sigma::GaloisKeys& galoisKeys, const std::string& path);

    void load_public_key(const sigma::SIGMAContext &context,
                         sigma::PublicKey &public_key,
                         const std::string &public_key_path);
    void load_secret_key(const sigma::SIGMAContext &context,
                         sigma::SecretKey &secret_key,
                         const std::string &secret_key_path);
    void load_galois_key(const sigma::SIGMAContext &context,
                         sigma::GaloisKeys &galois_keys,
                         const std::string &galois_keys_path);

} // util

#endif //CUSEAL_KEYUTIL_H
