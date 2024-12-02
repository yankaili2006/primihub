//
// Created by scwang on 2023/11/12.
//

#include "keyutil.h"

#include <fstream>

namespace util {

    void save_public_key(const sigma::PublicKey& publicKey, const std::string& path) {
        std::ofstream pkofs(path, std::ios::binary);
        publicKey.save(pkofs);
        pkofs.close();
    }

    void save_secret_key(const sigma::SecretKey& secretKey, const std::string& path) {
        std::ofstream pkofs(path, std::ios::binary);
        secretKey.save(pkofs);
        pkofs.close();
    }

    void save_galois_keys(const sigma::GaloisKeys& galoisKeys, const std::string& path) {
        std::ofstream gkofs(path, std::ios::binary);
        galoisKeys.save(gkofs);
        gkofs.close();
    }

    void load_public_key(const sigma::SIGMAContext &context,
                       sigma::PublicKey &public_key,
                       const std::string &public_key_path) {
        std::ifstream pkifs(public_key_path, std::ios::binary);
        public_key.load(context, pkifs);
        pkifs.close();
        std::cout << "Public key loaded successfully." << std::endl;
    }

    void load_secret_key(const sigma::SIGMAContext &context,
                   sigma::SecretKey &secret_key,
                   const std::string &secret_key_path) {
        std::ifstream skifs(secret_key_path, std::ios::binary);
        secret_key.load(context, skifs);
        skifs.close();
        std::cout << "Secret key loaded successfully." << std::endl;
    }

    void load_galois_key(const sigma::SIGMAContext &context,
                         sigma::GaloisKeys &galois_keys,
                         const std::string &galois_keys_path) {
        std::ifstream gkifs(galois_keys_path, std::ios::binary);
        galois_keys.load(context, gkifs);
        gkifs.close();
        std::cout << "Secret key loaded successfully." << std::endl;
    }

} // util