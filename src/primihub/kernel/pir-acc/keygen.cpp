
#include <iostream>
#include <fstream>
#include <sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/keyutil.h"

const std::string public_key_data_path = "../data/public_key.dat";
const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string galois_keys_data_path = "../data/galois_keys.dat";

int main() {

    TIMER_START;

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");

    // TODO: remove @wangshuchao
    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters parms(sigma::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    parms.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));

    sigma::SIGMAContext context(parms);

    sigma::KeyGenerator keygen(context);

//    sigma::PublicKey public_key;
//    keygen.create_public_key(public_key);
//    util::save_public_key(public_key, public_key_data_path);

    sigma::SecretKey secret_key;
    secret_key = keygen.secret_key();
    util::save_secret_key(secret_key, secret_key_data_path);

//    sigma::GaloisKeys galois_keys;
//    keygen.create_galois_keys(galois_keys);
//    util::save_galois_keys(galois_keys, galois_keys_data_path);

    TIMER_PRINT_NOW(KeyGen);

    return 0;
}
