#include "src/primihub/primitive/oprf/oprf.h"
#include <openssl/sha.h>
#include <cstring>
#include <stdexcept>

namespace primihub::oprf {

static const int CURVE_NID = NID_X9_62_prime256v1;  // P-256

OprfSender::OprfSender() { InitKey(); }
OprfSender::~OprfSender() {
  if (key_.sk) BN_free(key_.sk);
  if (key_.pk) EC_POINT_free(key_.pk);
  if (key_.group) EC_GROUP_free(key_.group);
}

void OprfSender::InitKey() {
  key_.group = EC_GROUP_new_by_curve_name(CURVE_NID);
  if (!key_.group) throw std::runtime_error("Failed to create EC group");
  EC_GROUP_set_point_conversion_form(key_.group, POINT_CONVERSION_COMPRESSED);
  key_.sk = BN_new();
  if (!key_.sk) throw std::runtime_error("Failed to create BIGNUM");
  key_.pk = EC_POINT_new(key_.group);
  if (!key_.pk) throw std::runtime_error("Failed to create EC_POINT");

  BIGNUM* order = BN_new();
  EC_GROUP_get_order(key_.group, order, nullptr);
  BN_rand_range(key_.sk, order);
  EC_POINT_mul(key_.group, key_.pk, key_.sk, nullptr, nullptr, nullptr);
  BN_free(order);
}

std::vector<uint8_t> OprfSender::Evaluate(const std::vector<uint8_t>& blinded_input) {
  if (blinded_input.empty()) return {};

  EC_POINT* blinded_point = EC_POINT_new(key_.group);
  EC_POINT_oct2point(key_.group, blinded_point, blinded_input.data(),
                     blinded_input.size(), nullptr);

  EC_POINT* result = EC_POINT_new(key_.group);
  EC_POINT_mul(key_.group, result, nullptr, blinded_point, key_.sk, nullptr);

  std::vector<uint8_t> out(65);
  EC_POINT_point2oct(key_.group, result, POINT_CONVERSION_UNCOMPRESSED,
                     out.data(), out.size(), nullptr);

  EC_POINT_free(blinded_point);
  EC_POINT_free(result);
  return out;
}

std::vector<uint8_t> OprfSender::BlindEvaluate(const std::vector<uint8_t>& input) {
  // Hash input to curve point
  uint8_t hash[32];
  SHA256(input.data(), input.size(), hash);
  BIGNUM* hash_bn = BN_bin2bn(hash, 32, nullptr);
  EC_POINT* point = EC_POINT_new(key_.group);
  EC_POINT_mul(key_.group, point, hash_bn, nullptr, nullptr, nullptr);
  EC_POINT_mul(key_.group, point, nullptr, point, key_.sk, nullptr);

  std::vector<uint8_t> out(33);
  EC_POINT_point2oct(key_.group, point, POINT_CONVERSION_COMPRESSED,
                     out.data(), out.size(), nullptr);
  BN_free(hash_bn);
  EC_POINT_free(point);
  return out;
}

OprfReceiver::OprfReceiver() {
  group_ = EC_GROUP_new_by_curve_name(CURVE_NID);
  EC_GROUP_set_point_conversion_form(group_, POINT_CONVERSION_COMPRESSED);
  r_ = BN_new();
  BIGNUM* order = BN_new();
  EC_GROUP_get_order(group_, order, nullptr);
  BN_rand_range(r_, order);
  BN_free(order);
}

std::vector<uint8_t> OprfReceiver::Blind(const std::vector<uint8_t>& input) {
  uint8_t hash[32];
  SHA256(input.data(), input.size(), hash);
  BIGNUM* hash_bn = BN_bin2bn(hash, 32, nullptr);

  EC_POINT* point = EC_POINT_new(group_);
  EC_POINT_mul(group_, point, hash_bn, nullptr, nullptr, nullptr);

  // Blind: M = H(x)^r
  EC_POINT* blinded = EC_POINT_new(group_);
  EC_POINT_mul(group_, blinded, nullptr, point, r_, nullptr);

  std::vector<uint8_t> out(33);
  EC_POINT_point2oct(group_, blinded, POINT_CONVERSION_COMPRESSED,
                     out.data(), out.size(), nullptr);

  EC_POINT_free(point);
  EC_POINT_free(blinded);
  BN_free(hash_bn);
  return out;
}

std::vector<uint8_t> OprfReceiver::Finalize(const std::vector<uint8_t>& input,
                                            const std::vector<uint8_t>& evaluated,
                                            const BIGNUM* r) {
  uint8_t hash[32];
  SHA256(input.data(), input.size(), hash);

  EC_POINT* evaluated_point = EC_POINT_new(group_);
  EC_POINT_oct2point(group_, evaluated_point, evaluated.data(),
                     evaluated.size(), nullptr);

  // Unblind: V = M'^{1/r}
  BIGNUM* r_inv = BN_new();
  BIGNUM* order = BN_new();
  EC_GROUP_get_order(group_, order, nullptr);
  BN_mod_inverse(r_inv, r, order, nullptr);

  EC_POINT* result = EC_POINT_new(group_);
  EC_POINT_mul(group_, result, nullptr, evaluated_point, r_inv, nullptr);

  std::vector<uint8_t> out(33);
  EC_POINT_point2oct(group_, result, POINT_CONVERSION_COMPRESSED,
                     out.data(), out.size(), nullptr);

  // Final hash to get the PRF output
  uint8_t final_hash[32];
  SHA256(out.data(), out.size(), final_hash);
  std::vector<uint8_t> final_out(final_hash, final_hash + 32);

  BN_free(r_inv);
  BN_free(order);
  EC_POINT_free(evaluated_point);
  EC_POINT_free(result);
  return final_out;
}

}  // namespace primihub::oprf
