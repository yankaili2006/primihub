#ifndef SRC_PRIMIHUB_PRIMITIVE_OPRF_OPRF_H_
#define SRC_PRIMIHUB_PRIMITIVE_OPRF_OPRF_H_

#include <string>
#include <vector>
#include <memory>
#include "openssl/evp.h"
#include "openssl/ec.h"

namespace primihub::oprf {

struct OprfKey {
  BIGNUM* sk{nullptr};
  EC_POINT* pk{nullptr};
  EC_GROUP* group{nullptr};
};

class OprfSender {
 public:
  OprfSender();
  ~OprfSender();
  OprfKey& key() { return key_; }
  const OprfKey& key() const { return key_; }
  std::vector<uint8_t> Evaluate(const std::vector<uint8_t>& blinded_input);
  std::vector<uint8_t> BlindEvaluate(const std::vector<uint8_t>& input);

 private:
  OprfKey key_;
  void InitKey();
};

class OprfReceiver {
 public:
  OprfReceiver();
  std::vector<uint8_t> Blind(const std::vector<uint8_t>& input);
  std::vector<uint8_t> Finalize(const std::vector<uint8_t>& input,
                                const std::vector<uint8_t>& evaluated,
                                const BIGNUM* r);
  BIGNUM* r() { return r_; }

 private:
  BIGNUM* r_{nullptr};
  EC_GROUP* group_{nullptr};
};

}  // namespace primihub::oprf

#endif  // SRC_PRIMIHUB_PRIMITIVE_OPRF_OPRF_H_
