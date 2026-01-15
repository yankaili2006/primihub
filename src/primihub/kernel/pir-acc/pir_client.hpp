#pragma once

#include "pir.hpp"
#include <memory>
#include <vector>

using namespace std;

class PIRClient {
public:
  PIRClient(const sigma::EncryptionParameters &encparms,
            const PirParams &pirparams);

  PirQuery generate_query(std::uint64_t desiredIndex);
  // Serializes the query into the provided stream and returns number of bytes
  // written
  int generate_serialized_query(std::uint64_t desiredIndex,
                                std::stringstream &stream);
  sigma::Plaintext decode_reply(PirReply &reply);

  std::vector<uint64_t> extract_coeffs(sigma::Plaintext pt);
  std::vector<uint64_t> extract_coeffs(sigma::Plaintext pt,
                                       std::uint64_t offset);
  std::vector<uint8_t> extract_bytes(sigma::Plaintext pt, std::uint64_t offset);

  std::vector<uint8_t> decode_reply(PirReply &reply, uint64_t offset);

  sigma::Plaintext decrypt(sigma::Ciphertext ct);

  sigma::GaloisKeys generate_galois_keys();

  // Index and offset of an element in an FV plaintext
  uint64_t get_fv_index(uint64_t element_index);
  uint64_t get_fv_offset(uint64_t element_index);

  // Only used for simple_query
  sigma::Ciphertext get_one();

  sigma::Plaintext replace_element(sigma::Plaintext pt,
                                  std::vector<std::uint64_t> new_element,
                                  std::uint64_t offset);

private:
  sigma::EncryptionParameters enc_params_;
  PirParams pir_params_;

  std::unique_ptr<sigma::Encryptor> encryptor_;
  std::unique_ptr<sigma::Decryptor> decryptor_;
  std::unique_ptr<sigma::Evaluator> evaluator_;
  std::unique_ptr<sigma::KeyGenerator> keygen_;
  std::unique_ptr<sigma::BatchEncoder> encoder_;
  std::shared_ptr<sigma::SIGMAContext> context_;

  vector<uint64_t> indices_; // the indices for retrieval.
  vector<uint64_t> inverse_scales_;

  friend class PIRServer;
};
