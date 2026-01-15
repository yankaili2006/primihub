#pragma once

#include "pir.hpp"
#include "pir_client.hpp"
#include <map>
#include <memory>
#include <vector>
#include <sigma.h>
#include "kernelutils.cuh"

class PIRServer {
public:
  PIRServer(const sigma::EncryptionParameters &enc_params,
            const PirParams &pir_params);

  // NOTE: server takes over ownership of db and frees it when it exits.
  // Caller cannot free db
  void set_database(std::unique_ptr<std::vector<sigma::Plaintext>> &&db);
  //void set_database(const std::unique_ptr<const std::uint8_t[]> &bytes,
                    //std::uint64_t ele_num, std::uint64_t ele_size);//
  void set_database(uint8_t* bytes,
          std::uint64_t ele_num, std::uint64_t ele_size);
  void preprocess_database();
  std::vector<sigma::Ciphertext> expand_query(const sigma::Ciphertext &encrypted,
                                             std::uint32_t m,
                                             std::uint32_t client_id);

  PirQuery deserialize_query(std::stringstream &stream);
  PirReply generate_reply(PirQuery &query, std::uint32_t client_id);
  // Serializes the reply into the provided stream and returns the number of
  // bytes written
  int serialize_reply(PirReply &reply, std::stringstream &stream);

  void set_galois_key(std::uint32_t client_id, sigma::GaloisKeys galkey);

  void encode(uint64_t* values_matrix, uint64_t* destination, size_t values_matrix_size) const;

  void populate_matrix_reps_index_map();

  void sealpir_apply_galois(
            const sigma::Ciphertext &encrypted, std::uint32_t galois_elt,
            const sigma::GaloisKeys &galois_keys, sigma::Ciphertext &destination);

    // Below simple operations are for interacting with the database WITHOUT PIR.
  // So they can be used to modify a particular element in the database or
  // to query a particular element (without privacy guarantees).
  void simple_set(std::uint64_t index, sigma::Plaintext pt);
  sigma::Ciphertext simple_query(std::uint64_t index);
  void set_one_ct(sigma::Ciphertext one);
  sigma::util::HostArray<std::size_t> matrix_reps_index_map_;
  sigma::util::DeviceArray<std::size_t> Device_matrix_reps_index_map_;

private:
  sigma::EncryptionParameters enc_params_; // SEAL parameters
  PirParams pir_params_;                  // PIR parameters

  //sigma::EncryptionParameters device_enc_params_;
  //PirParams device_pir_params_;

  std::unique_ptr<Database> db_;
  //std::unique_ptr<Database> device_db_;
  bool is_db_preprocessed_;
  std::map<int, sigma::GaloisKeys> galoisKeys_;
  std::unique_ptr<sigma::Evaluator> evaluator_;
  std::unique_ptr<sigma::BatchEncoder> encoder_;
  std::shared_ptr<sigma::SIGMAContext> context_;

  // This is only used for simple_query
  sigma::Ciphertext one_;

  void multiply_power_of_X(const sigma::Ciphertext &encrypted,
                           sigma::Ciphertext &destination, std::uint32_t index);
};
__global__ void g_set_coefficients(uint64_t coeff_per_ptxt, uint32_t logt,
                                   uint8_t* bytes, uint64_t ele_size, uint64_t bytes_per_ptxt,
                                   uint64_t  coeff_per_element, uint64_t* plaintext,
                                   uint64_t num_of_plaintexts, uint64_t db_size,
                                   uint64_t* result, uint32_t N, uint32_t slot_count);
