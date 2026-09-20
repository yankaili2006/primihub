/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/pirana_pir/pirana_pir.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "client.h"
#include "pir_parms.h"
#include "server.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

namespace {

constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kInPayloadSize = "payload_size";
constexpr const char* kOutRecovered = "recovered";

constexpr std::uint64_t kDefaultPayloadSize = 256;
constexpr std::uint64_t kMaxPayloads = 1ULL << 24;  // sanity bound

bool ParseU64(const std::string& s, std::uint64_t* out) {
  if (s.empty()) return false;
  std::uint64_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const std::uint64_t next = acc * 10 + static_cast<std::uint64_t>(c - '0');
    if (next < acc) return false;  // overflow
    acc = next;
  }
  *out = acc;
  return true;
}

// ---- bit packing: raw payload bytes <-> 17-bit plaintext slots ----
// PirParms uses prime_len=18 → each BatchEncoder slot holds a plain_modulus
// value < 2^17. Bytes are packed sequentially into consecutive slots
// (little-endian bit order); the tail of the last slot is zero.

std::vector<std::vector<std::uint64_t>> PackDb(
    const std::vector<std::string>& blobs, std::uint64_t payload_size,
    std::uint64_t num_payload_slot) {
  std::vector<std::vector<std::uint64_t>> db;
  db.reserve(blobs.size());
  for (const auto& blob : blobs) {
    std::vector<std::uint64_t> slots(num_payload_slot, 0);
    // 17 bits per slot; pack 8-bit bytes into the bit stream.
    std::uint64_t bit_off = 0;
    const std::uint64_t total_bits =
        std::min<std::uint64_t>(blob.size(), payload_size) * 8;
    const auto* bytes = reinterpret_cast<const unsigned char*>(blob.data());
    while (bit_off < total_bits) {
      const std::uint64_t slot = bit_off / 17;
      // gather up to 17 bits (may span 3 bytes)
      std::uint64_t v = 0;
      for (std::uint64_t b = 0; b < 17; ++b) {
        const std::uint64_t abs_bit = bit_off + b;
        if (abs_bit >= total_bits) break;
        const std::uint64_t src_byte = abs_bit / 8;
        const std::uint64_t src_bit = abs_bit % 8;
        v |= (static_cast<std::uint64_t>((bytes[src_byte] >> src_bit) & 1))
             << b;
      }
      slots[slot] = v;
      bit_off += 17;
    }
    db.push_back(std::move(slots));
  }
  return db;
}

std::string UnpackPayload(const std::vector<std::uint64_t>& slots,
                          std::uint64_t payload_size) {
  std::string out(payload_size, '\0');
  auto* bytes = reinterpret_cast<unsigned char*>(&out[0]);
  const std::uint64_t total_bits = payload_size * 8;
  for (std::uint64_t bit_off = 0; bit_off < total_bits; bit_off += 17) {
    const std::uint64_t slot = bit_off / 17;
    if (slot >= slots.size()) break;
    const std::uint64_t v = slots[slot];
    for (std::uint64_t b = 0; b < 17; ++b) {
      const std::uint64_t abs_bit = bit_off + b;
      if (abs_bit >= total_bits) break;
      bytes[abs_bit / 8] |=
          static_cast<unsigned char>(((v >> b) & 1) << (abs_bit % 8));
    }
  }
  return out;
}

// Reconstruct the queried payload from the single-query answer matrix.
// Slot traversal mirrors upstream PIRANA test.cc::test_pir_correctness —
// there every visited (i, index) position is asserted equal to
// real_item[count]; here the same traversal assigns answer values into
// payload[count], which recovers the payload slot sequence.
std::vector<std::uint64_t> ExtractSinglePayload(
    const std::vector<std::vector<std::uint64_t>>& answer, std::uint32_t index,
    PirParms& parms) {
  const std::uint64_t N = parms.get_seal_parms().poly_modulus_degree();
  const std::uint64_t half_N = N / 2;
  const std::uint64_t num_slot = parms.get_num_payload_slot();
  std::vector<std::uint64_t> payload(num_slot, 0);
  const std::uint64_t row_index = index / half_N;
  const std::uint64_t left_rotate_slot_0 =
      static_cast<std::uint64_t>(std::ceil(
          static_cast<double>(num_slot) / parms.get_pre_rotate()));
  const std::uint64_t offset =
      (index - std::min(left_rotate_slot_0, parms.get_rotate_step())) % half_N;
  std::uint64_t count = 0;
  std::uint64_t left_rotate_slot = left_rotate_slot_0;
  for (std::uint32_t i = 0; i < answer.size() && count < num_slot; i++) {
    for (std::uint32_t j = 0;
         j < parms.get_rotate_step() && count < num_slot; j++) {
      for (std::uint32_t k = 0;
           k < parms.get_pre_rotate() / 2 && count < num_slot; k++) {
        std::uint32_t pos = static_cast<std::uint32_t>(
            row_index * half_N +
            (k * parms.get_rotate_step() + j + offset) % half_N);
        payload[count] = answer[i][pos % N];
        count++;
      }
      for (std::uint32_t k = 0;
           k < parms.get_pre_rotate() / 2 && count < num_slot; k++) {
        std::uint32_t pos = static_cast<std::uint32_t>(
            (row_index + 1) * half_N +
            (k * parms.get_rotate_step() + j + offset) % half_N);
        payload[count] = answer[i][pos % N];
        count++;
      }
    }
    left_rotate_slot -= parms.get_rotate_step();
  }
  if (count < num_slot) {
    LOG(WARNING) << "PiranaPirOperator: single-query traversal recovered "
                 << count << "/" << num_slot << " slots";
  }
  return payload;
}

}  // namespace

retcode PiranaPirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "PiranaPirOperator: result is null";
    return retcode::FAIL;
  }
  // ---- Parse inputs ----
  const auto db_it = input.find(kInDb);
  if (db_it == input.end() || db_it->second.empty()) {
    LOG(ERROR) << "PiranaPirOperator: input missing non-empty '" << kInDb
               << "' (vector of raw payload byte strings)";
    return retcode::FAIL;
  }
  const auto idx_it = input.find(kInIndices);
  if (idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "PiranaPirOperator: input missing non-empty '"
               << kInIndices << "' (vector of decimal index strings)";
    return retcode::FAIL;
  }
  const auto& blobs = db_it->second;
  const auto& idx_strs = idx_it->second;
  const std::uint64_t n = blobs.size();
  if (n > kMaxPayloads) {
    LOG(ERROR) << "PiranaPirOperator: db_content size " << n
               << " exceeds sanity bound " << kMaxPayloads;
    return retcode::FAIL;
  }

  std::vector<std::uint32_t> indices;
  indices.reserve(idx_strs.size());
  for (std::size_t i = 0; i < idx_strs.size(); ++i) {
    std::uint64_t idx = 0;
    if (!ParseU64(idx_strs[i], &idx) || idx >= n) {
      LOG(ERROR) << "PiranaPirOperator: query_indices[" << i << "]='"
                 << idx_strs[i] << "' is not a valid index in [0, " << n
                 << ")";
      return retcode::FAIL;
    }
    indices.push_back(static_cast<std::uint32_t>(idx));
  }

  // ---- Payload size: max blob length (uniform slot count for all rows) ----
  std::uint64_t payload_size = kDefaultPayloadSize;
  const auto ps_it = input.find(kInPayloadSize);
  if (ps_it != input.end() && !ps_it->second.empty()) {
    if (!ParseU64(ps_it->second[0], &payload_size) || payload_size == 0) {
      LOG(ERROR) << "PiranaPirOperator: invalid payload_size '"
                 << ps_it->second[0] << "'";
      return retcode::FAIL;
    }
  } else {
    for (const auto& b : blobs) {
      payload_size = std::max<std::uint64_t>(payload_size, b.size());
    }
  }

  try {
    const bool batch = indices.size() > 1;
    std::vector<std::vector<std::uint64_t>> recovered_slots;

    if (!batch) {
      // ---- Single-query PIR ----
      PirParms parms(n, payload_size);
      Client client(parms);
      std::stringstream keys = client.save_keys();
      Server server(parms, /*random_db=*/true);
      server.set_keys(keys);
      server.load_external_db(
          PackDb(blobs, payload_size, parms.get_num_payload_slot()));

      std::stringstream query = client.gen_query(indices[0]);
      std::stringstream response = server.gen_response(query);
      auto answer = client.extract_answer(response);
      recovered_slots.push_back(
          ExtractSinglePayload(answer, indices[0], parms));
    } else {
      // ---- Multi-query (batch) PIR: cuckoo-hash bundles ----
      const std::uint64_t num_query = indices.size();
      const bool is_compress = true;  // upstream default (README -c 1)
      PirParms parms(n, payload_size, num_query, /*is_batch=*/true,
                     is_compress);
      Client client(parms);
      std::stringstream keys = client.save_keys();
      Server server(parms, /*is_batch=*/true, /*random_db=*/true);
      server.set_keys(keys);
      server.load_external_db(
          PackDb(blobs, payload_size, parms.get_num_payload_slot()));

      std::stringstream query = client.gen_batch_query(indices);
      std::stringstream response = server.gen_batch_response(query);
      auto answer = client.extract_batch_answer(response);

      // Slot extraction mirrors test.cc::test_batch_pir_correctness.
      const auto table = parms.get_cuckoo_table();
      const std::uint64_t N =
          parms.get_seal_parms().poly_modulus_degree();
      for (std::uint32_t q : indices) {
        kuku::QueryResult res = table->query(kuku::make_item(0, q));
        if (!res.found()) {
          LOG(ERROR) << "PiranaPirOperator: cuckoo query for index " << q
                     << " not found in table";
          return retcode::FAIL;
        }
        const auto loc = res.location();
        std::vector<std::uint64_t> slots(
            parms.get_num_payload_slot(), 0);
        if (!parms.get_is_compress()) {
          const auto bundle_size = parms.get_bundle_size();
          for (std::uint32_t i = 0; i < parms.get_num_payload_slot(); i++) {
            const auto slot_index = loc % N;
            const auto bundle_index = loc / N;
            slots[i] =
                answer.at(bundle_size * i + bundle_index).at(slot_index);
          }
        } else {
          const auto num_slot = parms.get_num_slot();
          std::uint32_t slot = 0, ct_index = 0;
          for (std::uint32_t i = 0; i < slots.size(); i++, slot++) {
            if (slot == num_slot) {
              slot = 0;
              ct_index++;
            }
            slots[i] = answer.at(ct_index).at(slot + loc * num_slot);
          }
        }
        recovered_slots.push_back(std::move(slots));
      }
    }

    // ---- Unpack to raw bytes ----
    std::vector<std::string> recovered;
    recovered.reserve(recovered_slots.size());
    for (auto& slots : recovered_slots) {
      recovered.push_back(UnpackPayload(slots, payload_size));
    }
    (*result)[kOutRecovered] = std::move(recovered);
    return retcode::SUCCESS;
  } catch (const std::exception& e) {
    LOG(ERROR) << "PiranaPirOperator: SEAL/Kuku exception: " << e.what();
    return retcode::FAIL;
  }
}

namespace {

PirCapabilities PiranaCaps() {
  PirCapabilities caps;
  caps.is_real = true;  // full SEAL pipeline through OnExecute
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = false;
  caps.hint_per_database = false;
  caps.threat_model = ThreatModel::SemiHonest;
  // Upstream demo: single-query gen_response is ms-scale at n=16384/x=256;
  // batch mode amortizes rotation cost across queries.
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 1'048'576;  // 2^20, BFV poly 4096 regime
  caps.backends = {Backend::CPU};
  // Constant-weight (hamming weight 2) codeword query: 2 ciphertexts.
  caps.typical_query_comm_bytes = 2 * 96 * 1024;
  caps.typical_hint_size_bytes = 0;  // Galois keys ~ a few MB, no DB hint
  return caps;
}

PirRegistrar<PiranaPirOperator> pirana_pir_registrar_("pirana", PiranaCaps());

}  // namespace

}  // namespace primihub::pir
