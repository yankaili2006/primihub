/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Database implementation. Ports the bookkeeping from upstream
 * simplepir's pir/database.go + pir/utils.go.
 */
#include "src/primihub/kernel/pir/operator/pir_core/database.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

#include <glog/logging.h>

namespace primihub::pir::core {

uint64_t BaseP(uint64_t p, uint64_t m, uint64_t i) {
  if (p == 0) {
    LOG(FATAL) << "BaseP: p == 0";
  }
  for (uint64_t j = 0; j < i; ++j) {
    m /= p;
  }
  return m % p;
}

uint64_t ReconstructFromBaseP(uint64_t p, const uint64_t* vals,
                              std::size_t n) {
  uint64_t res = 0;
  uint64_t coeff = 1;
  for (std::size_t i = 0; i < n; ++i) {
    res += coeff * vals[i];
    coeff *= p;
  }
  return res;
}

uint64_t ComputeNumEntriesBaseP(uint64_t p, uint64_t log_q) {
  if (p < 2) {
    LOG(FATAL) << "ComputeNumEntriesBaseP: p < 2 (got " << p << ")";
  }
  const double log_p = std::log2(static_cast<double>(p));
  if (log_p <= 0.0) {
    LOG(FATAL) << "ComputeNumEntriesBaseP: log2(p) <= 0 (p=" << p << ")";
  }
  return static_cast<uint64_t>(
      std::ceil(static_cast<double>(log_q) / log_p));
}

retcode NumDbEntries(uint64_t n, uint64_t row_length, uint64_t p,
                     uint64_t* db_elems, uint64_t* elems_per_entry,
                     uint64_t* entries_per_elem, std::string* err) {
  if (n == 0 || row_length == 0 || p < 2) {
    if (err) {
      std::ostringstream oss;
      oss << "NumDbEntries: invalid input n=" << n
          << " row_length=" << row_length << " p=" << p;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  const double log_p = std::log2(static_cast<double>(p));
  if (static_cast<double>(row_length) <= log_p) {
    // Pack multiple entries into one Z_p elem.
    const uint64_t logp_floor = static_cast<uint64_t>(log_p);
    const uint64_t per_elem = logp_floor / row_length;
    if (per_elem == 0) {
      if (err) {
        std::ostringstream oss;
        oss << "NumDbEntries: per_elem = floor(log_p)/row_length = 0 "
            << "for p=" << p << " row_length=" << row_length;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    const uint64_t entries = static_cast<uint64_t>(
        std::ceil(static_cast<double>(n) / static_cast<double>(per_elem)));
    if (entries == 0 || entries > n) {
      if (err) {
        std::ostringstream oss;
        oss << "NumDbEntries: computed entries=" << entries
            << " out of range for n=" << n;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    *db_elems = entries;
    *elems_per_entry = 1;
    *entries_per_elem = per_elem;
    return retcode::SUCCESS;
  }

  // Need multiple Z_p elems per DB entry.
  // log_q is implicit at 32 for SimplePIR's current parameter table;
  // callers that need a different log_q should call
  // ComputeNumEntriesBaseP themselves. Keep the wire matching the Go
  // version by using 32 here too — both code paths land at row=N*ne.
  const uint64_t ne = ComputeNumEntriesBaseP(p, /*log_q=*/32);
  *db_elems = n * ne;
  *elems_per_entry = ne;
  *entries_per_elem = 0;
  return retcode::SUCCESS;
}

retcode ApproxSquareDatabaseDims(uint64_t n, uint64_t row_length,
                                  uint64_t p,
                                  uint64_t* l, uint64_t* m,
                                  std::string* err) {
  uint64_t db_elems = 0, elems_per_entry = 0, entries_per_elem = 0;
  auto rc = NumDbEntries(n, row_length, p, &db_elems, &elems_per_entry,
                         &entries_per_elem, err);
  if (rc != retcode::SUCCESS) return rc;

  uint64_t l_val = static_cast<uint64_t>(
      std::floor(std::sqrt(static_cast<double>(db_elems))));
  if (elems_per_entry == 0) elems_per_entry = 1;
  const uint64_t rem = l_val % elems_per_entry;
  if (rem != 0) l_val += elems_per_entry - rem;
  if (l_val == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "ApproxSquareDatabaseDims: degenerate l=0 from db_elems="
          << db_elems;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  uint64_t m_val = static_cast<uint64_t>(
      std::ceil(static_cast<double>(db_elems) / static_cast<double>(l_val)));
  *l = l_val;
  *m = m_val;
  return retcode::SUCCESS;
}

retcode Database::SetupShape(uint64_t num, uint64_t row_length,
                             const LweParams& params, std::string* err) {
  if (num == 0 || row_length == 0) {
    if (err) {
      *err = "Database::SetupShape: num and row_length must be > 0";
    }
    return retcode::FAIL;
  }
  if (params.p == 0 || params.logq == 0) {
    if (err) {
      *err =
          "Database::SetupShape: LweParams not initialized (call "
          "LweParams::Pick first; p or logq is zero)";
    }
    return retcode::FAIL;
  }

  uint64_t db_elems = 0, elems_per_entry = 0, entries_per_elem = 0;
  auto rc = NumDbEntries(num, row_length, params.p, &db_elems,
                          &elems_per_entry, &entries_per_elem, err);
  if (rc != retcode::SUCCESS) return rc;

  info_.num = num;
  info_.row_length = row_length;
  info_.p = params.p;
  info_.logq = params.logq;
  info_.ne = elems_per_entry;
  info_.x = elems_per_entry;
  info_.packing = entries_per_elem;
  info_.basis = 0;
  info_.squishing = 0;
  info_.cols = 0;

  // Make `x` divide `ne` — for the current parameter table this is
  // already true (ne small) but the loop matches upstream.
  while (info_.ne > 0 && info_.ne % info_.x != 0) {
    info_.x += 1;
  }

  if (db_elems > params.l * params.m) {
    if (err) {
      std::ostringstream oss;
      oss << "Database::SetupShape: db_elems=" << db_elems
          << " exceeds matrix capacity l*m=" << (params.l * params.m)
          << " (LweParams probably not sized for n=" << num << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (info_.ne > 0 && params.l % info_.ne != 0) {
    if (err) {
      std::ostringstream oss;
      oss << "Database::SetupShape: ne=" << info_.ne
          << " must divide params.l=" << params.l;
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  data_ = Matrix::Zeros(params.l, params.m);
  return retcode::SUCCESS;
}

Database Database::MakeRandom(uint64_t num, uint64_t row_length,
                              const LweParams& params, std::string* err,
                              retcode* rc) {
  Database db;
  auto setup = db.SetupShape(num, row_length, params, err);
  if (setup != retcode::SUCCESS) {
    if (rc) *rc = setup;
    return db;
  }
  // Uniform Z_p data — UniformRandom is bounded by 2^logmod, so we use
  // logmod=ceil(log2(p)) and then ReduceMod to fold into [0, p).
  const uint64_t log_p = static_cast<uint64_t>(
      std::ceil(std::log2(static_cast<double>(params.p))));
  db.data_ = Matrix::UniformRandom(params.l, params.m,
                                   static_cast<uint32_t>(log_p));
  db.data_.ReduceMod(static_cast<uint32_t>(params.p));
  // Shift into [-p/2, p/2] — matches upstream's `D.Data.Sub(p / 2)`.
  db.data_.ScalarSub(static_cast<uint32_t>(params.p / 2));
  if (rc) *rc = retcode::SUCCESS;
  return db;
}

uint64_t ReconstructElem(std::vector<uint64_t> vals, uint64_t index,
                          const DBinfo& info) {
  if (info.p == 0) {
    LOG(FATAL) << "ReconstructElem: info.p == 0";
  }
  if (info.logq == 0 || info.logq > 63) {
    LOG(FATAL) << "ReconstructElem: invalid info.logq=" << info.logq;
  }
  const uint64_t q = uint64_t{1} << info.logq;
  for (std::size_t i = 0; i < vals.size(); ++i) {
    vals[i] = (vals[i] + info.p / 2) % q;
    vals[i] = vals[i] % info.p;
  }
  uint64_t val = ReconstructFromBaseP(info.p, vals.data(), vals.size());
  if (info.packing > 0) {
    // Multiple DB entries packed in one Z_p; extract the right one.
    const uint64_t row_modulus = uint64_t{1} << info.row_length;
    val = BaseP(row_modulus, val, index % info.packing);
  }
  return val;
}

retcode Database::Squish(uint64_t basis, uint64_t squishing,
                         std::string* err) {
  if (basis == 0 || squishing == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "Database::Squish: invalid input basis=" << basis
          << " squishing=" << squishing
          << " (both must be > 0; upstream picks 10/3)";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (info_.squishing != 0) {
    if (err) {
      *err =
          "Database::Squish: already squished. Call Unsquish first or"
          " operate on a fresh Database.";
    }
    return retcode::FAIL;
  }
  // Mirrors upstream's panic-on-bad-params check, surfaced as FAIL.
  if (info_.p > (uint64_t{1} << basis)) {
    if (err) {
      std::ostringstream oss;
      oss << "Database::Squish: p=" << info_.p << " > 2^basis (basis="
          << basis
          << "). Increase basis or pick a smaller p row in LweParams.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (info_.logq < basis * squishing) {
    if (err) {
      std::ostringstream oss;
      oss << "Database::Squish: logq=" << info_.logq
          << " < basis*squishing=" << (basis * squishing)
          << ". Each packed Z_q slot must fit `squishing` `basis`-bit"
          << " Z_p values; the math runs but cells would alias.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  info_.basis = basis;
  info_.squishing = squishing;
  info_.cols = data_.cols();
  data_.Squish(basis, squishing);
  return retcode::SUCCESS;
}

retcode Database::Unsquish(std::string* err) {
  if (info_.squishing == 0) {
    if (err) {
      *err =
          "Database::Unsquish: not squished (info.squishing == 0)."
          " Squish must run first; this method is the inverse only.";
    }
    return retcode::FAIL;
  }
  data_.Unsquish(info_.basis, info_.squishing, info_.cols);
  info_.basis = 0;
  info_.squishing = 0;
  info_.cols = 0;
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::core
