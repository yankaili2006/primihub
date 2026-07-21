/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * RAII wrapper over the underhood/rlwe BFV context (ported from
 * underhood/underhood/params.go `params`), tiptoe chunk 1.1c. Real mode only:
 * includes @underhood//:rlwe (rlwe.h), which needs the SEAL toolchain, so this
 * header is compiled only under --define=enable_tiptoe_real=1. See
 * docs/pir/tiptoe-port-plan.md.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PARAMS_H_

#include <cstddef>
#include <cstdint>

#include "rlwe.h"  // @underhood//:rlwe (includes = ["rlwe"])

namespace primihub::pir::tiptoe {

// Owns the underhood/rlwe BFV context (context_new/context_free). Non-copyable,
// movable -- mirrors params.go's `params` + Free() (Go relies on an explicit
// Free() for the C++ object; here RAII handles it). The context fixes the BFV
// parameters n=2048, p=65537, logq=38.
class Params {
 public:
  Params() : ctx_(context_new()) {}
  ~Params() {
    if (ctx_ != nullptr) context_free(ctx_);
  }

  Params(const Params&) = delete;
  Params& operator=(const Params&) = delete;

  Params(Params&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }
  Params& operator=(Params&& other) noexcept {
    if (this != &other) {
      if (ctx_ != nullptr) context_free(ctx_);
      ctx_ = other.ctx_;
      other.ctx_ = nullptr;
    }
    return *this;
  }

  // Underlying rlwe context, for the client/server layers (chunks 1.1d/1.1e).
  context_t* ctx() const { return ctx_; }

  std::size_t N() const { return context_n(ctx_); }       // poly modulus degree
  std::uint64_t P() const {                                // plaintext modulus
    return static_cast<std::uint64_t>(context_p(ctx_));
  }
  std::size_t LogQ() const { return context_logq(ctx_); }  // coeff modulus bits

 private:
  context_t* ctx_;
};

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PARAMS_H_
