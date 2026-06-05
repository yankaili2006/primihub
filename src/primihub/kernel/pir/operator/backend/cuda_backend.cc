/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/backend/cuda_backend.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace primihub::pir {

bool CudaBackend::Available() const {
#ifdef HAVE_CUDA
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) return false;
  return count > 0;
#else
  return false;
#endif
}

}  // namespace primihub::pir
