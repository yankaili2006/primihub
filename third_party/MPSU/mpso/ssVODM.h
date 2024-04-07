// COPYRIGHT 2023
// Author: lx-1234

#pragma once

#include <volePSI/GMW/Gmw.h>
#include <string>
#include "../circuit/Circuit.h"
#include "../lowMC/lowmcDecGmw.h"

namespace mpsu {

void ssVODMSend(LowMC &cipher, mMatrix<mblock> &in,
                BitVector &out, Socket &chl, u32 numThreads,
                std::string triplePath);

// todo: reconstruct the output will get the opposite result, why?
void ssVODMRecv(LowMC &cipher, mMatrix<mblock> &in,
                BitVector &out, Socket &chl, u32 numThreads,
                std::string triplePath);

}  // namespace mpsu
