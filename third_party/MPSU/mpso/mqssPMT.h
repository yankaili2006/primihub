// COPYRIGHT 2023
// Author: lx-1234

#pragma once

#include "ssVODM.h"
#include <string>
#include <vector>
#include "../common/Defines.h"
#include "../common/util.h"

namespace mpsu {

void mqssPMTSend(u32 numElements, LowMC &cipher,
                 BitVector &out, std::vector<block> &okvs,
                 Socket &chl, u32 numThreads, std::string triplePath = "");

// we don't need cipher's key, only use round constants
void mqssPMTRecv(std::vector<block> &set, LowMC &cipher,
                 BitVector &out, Socket &chl,
                 u32 numThreads, std::string triplePath = "");

}  // namespace mpsu
