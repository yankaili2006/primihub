// COPYRIGHT 2023
// Author: lx-1234

#pragma once

#include <volePSI/GMW/Gmw.h>
#include <coproto/Socket/AsioSocket.h>
#include <fstream>
#include <string>
#include "../circuit/Circuit.h"
#include "../lowMC/LowMC.h"

namespace mpsu {

void twoPartyTripleGen(u32 myIdx, u32 idx, u32 numElements,
                       u32 numThreads, Socket &chl, std::string fileName);

void tripleGenParty(u32 idx, u32 numParties, u32 numElements, u32 numThreads);

}  // namespace mpsu
