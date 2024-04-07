// COPYRIGHT 2023
// Author: lx-1234

#pragma once

#include "LowMC.h"
#include <string>
#include "../common/Defines.h"

namespace mpsu {

void mblockReverse(mblock &x);

// Sender holds ciphertext, cipher could be any key,
// we only need to know the round constants
void lowMCDecryptGmwSend(LowMC &cipher, mMatrix<mblock> &in,
                         Socket &chl, u32 numThreads, std::string triplePath);

// Receiver holds key
void lowMCDecryptGmwRecv(LowMC &cipher, mMatrix<mblock> &in,
                         Socket &chl, u32 numThreads, std::string triplePath);

}  // namespace mpsu
