// COPYRIGHT 2023
// Author: lx-1234


#pragma once

#include <cryptoTools/Circuit/BetaCircuit.h>
#include <cryptoTools/Circuit/Gate.h>
#include "../common/Defines.h"


using BetaCircuit = oc::BetaCircuit;
using BetaBundle = oc::BetaBundle;
using GateType = oc::GateType;

namespace mpsu {

BetaCircuit inverse_of_S_box_layer(u8 num_S_box);

// n is power of 2, should evaluate on a 64-bit input
BetaCircuit lessthanN(u32 n);

}  // namespace mpsu
