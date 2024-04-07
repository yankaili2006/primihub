// COPYRIGHT 2023
// Author: lx-1234

#pragma once


#include <cryptoTools/Common/CLP.h>
#include <iostream>
#include <bitset>
#include <vector>
#include "Defines.h"

namespace mpsu {

// permute data according to pi
void permute(std::vector<u32> &pi, std::vector<block> &data);

void printPermutation(std::vector<u32> &pi);

void blockToBitset(block &data, std::bitset<128> &out);

void bitsetToBlock(std::bitset<128> &data, block &out);

}  // namespace mpsu
