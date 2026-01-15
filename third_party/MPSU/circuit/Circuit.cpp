// COPYRIGHT 2023
// Author: lx-1234


#include "Circuit.h"

namespace mpsu {

BetaCircuit inverse_of_S_box_layer(u8 num_S_box) {
    BetaCircuit cd;

    BetaBundle a(128);  // 初始128位为数据, 第129位为0, 第130位为1
    BetaBundle b(128);  // 初始128位为数据, 第129位为0, 第130位为1
    BetaBundle temp(5 * num_S_box);

    cd.addInputBundle(a);
    cd.addOutputBundle(b);
    cd.addTempWireBundle(temp);

    for (u8 i = 0; i < 128 - 3 * num_S_box; i++)
        cd.addCopy(a.mWires[i], b.mWires[i]);

    u8 j = 0;
    for (u8 i = 128 - 3 * num_S_box; i < 128; i+=3) {
        cd.addGate(a.mWires[i+1], a.mWires[i+2],
                   GateType::And, temp.mWires[j]);  // t0=z1*z2
        cd.addGate(a.mWires[i], a.mWires[i+1],
                   GateType::Xor, temp.mWires[j+1]);  // t1=z0+z1
        cd.addGate(temp.mWires[j], temp.mWires[j+1],
                   GateType::Xor, b.mWires[i]);  // x0=t0+t1

        cd.addGate(a.mWires[i], a.mWires[i+2],
                   GateType::And, temp.mWires[j+2]);  // t2=z0*z2
        cd.addGate(temp.mWires[j+2], a.mWires[i+1],
                   GateType::Xor, b.mWires[i+1]);  // x1=t2+z1

        cd.addGate(a.mWires[i], a.mWires[i+1],
                   GateType::And, temp.mWires[j+3]);  // t3=z0*z1
        cd.addGate(temp.mWires[j+3], temp.mWires[j+1],
                   GateType::Xor, temp.mWires[j+4]);  // t4=t3+t1
        cd.addGate(temp.mWires[j+4], a.mWires[i+2],
                   GateType::Xor, b.mWires[i+2]);  // x2=t4+z2

        j += 5;
    }

    cd.levelByAndDepth();
    return cd;
}

BetaCircuit lessthanN(u32 n) {
    BetaCircuit cd;

    BetaBundle a(128);
    cd.addInputBundle(a);

    // 64-bit elements in a 128-bit bitset
    u32 bits = 128 - oc::log2ceil(n);

    u64 step = 1;

    while (step < bits) {
        for (u64 i = 0; i + step < bits; i += step * 2) {
            cd.addGate(a.mWires[i], a.mWires[i + step],
                       oc::GateType::Or, a.mWires[i]);
        }
        step *= 2;
    }

    cd.addInvert(a[0]);
    a.mWires.resize(1);
    cd.mOutputs.push_back(a);

    cd.levelByAndDepth();
    return cd;
}

}  // namespace mpsu
