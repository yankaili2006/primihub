#include "gtest/gtest.h"
#include "src/primihub/operator/aby3_operator.h"
#include "src/primihub/protocol/aby3/encryptor.h"
#include "src/primihub/protocol/aby3/evaluator/evaluator.h"
#include "src/primihub/protocol/aby3/runtime.h"
#include "src/primihub/util/network/socket/commpkg.h"
#include "src/primihub/util/network/socket/ioservice.h"
#include "src/primihub/util/network/socket/session.h"

using namespace primihub;

static std::shared_ptr<CommPkg> setup_comm(u64 partyIdx, Sh3Encryptor &enc,
                                           Sh3Evaluator &eval,
                                           Sh3Runtime &runtime) {
  IOService ios;
  CommPkg comm;
  Session ep_next, ep_prev;
  switch (partyIdx) {
  case 0:
    ep_next.start(ios, "127.0.0.1", 1313, SessionMode::Server, "01");
    ep_prev.start(ios, "127.0.0.1", 1414, SessionMode::Server, "02");
    break;
  case 1:
    ep_next.start(ios, "127.0.0.1", 1515, SessionMode::Server, "12");
    ep_prev.start(ios, "127.0.0.1", 1313, SessionMode::Client, "01");
    break;
  default:
    ep_next.start(ios, "127.0.0.1", 1414, SessionMode::Client, "02");
    ep_prev.start(ios, "127.0.0.1", 1515, SessionMode::Client, "12");
    break;
  }
  comm.setNext(ep_next.addChannel());
  comm.setPrev(ep_prev.addChannel());
  comm.mNext().waitForConnection();
  comm.mPrev().waitForConnection();
  comm.mNext().send(partyIdx);
  comm.mPrev().send(partyIdx);
  u64 prev_party = 0, next_party = 0;
  comm.mNext().recv(next_party);
  comm.mPrev().recv(prev_party);
  EXPECT_EQ(next_party, (partyIdx + 1) % 3);
  EXPECT_EQ(prev_party, (partyIdx + 2) % 3);
  enc.init(partyIdx, comm, sysRandomSeed());
  eval.init(partyIdx, comm, sysRandomSeed());
  auto commPtr = std::make_shared<CommPkg>(comm.mPrev(), comm.mNext());
  runtime.init(partyIdx, commPtr);
  return commPtr;
}

static void test_pow_quo_drelu(u64 partyIdx) {
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  auto commPtr = setup_comm(partyIdx, enc, eval, runtime);

  MPCOperator mpc(partyIdx, "01", "02");
  mpc.setup(commPtr);

  // Test QuoDertermine: returns -1 for negative, 1 for positive
  {
    u64 rows = 3;
    eMatrix<double> input(rows, 1);
    input(0, 0) = 3.0;
    input(1, 0) = -5.0;
    input(2, 0) = 0.0;

    f64Matrix<D16> fixed_input(rows, 1);
    for (u64 i = 0; i < rows; i++)
      fixed_input(i, 0) = input(i, 0);

    sf64Matrix<D16> sh_input(rows, 1);
    if (partyIdx == 0)
      mpc.enc.localFixedMatrix(mpc.runtime, fixed_input, sh_input).get();
    else
      mpc.enc.remoteFixedMatrix(mpc.runtime, sh_input).get();

    sf64Matrix<D16> quo_result = mpc.MPC_QuoDertermine(sh_input);
    eMatrix<double> plain_quo = mpc.revealAll(quo_result);
    LOG(INFO) << "Party " << partyIdx << " QuoDertermine: " << plain_quo;
    // For fixed-point: sign should be -1 or 1
    EXPECT_NEAR(plain_quo(0, 0), 1.0, 0.001);
    EXPECT_NEAR(plain_quo(1, 0), -1.0, 0.001);
    EXPECT_NEAR(plain_quo(2, 0), 1.0, 0.001);
  }

  // Test Pow: compute 2^rank for values > 0.5
  {
    u64 rows = 2;
    eMatrix<double> input(rows, 1);
    input(0, 0) = 0.75;
    input(1, 0) = 2.0;

    f64Matrix<D16> fixed_input(rows, 1);
    for (u64 i = 0; i < rows; i++)
      fixed_input(i, 0) = input(i, 0);

    sf64Matrix<D16> sh_input(rows, 1);
    if (partyIdx == 0)
      mpc.enc.localFixedMatrix(mpc.runtime, fixed_input, sh_input).get();
    else
      mpc.enc.remoteFixedMatrix(mpc.runtime, sh_input).get();

    eMatrix<i64> pow_result = mpc.MPC_Pow(sh_input);
    if (partyIdx == 0) {
      LOG(INFO) << "Party " << partyIdx << " Pow result: " << pow_result;
      // Pow returns the rank value (exponent of 2)
      EXPECT_GT(pow_result(0, 0), 0);
    }
  }

  // Test Pow2: compute 2^(-rank) for values < 0.5
  {
    u64 rows = 2;
    eMatrix<double> input(rows, 1);
    input(0, 0) = 0.25;
    input(1, 0) = 0.1;

    f64Matrix<D16> fixed_input(rows, 1);
    for (u64 i = 0; i < rows; i++)
      fixed_input(i, 0) = input(i, 0);

    sf64Matrix<D16> sh_input(rows, 1);
    if (partyIdx == 0)
      mpc.enc.localFixedMatrix(mpc.runtime, fixed_input, sh_input).get();
    else
      mpc.enc.remoteFixedMatrix(mpc.runtime, sh_input).get();

    eMatrix<i64> pow2_result = mpc.MPC_Pow2(sh_input);
    if (partyIdx == 0) {
      LOG(INFO) << "Party " << partyIdx << " Pow2 result: " << pow2_result;
      // Pow2 returns negative values (since result = -rank)
      EXPECT_LT(pow2_result(0, 0), 0);
    }
  }

  mpc.fini();
}

TEST(pow_quo_drelu_operator, aby3_3pc_test) {
  pid_t pid = fork();
  if (pid != 0) { test_pow_quo_drelu(0); return; }
  pid = fork();
  if (pid != 0) { sleep(1); test_pow_quo_drelu(1); return; }
  sleep(3); test_pow_quo_drelu(2);
}
