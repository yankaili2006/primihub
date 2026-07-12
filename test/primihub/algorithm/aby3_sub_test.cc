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

static void test_sub(u64 partyIdx) {
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  auto commPtr = setup_comm(partyIdx, enc, eval, runtime);

  MPCOperator mpc(partyIdx, "01", "02");
  mpc.setup(commPtr);

  u64 rows = 2, cols = 2;
  eMatrix<i64> input1(rows, cols), input2(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++) {
      input1(i, j) = static_cast<i64>(i + j + 10);
      input2(i, j) = static_cast<i64>(i + j + 1);
    }

  si64Matrix sh1(rows, cols), sh2(rows, cols);
  mpc.createShares(input1, sh1);
  mpc.createShares(input2, sh2);

  // Test MPC_Sub
  std::vector<si64Matrix> subtrahends = {sh2};
  si64Matrix diff = mpc.MPC_Sub(sh1, subtrahends);
  i64Matrix plain_diff = mpc.revealAll(diff);
  LOG(INFO) << "Party " << partyIdx << " SUB result: " << plain_diff;
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++)
      EXPECT_EQ(plain_diff(i, j), input1(i, j) - input2(i, j));

  // Test MPC_Sub_Const
  i64 const_val = 5;
  si64Matrix sub_const = mpc.MPC_Sub_Const(const_val, sh1, true);
  i64Matrix plain_sub_const = mpc.revealAll(sub_const);
  LOG(INFO) << "Party " << partyIdx << " SUB_CONST result: " << plain_sub_const;
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++)
      EXPECT_EQ(plain_sub_const(i, j), input1(i, j) - const_val);

  mpc.fini();
}

TEST(sub_operator, aby3_3pc_test) {
  pid_t pid = fork();
  if (pid != 0) { test_sub(0); return; }
  pid = fork();
  if (pid != 0) { sleep(1); test_sub(1); return; }
  sleep(3); test_sub(2);
}
