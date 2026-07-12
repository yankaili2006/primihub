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

static void test_dot_mul(u64 partyIdx) {
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  auto commPtr = setup_comm(partyIdx, enc, eval, runtime);

  MPCOperator mpc(partyIdx, "01", "02");
  mpc.setup(commPtr);

  u64 rows = 3, cols = 1;
  eMatrix<i64> plainA(rows, cols), plainB(rows, cols);
  for (u64 i = 0; i < rows; i++) {
    plainA(i, 0) = static_cast<i64>(i + 1);
    plainB(i, 0) = static_cast<i64>(i + 2);
  }

  si64Matrix shA(rows, cols), shB(rows, cols);
  if (partyIdx == 0) {
    mpc.createShares(plainA, shA);
    mpc.createShares(shB);
  } else if (partyIdx == 1) {
    mpc.createShares(shA);
    mpc.createShares(plainB, shB);
  } else {
    mpc.createShares(shA);
    mpc.createShares(shB);
  }

  si64Matrix dot_result = mpc.MPC_Dot_Mul(shA, shB);
  i64Matrix plain_dot = mpc.revealAll(dot_result);
  LOG(INFO) << "Party " << partyIdx << " DOT_MUL result: " << plain_dot;
  for (u64 i = 0; i < rows; i++) {
    i64 expected = plainA(i, 0) * plainB(i, 0);
    EXPECT_EQ(plain_dot(i, 0), expected);
  }
  mpc.fini();
}

TEST(dot_mul_operator, aby3_3pc_test) {
  pid_t pid = fork();
  if (pid != 0) { test_dot_mul(0); return; }
  pid = fork();
  if (pid != 0) { sleep(1); test_dot_mul(1); return; }
  sleep(3); test_dot_mul(2);
}
