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

template <Decimal D>
static void test_abs(sf64Matrix<D> &sh_input, MPCOperator &mpc,
                     eMatrix<double> &expected, u64 rows) {
  sf64Matrix<D> abs_result = mpc.MPC_Abs(sh_input);
  eMatrix<double> plain_abs = mpc.revealAll(abs_result);
  LOG(INFO) << "ABS result: " << plain_abs;
  for (u64 i = 0; i < rows; i++)
    EXPECT_NEAR(plain_abs(i, 0), std::abs(expected(i, 0)), 0.001);
}

template <Decimal D>
static void test_drelu(sf64Matrix<D> &sh_input, MPCOperator &mpc,
                       eMatrix<double> &expected, u64 rows) {
  sf64Matrix<D> drelu_result = mpc.MPC_DReLu(sh_input);
  eMatrix<double> plain_drelu = mpc.revealAll(drelu_result);
  LOG(INFO) << "DReLu result: " << plain_drelu;
  for (u64 i = 0; i < rows; i++) {
    double exp_val = expected(i, 0) > 0 ? 1.0 : 0.0;
    EXPECT_NEAR(plain_drelu(i, 0), exp_val, 0.001);
  }
}

static void test_abs_drelu(u64 partyIdx) {
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  auto commPtr = setup_comm(partyIdx, enc, eval, runtime);

  MPCOperator mpc(partyIdx, "01", "02");
  mpc.setup(commPtr);

  u64 rows = 4;
  eMatrix<double> input(rows, 1);
  input(0, 0) = 3.5;
  input(1, 0) = -2.5;
  input(2, 0) = 0.0;
  input(3, 0) = -7.8;

  f64Matrix<D16> fixed_input(rows, 1);
  for (u64 i = 0; i < rows; i++)
    fixed_input(i, 0) = input(i, 0);

  sf64Matrix<D16> sh_input(rows, 1);
  if (partyIdx == 0)
    mpc.enc.localFixedMatrix(mpc.runtime, fixed_input, sh_input).get();
  else
    mpc.enc.remoteFixedMatrix(mpc.runtime, sh_input).get();

  test_abs<D16>(sh_input, mpc, input, rows);
  test_drelu<D16>(sh_input, mpc, input, rows);
  mpc.fini();
}

TEST(abs_operator, aby3_3pc_test) {
  pid_t pid = fork();
  if (pid != 0) { test_abs_drelu(0); return; }
  pid = fork();
  if (pid != 0) { sleep(1); test_abs_drelu(1); return; }
  sleep(3); test_abs_drelu(2);
}
