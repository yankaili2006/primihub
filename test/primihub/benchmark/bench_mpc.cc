#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "gtest/gtest.h"
#include "src/primihub/operator/aby3_operator.h"
#include "src/primihub/protocol/aby3/encryptor.h"
#include "src/primihub/protocol/aby3/evaluator/evaluator.h"
#include "src/primihub/protocol/aby3/runtime.h"
#include "src/primihub/util/network/socket/commpkg.h"
#include "src/primihub/util/network/socket/ioservice.h"
#include "src/primihub/util/network/socket/session.h"

using namespace primihub;
using namespace std::chrono;
using DCache = std::vector<double>;

struct BenchResult {
  std::string name;
  std::string dtype;
  uint64_t rows;
  uint64_t cols;
  uint64_t data_size;
  double avg_ms;
  double min_ms;
  double max_ms;
  double stddev_ms;
  std::string category;
};

struct BenchSummary {
  std::string category;
  int count;
  double total_ms;
  double avg_throughput;
};

static std::vector<BenchResult> results;
static const std::string CSV_FILE = "benchmark_results.csv";
static constexpr int ITERATIONS = 7;
static constexpr int WARMUP = 2;

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

static DCache collect_times(auto &&f, int warmup, int iterations) {
  for (int i = 0; i < warmup; i++) f();
  DCache times;
  times.reserve(iterations);
  for (int i = 0; i < iterations; i++) {
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    times.push_back(duration<double, std::milli>(end - start).count());
  }
  return times;
}

static BenchResult analyze(const std::string &name, const std::string &dtype,
                           uint64_t rows, uint64_t cols,
                           const DCache &times, const std::string &cat) {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double avg = sum / times.size();
  double min_v = *std::min_element(times.begin(), times.end());
  double max_v = *std::max_element(times.begin(), times.end());
  double sq = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
  double stddev = std::sqrt(sq / times.size() - avg * avg);
  return {name, dtype, rows, cols, rows * cols, avg, min_v, max_v, stddev, cat};
}

static BenchResult bench(const std::string &name, const std::string &dtype,
                         uint64_t rows, uint64_t cols, auto &&f,
                         int warmup = WARMUP, int iter = ITERATIONS,
                         const std::string &cat = "local") {
  auto times = collect_times(f, warmup, iter);
  auto r = analyze(name, dtype, rows, cols, times, cat);
  results.push_back(r);
  return r;
}

// ─── si64 benchmarks ───────────────────────────────────────────
static si64Matrix make_shares(MPCOperator &mpc, const eMatrix<i64> &plain) {
  si64Matrix sh(plain.rows(), plain.cols());
  mpc.createShares(plain, sh);
  return sh;
}

static void bench_add(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p1(rows, cols), p2(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++) {
      p1(i, j) = static_cast<i64>(i + j);
      p2(i, j) = static_cast<i64>(i - j);
    }
  auto s1 = make_shares(mpc, p1);
  auto s2 = make_shares(mpc, p2);
  std::vector<si64Matrix> vec = {s1, s2};
  bench("ADD", "si64", rows, cols, [&]() { (void)mpc.MPC_Add(vec); },
        1, 5, "local_nocomm");
}

static void bench_add_const(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++)
      p(i, j) = static_cast<i64>(i + j);
  auto s = make_shares(mpc, p);
  i64 c = 10;
  bench("ADD_CONST", "si64", rows, cols, [&]() { (void)mpc.MPC_Add_Const(c, s); },
        1, 5, "local_nocomm");
}

static void bench_sub(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p1(rows, cols), p2(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++) {
      p1(i, j) = static_cast<i64>(i + j + 10);
      p2(i, j) = static_cast<i64>(i + j + 1);
    }
  auto s1 = make_shares(mpc, p1);
  auto s2 = make_shares(mpc, p2);
  std::vector<si64Matrix> vec = {s2};
  bench("SUB", "si64", rows, cols, [&]() { (void)mpc.MPC_Sub(s1, vec); },
        1, 5, "local_nocomm");
}

static void bench_sub_const(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++)
      p(i, j) = static_cast<i64>(i + j + 10);
  auto s = make_shares(mpc, p);
  i64 c = 3;
  bench("SUB_CONST", "si64", rows, cols, [&]() { (void)mpc.MPC_Sub_Const(c, s, true); },
        1, 5, "local_nocomm");
}

static void bench_mul(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p1(rows, cols), p2(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++) {
      p1(i, j) = static_cast<i64>(i + j + 1);
      p2(i, j) = static_cast<i64>(i + j + 2);
    }
  auto s1 = make_shares(mpc, p1);
  auto s2 = make_shares(mpc, p2);
  std::vector<si64Matrix> vec = {s1, s2};
  bench("MUL", "si64", rows, cols, [&]() { (void)mpc.MPC_Mul(vec); },
        1, 5, "interactive");
}

static void bench_mul_const(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++)
      p(i, j) = static_cast<i64>(i + j + 1);
  auto s = make_shares(mpc, p);
  i64 c = 3;
  bench("MUL_CONST", "si64", rows, cols, [&]() { (void)mpc.MPC_Mul_Const(c, s); },
        1, 5, "local_nocomm");
}

static void bench_dot_mul(MPCOperator &mpc, uint64_t rows, uint64_t cols) {
  eMatrix<i64> p1(rows, cols), p2(rows, cols);
  for (u64 i = 0; i < rows; i++)
    for (u64 j = 0; j < cols; j++) {
      p1(i, j) = static_cast<i64>(i + j + 1);
      p2(i, j) = static_cast<i64>(i + j + 2);
    }
  auto s1 = make_shares(mpc, p1);
  auto s2 = make_shares(mpc, p2);
  bench("DOT_MUL", "si64", rows, cols, [&]() { (void)mpc.MPC_Dot_Mul(s1, s2); },
        1, 5, "interactive");
}

static void bench_compare(MPCOperator &mpc, uint64_t rows) {
  i64Matrix m1(rows, 1), m2(rows, 1);
  for (u64 i = 0; i < rows; i++) {
    m1(i, 0) = static_cast<i64>(i * 2);
    m2(i, 0) = static_cast<i64>(i * 3);
  }
  bench("CMP", "si64", rows, 1, [&]() { sbMatrix r; mpc.MPC_Compare(m1, r); },
        0, 3, "interactive_circuit");
}

// ─── sf64 benchmarks ───────────────────────────────────────────
static sf64Matrix<D16> make_fp_shares(MPCOperator &mpc,
                                       const eMatrix<double> &plain) {
  sf64Matrix<D16> sh(plain.rows(), plain.cols());
  if (mpc.partyIdx == 0) {
    f64Matrix<D16> f(plain.rows(), plain.cols());
    for (i64 i = 0; i < plain.size(); i++) f(i) = plain(i);
    mpc.enc.localFixedMatrix(mpc.runtime, f, sh).get();
  } else {
    mpc.enc.remoteFixedMatrix(mpc.runtime, sh).get();
  }
  return sh;
}

static void bench_abs(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> p(rows, 1);
  for (u64 i = 0; i < rows; i++)
    p(i, 0) = (i % 2 == 0) ? (i + 1.0) : -(i + 1.0);
  auto s = make_fp_shares(mpc, p);
  bench("ABS", "sf64", rows, 1, [&]() { (void)mpc.MPC_Abs(s); },
        1, 5, "interactive_piecewise");
}

static void bench_drelu(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> p(rows, 1);
  for (u64 i = 0; i < rows; i++)
    p(i, 0) = (i % 2 == 0) ? 1.0 : -1.0;
  auto s = make_fp_shares(mpc, p);
  bench("DReLu", "sf64", rows, 1, [&]() { (void)mpc.MPC_DReLu(s); },
        1, 5, "interactive_piecewise");
}

static void bench_div(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> pa(rows, 1), pb(rows, 1);
  for (u64 i = 0; i < rows; i++) {
    pa(i, 0) = i + 10.0;
    pb(i, 0) = i + 2.0;
  }
  auto sa = make_fp_shares(mpc, pa);
  sf64Matrix<D16> sb(rows, 1);
  if (mpc.partyIdx == 1) {
    f64Matrix<D16> fb(rows, 1);
    for (u64 i = 0; i < rows; i++) fb(i, 0) = pb(i, 0);
    mpc.enc.localFixedMatrix(mpc.runtime, fb, sb).get();
  } else {
    mpc.enc.remoteFixedMatrix(mpc.runtime, sb).get();
  }
  bench("DIV", "sf64", rows, 1, [&]() { (void)mpc.MPC_Div(sa, sb); },
        0, 3, "interactive_heavy");
}

static void bench_quo(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> p(rows, 1);
  for (u64 i = 0; i < rows; i++)
    p(i, 0) = (i % 2 == 0) ? (i + 2.0) : -(i + 1.0);
  auto s = make_fp_shares(mpc, p);
  bench("Quo", "sf64", rows, 1, [&]() { (void)mpc.MPC_QuoDertermine(s); },
        1, 5, "interactive_piecewise");
}

static void bench_pow(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> p(rows, 1);
  for (u64 i = 0; i < rows; i++)
    p(i, 0) = 0.5 + (i % 5) * 0.1;
  auto s = make_fp_shares(mpc, p);
  bench("Pow", "sf64", rows, 1, [&]() { (void)mpc.MPC_Pow(s); },
        0, 3, "interactive_heavy");
}

static void bench_pow2(MPCOperator &mpc, uint64_t rows) {
  eMatrix<double> p(rows, 1);
  for (u64 i = 0; i < rows; i++)
    p(i, 0) = 0.1 + (i % 5) * 0.05;
  auto s = make_fp_shares(mpc, p);
  bench("Pow2", "sf64", rows, 1, [&]() { (void)mpc.MPC_Pow2(s); },
        0, 3, "interactive_heavy");
}

// ─── 2D matrix benchmark ──────────────────────────────────────
static void bench_2d_matrix(MPCOperator &mpc) {
  auto r = bench("ADD_2D", "si64", 32, 32, [&]() {
    u64 rows = 32, cols = 32;
    eMatrix<i64> p1(rows, cols), p2(rows, cols);
    for (u64 i = 0; i < rows; i++)
      for (u64 j = 0; j < cols; j++) {
        p1(i, j) = static_cast<i64>(i + j);
        p2(i, j) = static_cast<i64>(i - j);
      }
    auto s1 = make_shares(mpc, p1);
    auto s2 = make_shares(mpc, p2);
    (void)mpc.MPC_Add({s1, s2});
  }, 1, 5, "local_nocomm");
}

// ─── Reporting ─────────────────────────────────────────────────
static void print_results() {
  std::cout << "\n" << std::string(100, '=') << std::endl;
  std::cout << "  PrimiHub MPC Operator Performance Benchmark" << std::endl;
  std::cout << std::string(100, '=') << std::endl;
  std::cout << "  Iterations: " << ITERATIONS << "  Warmup: " << WARMUP << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  std::cout << std::left
    << std::setw(10) << "Operator"
    << std::setw(8) << "Type"
    << std::setw(6) << "Rows"
    << std::setw(6) << "Cols"
    << std::setw(10) << "Elements"
    << std::setw(10) << "Avg(ms)"
    << std::setw(10) << "StdDev"
    << std::setw(10) << "Min(ms)"
    << std::setw(10) << "Max(ms)"
    << std::setw(14) << "Throughput"
    << std::setw(22) << "Category"
    << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  for (auto &r : results) {
    double thr = r.data_size / (r.avg_ms / 1000.0);
    std::string unit = "ops/s";
    if (thr > 1e6) { thr /= 1e6; unit = "Mops/s"; }
    else if (thr > 1e3) { thr /= 1e3; unit = "Kops/s"; }

    std::cout << std::left
      << std::setw(10) << r.name
      << std::setw(8) << r.dtype
      << std::setw(6) << r.rows
      << std::setw(6) << r.cols
      << std::setw(10) << r.data_size
      << std::setw(10) << std::fixed << std::setprecision(4) << r.avg_ms
      << std::setw(10) << std::fixed << std::setprecision(4) << r.stddev_ms
      << std::setw(10) << std::fixed << std::setprecision(4) << r.min_ms
      << std::setw(10) << std::fixed << std::setprecision(4) << r.max_ms
      << std::setw(14) << std::fixed << std::setprecision(3) << thr << " " << unit
      << r.category
      << std::endl;
  }
  std::cout << std::string(100, '-') << std::endl;

  // Summary by category
  std::map<std::string, BenchSummary> cats;
  for (auto &r : results) {
    auto &c = cats[r.category];
    c.category = r.category;
    c.count++;
    c.total_ms += r.avg_ms;
    double thr = r.data_size / (r.avg_ms / 1000.0);
    c.avg_throughput += thr;
  }
  std::cout << "\n" << std::string(60, '-') << std::endl;
  std::cout << "  Performance Summary by Category" << std::endl;
  std::cout << std::string(60, '-') << std::endl;
  std::cout << std::left
    << std::setw(24) << "Category"
    << std::setw(8) << "Count"
    << std::setw(14) << "Total(ms)"
    << "Avg Throughput" << std::endl;
  for (auto &[key, c] : cats) {
    c.avg_throughput /= c.count;
    std::string unit = "ops/s";
    double disp = c.avg_throughput;
    if (disp > 1e6) { disp /= 1e6; unit = "Mops/s"; }
    else if (disp > 1e3) { disp /= 1e3; unit = "Kops/s"; }
    std::cout << std::left
      << std::setw(24) << c.category
      << std::setw(8) << c.count
      << std::setw(14) << std::fixed << std::setprecision(2) << c.total_ms
      << std::fixed << std::setprecision(3) << disp << " " << unit
      << std::endl;
  }
  std::cout << std::string(60, '-') << std::endl;

  // CSV export
  std::ofstream csv(CSV_FILE);
  csv << "operator,type,rows,cols,elements,avg_ms,stddev_ms,min_ms,max_ms,throughput_ops_s,category\n";
  for (auto &r : results) {
    double thr = r.data_size / (r.avg_ms / 1000.0);
    csv << r.name << "," << r.dtype << ","
        << r.rows << "," << r.cols << "," << r.data_size << ","
        << r.avg_ms << "," << r.stddev_ms << ","
        << r.min_ms << "," << r.max_ms << ","
        << thr << "," << r.category << "\n";
  }
  csv.close();
  std::cout << "\n[Benchmark] Results exported to " << CSV_FILE << std::endl;
}

// ─── Runner ────────────────────────────────────────────────────
static void run_benchmarks(u64 partyIdx) {
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  auto commPtr = setup_comm(partyIdx, enc, eval, runtime);
  MPCOperator mpc(partyIdx, "01", "02");
  mpc.setup(commPtr);

  if (partyIdx != 0) {
    mpc.fini();
    return;
  }

  std::vector<uint64_t> sizes = {64, 256, 1024};
  std::cout << "Party 0 (Leader) running benchmarks...\n";

  // 1D vector benchmarks (all operators)
  for (auto n : sizes) {
    bench_add(mpc, n, 1);
    bench_add_const(mpc, n, 1);
    bench_sub(mpc, n, 1);
    bench_sub_const(mpc, n, 1);
    bench_mul(mpc, n, 1);
    bench_mul_const(mpc, n, 1);
    bench_dot_mul(mpc, n, 1);
    bench_compare(mpc, n);
    bench_abs(mpc, n);
    bench_drelu(mpc, n);
    bench_quo(mpc, n);
  }
  for (auto n : {64, 256}) {
    bench_div(mpc, n);
    bench_pow(mpc, n);
    bench_pow2(mpc, n);
  }

  // 2D matrix benchmark
  bench_2d_matrix(mpc);

  print_results();
  mpc.fini();
}

TEST(mpc_benchmark, performance_test) {
  pid_t pid = fork();
  if (pid != 0) { run_benchmarks(0); return; }
  pid = fork();
  if (pid != 0) { sleep(1); run_benchmarks(1); return; }
  sleep(3); run_benchmarks(2);
}
