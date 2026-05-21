// Copyright [2023] <PrimiHub>

#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include "src/primihub/algorithm/mpc_statistics.h"
#include "src/primihub/common/common.h"
#include "test/primihub/algorithm/statistics_util.h"

using namespace primihub;
using namespace primihub::test;

static constexpr double EPS = 1e-6;

class MPCStatisticsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    BuildPartyNodeInfo(&node_list_);
  }

  void RunStatisticsTest(const std::string& node_id, rpc::Task& task,
                         std::shared_ptr<DatasetService> data_service) {
    PartyConfig config(node_id, task);
    MPCStatisticsExecutor exec(config, data_service);
    EXPECT_EQ(exec.loadParams(task), 0);
    EXPECT_EQ(exec.initPartyComm(), 0);
    EXPECT_EQ(exec.loadDataset(), 0);
    EXPECT_EQ(exec.execute(), 0);
    EXPECT_EQ(exec.saveModel(), 0);
    exec.finishPartyComm();
  }

  void BuildAndRun(const std::string& func_type,
                   const std::vector<std::string>& columns,
                   const std::vector<std::string>& party_datasets,
                   const std::map<std::string, int>& convert_option,
                   const std::string& result_prefix) {
    std::string task_detail = BuildTaskDetail(
        func_type, party_datasets, columns);

    std::map<std::string, std::map<std::string, std::string>> dataset_info;
    int party_id = 0;
    for (const auto& ds : party_datasets) {
      std::string out_file = "data/result/";
      out_file.append(result_prefix)
              .append("_party_").append(std::to_string(party_id)).append(".csv");
      dataset_info[ds] = {
        {"outputFilePath", out_file},
        {"newDataSetId", "new_" + ds}
      };
      party_id++;
    }
    std::string column_info = BuildColumnInfo(dataset_info, convert_option);

    std::map<std::string, std::string> params_info = {
      {"ColumnInfo", column_info},
      {"TaskDetail", task_detail}
    };

    rpc::Task tasks[3];
    std::vector<std::string> party_names = {"PARTY0", "PARTY1", "PARTY2"};
    std::vector<std::map<std::string, std::string>> party_ds = {
      {{"Data_File", party_datasets[0]}},
      {{"Data_File", party_datasets[1]}},
      {{"Data_File", party_datasets[2]}}
    };
    for (int i = 0; i < 3; i++) {
      BuildTaskConfig(party_names[i], node_list_, party_ds[i],
                      params_info, &tasks[i]);
    }

    pid_t pid = fork();
    if (pid != 0) {
      primihub::Node node;
      auto meta_service = primihub::service::MetaServiceFactory::Create(
          primihub::service::MetaServiceMode::MODE_MEMORY, node);
      auto service = std::make_shared<DatasetService>(std::move(meta_service));
      std::vector<DatasetMetaInfo> meta_infos{
        {party_datasets[0], "csv", "data/mpc_test.csv"}};
      registerDataSet(meta_infos, service);
      RunStatisticsTest("node_1", tasks[0], service);
      return;
    }

    pid = fork();
    if (pid != 0) {
      sleep(1);
      primihub::Node node;
      auto meta_service = primihub::service::MetaServiceFactory::Create(
          primihub::service::MetaServiceMode::MODE_MEMORY, node);
      auto service = std::make_shared<DatasetService>(std::move(meta_service));
      std::vector<DatasetMetaInfo> meta_infos{
        {party_datasets[1], "csv", "data/mpc_test.csv"}};
      registerDataSet(meta_infos, service);
      RunStatisticsTest("node_2", tasks[1], service);
      return;
    }

    sleep(3);
    primihub::Node node;
    auto meta_service = primihub::service::MetaServiceFactory::Create(
        primihub::service::MetaServiceMode::MODE_MEMORY, node);
    auto service = std::make_shared<DatasetService>(std::move(meta_service));
    std::vector<DatasetMetaInfo> meta_infos{
      {party_datasets[2], "csv", "data/mpc_test.csv"}};
    registerDataSet(meta_infos, service);
    RunStatisticsTest("node_3", tasks[2], service);
  }

  static void VerifySumResult(const std::vector<double>& result,
                              int expected_count) {
    ASSERT_FALSE(result.empty());
    EXPECT_GT(result.size(), 0);
    for (double v : result) {
      EXPECT_TRUE(std::isfinite(v));
    }
  }

  static void VerifyAvgResult(const std::vector<double>& result,
                              int expected_count) {
    ASSERT_FALSE(result.empty());
    for (double v : result) {
      EXPECT_TRUE(std::isfinite(v));
    }
  }

  std::vector<rpc::Node> node_list_;
};

TEST_F(MPCStatisticsTest, SumOperation) {
  std::vector<std::string> party_datasets{
    "sum_test_data_0", "sum_test_data_1", "sum_test_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("2", columns, party_datasets, convert_option, "mpc_sum");
}

TEST_F(MPCStatisticsTest, AvgOperation) {
  std::vector<std::string> party_datasets{
    "avg_test_data_0", "avg_test_data_1", "avg_test_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("1", columns, party_datasets, convert_option, "mpc_avg");
}

TEST_F(MPCStatisticsTest, MaxOperation) {
  std::vector<std::string> party_datasets{
    "max_test_data_0", "max_test_data_1", "max_test_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("3", columns, party_datasets, convert_option, "mpc_max");
}

TEST_F(MPCStatisticsTest, MinOperation) {
  std::vector<std::string> party_datasets{
    "min_test_data_0", "min_test_data_1", "min_test_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("4", columns, party_datasets, convert_option, "mpc_min");
}

TEST_F(MPCStatisticsTest, TTestOperation) {
  std::vector<std::string> party_datasets{
    "ttest_data_0", "ttest_data_1", "ttest_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("5", columns, party_datasets, convert_option, "mpc_ttest");
}

TEST_F(MPCStatisticsTest, FTestOperation) {
  std::vector<std::string> party_datasets{
    "ftest_data_0", "ftest_data_1", "ftest_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("6", columns, party_datasets, convert_option, "mpc_ftest");
}

TEST_F(MPCStatisticsTest, ChiSquareTestOperation) {
  std::vector<std::string> party_datasets{
    "chi2_data_0", "chi2_data_1", "chi2_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("7", columns, party_datasets, convert_option, "mpc_chi2");
}

TEST_F(MPCStatisticsTest, RegressionOperation) {
  std::vector<std::string> party_datasets{
    "reg_data_0", "reg_data_1", "reg_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("8", columns, party_datasets, convert_option, "mpc_regression");
}

TEST_F(MPCStatisticsTest, CorrelationOperation) {
  std::vector<std::string> party_datasets{
    "corr_data_0", "corr_data_1", "corr_data_2"
  };
  std::vector<std::string> columns{"x_1", "x_2"};
  std::map<std::string, int> convert_option{
    {"x_1", 2}, {"x_2", 2}
  };
  BuildAndRun("9", columns, party_datasets, convert_option, "mpc_correlation");
}

TEST_F(MPCStatisticsTest, ParseParamsInvalidJson) {
  std::vector<std::string> party_datasets{
    "invalid_data_0", "invalid_data_1", "invalid_data_2"
  };
  std::vector<std::string> columns{"x_1"};
  std::map<std::string, int> convert_option{{"x_1", 2}};
  std::map<std::string, std::map<std::string, std::string>> dataset_info;
  dataset_info[party_datasets[0]] = {
    {"outputFilePath", "data/result/bad.csv"},
    {"newDataSetId", "new_bad"}
  };
  std::string column_info = BuildColumnInfo(dataset_info, convert_option);
  std::string bad_json = R"({"bad": "data"})";
  std::map<std::string, std::string> params_info = {
    {"ColumnInfo", column_info},
    {"TaskDetail", bad_json}
  };

  rpc::Task task;
  std::map<std::string, std::string> ds{{"Data_File", party_datasets[0]}};
  BuildTaskConfig("PARTY0", node_list_, ds, params_info, &task);

  primihub::Node node;
  auto meta_service = primihub::service::MetaServiceFactory::Create(
      primihub::service::MetaServiceMode::MODE_MEMORY, node);
  auto service = std::make_shared<DatasetService>(std::move(meta_service));
  PartyConfig config("node_1", task);
  MPCStatisticsExecutor exec(config, service);
  EXPECT_NE(exec.loadParams(task), 0);
}

TEST_F(MPCStatisticsTest, EmptyColumnSelection) {
  std::vector<std::string> party_datasets{
    "empty_data_0", "empty_data_1", "empty_data_2"
  };
  std::vector<std::string> empty_columns{};
  std::string task_detail = BuildTaskDetail("2", party_datasets, empty_columns);
  std::map<std::string, std::string> params_info = {
    {"ColumnInfo", "{}"},
    {"TaskDetail", task_detail}
  };

  rpc::Task task;
  std::map<std::string, std::string> ds{{"Data_File", party_datasets[0]}};
  BuildTaskConfig("PARTY0", node_list_, ds, params_info, &task);

  primihub::Node node;
  auto meta_service = primihub::service::MetaServiceFactory::Create(
      primihub::service::MetaServiceMode::MODE_MEMORY, node);
  auto service = std::make_shared<DatasetService>(std::move(meta_service));
  PartyConfig config("node_1", task);
  MPCStatisticsExecutor exec(config, service);
  EXPECT_NE(exec.loadParams(task), 0);
}

TEST(MPCStatisticsTypeTest, TypeMapping) {
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::AVG), 0);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::SUM), 1);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::STAT_MAX), 2);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::STAT_MIN), 3);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::T_TEST), 4);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::STAT_F_TEST), 5);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::CHI_SQUARE_TEST), 6);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::REGRESSION), 7);
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::CORRELATION), 8);
}

TEST(MPCStatisticsTypeTest, UnknownType) {
  EXPECT_EQ(static_cast<int>(MPCStatisticsType::UNKNOWN), -1);
}
