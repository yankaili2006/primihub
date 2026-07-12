#include "gtest/gtest.h"
#include <tuple>
#include <vector>
#include <memory>
#include "src/primihub/algorithm/mpc_statistics.h"
#include "test/primihub/algorithm/statistics_util.h"

using namespace primihub;
using namespace primihub::test;

static void RunStatisticsTTest(std::string node_id, rpc::Task &task,
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

TEST(statistics_ttest, statistics_ttest_test) {
  std::vector<rpc::Node> node_list;
  BuildPartyNodeInfo(&node_list);
  std::vector<std::string> party_datasets{
    "avg_test_data_0", "avg_test_data_1", "avg_test_data_2"
  };
  std::vector<std::string> checked_columns{"x_1", "x_2"};
  std::string function_name = "5";
  std::string task_detail = BuildTaskDetail(function_name, party_datasets, checked_columns);
  LOG(INFO) << task_detail;
  std::map<std::string, int> convert_option;
  for (const auto& colum_name : checked_columns) {
    convert_option[colum_name] = 2;
  }
  std::map<std::string, std::map<std::string, std::string>> dataset_info;
  int party_id = 0;
  for (const auto& dataset_id : party_datasets) {
    std::string out_file = "data/result/";
    out_file.append("mpc_ttest_party_").append(std::to_string(party_id)).append(".csv");
    std::string new_dataset_id = "new_" + dataset_id;
    dataset_info[dataset_id] = {
      {"outputFilePath", out_file},
      {"newDataSetId", new_dataset_id}
    };
    party_id++;
  }
  std::string column_info = BuildColumnInfo(dataset_info, convert_option);
  LOG(INFO) << column_info;
  std::map<std::string, std::string> params_info = {
    {"ColumnInfo", column_info},
    {"TaskDetail", task_detail}
  };
  rpc::Task task1;
  std::map<std::string, std::string> party0_datasets{
    {"Data_File", party_datasets[0]}};
  BuildTaskConfig("PARTY0", node_list, party0_datasets, params_info, &task1);
  rpc::Task task2;
  std::map<std::string, std::string> party1_datasets{
    {"Data_File", party_datasets[1]}};
  BuildTaskConfig("PARTY1", node_list, party1_datasets, params_info, &task2);
  rpc::Task task3;
  std::map<std::string, std::string> party2_datasets{
    {"Data_File", party_datasets[2]}};
  BuildTaskConfig("PARTY2", node_list, party2_datasets, params_info, &task3);

  pid_t pid = fork();
  if (pid != 0) {
    primihub::Node node;
    auto meta_service = primihub::service::MetaServiceFactory::Create(
        primihub::service::MetaServiceMode::MODE_MEMORY, node);
    auto service = std::make_shared<DatasetService>(std::move(meta_service));
    std::vector<DatasetMetaInfo> meta_infos {
      {party_datasets[0], "csv", "data/mpc_test.csv"},
    };
    registerDataSet(meta_infos, service);
    RunStatisticsTTest("node_1", task1, service);
    return;
  }
  pid = fork();
  if (pid != 0) {
    sleep(1);
    primihub::Node node;
    auto meta_service = primihub::service::MetaServiceFactory::Create(
        primihub::service::MetaServiceMode::MODE_MEMORY, node);
    auto service = std::make_shared<DatasetService>(std::move(meta_service));
    std::vector<DatasetMetaInfo> meta_infos {
      {party_datasets[1], "csv", "data/mpc_test.csv"},
    };
    registerDataSet(meta_infos, service);
    RunStatisticsTTest("node_2", task2, service);
    return;
  }
  sleep(3);
  primihub::Node node;
  auto meta_service = primihub::service::MetaServiceFactory::Create(
      primihub::service::MetaServiceMode::MODE_MEMORY, node);
  auto service = std::make_shared<DatasetService>(std::move(meta_service));
  std::vector<DatasetMetaInfo> meta_infos {
    {party_datasets[2], "csv", "data/mpc_test.csv"},
  };
  registerDataSet(meta_infos, service);
  RunStatisticsTTest("node_3", task3, service);
}
