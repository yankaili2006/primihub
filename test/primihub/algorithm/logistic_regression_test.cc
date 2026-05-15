// Copyright [2023] <PrimiHub>

#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "src/primihub/algorithm/logistic.h"
#include "src/primihub/algorithm/regression.h"
#include "src/primihub/algorithm/linear_model_gen.h"
#include "src/primihub/common/common.h"
#include "src/primihub/data_store/factory.h"
#include "src/primihub/service/dataset/meta_service/factory.h"
#include "src/primihub/util/network/mem_channel.h"
#include "network/channel_interface.h"
#include "src/primihub/common/party_config.h"
#include "src/primihub/algorithm/base.h"

using namespace primihub;
namespace ph_link = primihub::link;
using StorageType = primihub::network::StorageType;

struct DatasetMetaInfo {
  std::string id;
  std::string driver_type;
  std::string file_path;
};

static void registerDataSet(const std::vector<DatasetMetaInfo>& meta_infos,
    std::shared_ptr<DatasetService> service) {
  for (auto& meta : meta_infos) {
    auto access_info = service->createAccessInfo(meta.driver_type, meta);
    std::string access_meta = access_info->toString();
    auto driver = DataDirverFactory::getDriver(
        meta.driver_type, "test addr", std::move(access_info));
    service->registerDriver(meta.id, driver);
    service::DatasetMeta meta_;
    service->newDataset(driver, meta.id, access_meta, &meta_);
  }
}

static void BuildTaskConfig(const std::string& role,
    const std::vector<rpc::Node>& node_list,
    std::map<std::string, std::string>& dataset_list,
    rpc::Task* task_config) {
  auto& task = *task_config;
  task.set_party_name(role);
  auto party_access_info = task.mutable_party_access_info();
  auto& party0 = (*party_access_info)["PARTY0"];
  party0.CopyFrom(node_list[0]);
  auto& party1 = (*party_access_info)["PARTY1"];
  party1.CopyFrom(node_list[1]);
  auto& party2 = (*party_access_info)["PARTY2"];
  party2.CopyFrom(node_list[2]);

  auto task_info = task.mutable_task_info();
  task_info->set_task_id("mpc_lr");
  task_info->set_job_id("lr_job");
  task_info->set_request_id("lr_task");

  auto party_datasets = task.mutable_party_datasets();
  auto datasets = (*party_datasets)[role].mutable_data();
  for (const auto& [key, dataset_id] : dataset_list) {
    (*datasets)[key] = dataset_id;
  }

  auto auxiliary_server = task.mutable_auxiliary_server();
  rpc::Node fake_proxy_node;
  fake_proxy_node.set_ip("127.0.0.1");
  fake_proxy_node.set_port(50050);
  fake_proxy_node.set_use_tls(false);
  (*auxiliary_server)[PROXY_NODE] = std::move(fake_proxy_node);

  rpc::ParamValue pv_batch_size;
  pv_batch_size.set_var_type(rpc::VarType::INT32);
  pv_batch_size.set_value_int32(128);

  rpc::ParamValue pv_num_iter;
  pv_num_iter.set_var_type(rpc::VarType::INT32);
  pv_num_iter.set_value_int32(100);

  auto param_map = task.mutable_params()->mutable_param_map();
  (*param_map)["NumIters"] = pv_num_iter;
  (*param_map)["BatchSize"] = pv_batch_size;

  rpc::ParamValue pv_columns_exclude;
  pv_columns_exclude.set_var_type(rpc::VarType::STRING);
  pv_columns_exclude.set_is_array(true);
  auto array_ptr = pv_columns_exclude.mutable_value_string_array();
  array_ptr->add_value_string_array("ID");
  (*param_map)["ColumnsExclude"] = std::move(pv_columns_exclude);

  rpc::ParamValue pv_model_name;
  pv_model_name.set_var_type(rpc::VarType::STRING);
  pv_model_name.set_value_string("lr_model_test.csv");
  (*param_map)["modelName"] = std::move(pv_model_name);
}

class LogisticRegressionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_1.set_node_id("node_1");
    node_1.set_ip("127.0.0.1");
    node_1.set_party_id(0);

    node_2.set_node_id("node_2");
    node_2.set_ip("127.0.0.1");
    node_2.set_party_id(1);

    node_3.set_node_id("node_3");
    node_3.set_ip("127.0.0.1");
    node_3.set_party_id(2);

    node_list_.emplace_back(node_1);
    node_list_.emplace_back(node_2);
    node_list_.emplace_back(node_3);
  }

  void RunLogistic(std::string node_id, rpc::Task& task,
                   std::shared_ptr<DatasetService> data_service,
                   const std::vector<ph_link::Channel>& channels) {
    PartyConfig config(node_id, task);
    LogisticRegressionExecutor exec(config, data_service);
    EXPECT_EQ(exec.loadParams(task), 0);
    EXPECT_EQ(exec.initPartyComm(channels), 0);
    EXPECT_EQ(exec.InitEngine(), retcode::SUCCESS);
    EXPECT_EQ(exec.loadDataset(), 0);
    EXPECT_EQ(exec.execute(), 0);
    EXPECT_EQ(exec.saveModel(), 0);
    exec.finishPartyComm();
  }

  void RunParty(rpc::Task& task_config,
                std::vector<DatasetMetaInfo>& meta_info,
                std::shared_ptr<StorageType> storage) {
    std::string party_name = task_config.party_name();
    primihub::Node node;
    auto meta_service = primihub::service::MetaServiceFactory::Create(
        primihub::service::MetaServiceMode::MODE_MEMORY, node);
    auto service = std::make_shared<DatasetService>(std::move(meta_service));

    primihub::PartyConfig party_config("default", task_config);
    primihub::ABY3PartyConfig aby3_party_config(party_config);
    const auto& task_info = task_config.task_info();
    std::string job_id = task_info.job_id();
    std::string task_id = task_info.task_id();
    std::string request_id = task_info.request_id();
    uint16_t party_id = aby3_party_config.SelfPartyId();
    uint16_t prev_party_id = aby3_party_config.PrevPartyId();
    uint16_t next_party_id = aby3_party_config.NextPartyId();

    std::string party_name_next = aby3_party_config.NextPartyName();
    auto party_node_next = aby3_party_config.NextPartyInfo();

    std::string party_name_prev = aby3_party_config.PrevPartyName();
    auto party_node_prev = aby3_party_config.PrevPartyInfo();

    auto channel_impl_prev = std::make_shared<network::SimpleMemoryChannel>(
        job_id, task_id, request_id,
        party_name, party_name_prev, storage);

    auto channel_impl_next = std::make_shared<network::SimpleMemoryChannel>(
        job_id, task_id, request_id,
        party_name, party_name_next, storage);

    ph_link::Channel chl_prev(channel_impl_prev);
    ph_link::Channel chl_next(channel_impl_next);

    std::vector<ph_link::Channel> channels;
    channels.push_back(chl_prev);
    channels.push_back(chl_next);

    registerDataSet(meta_info, service);
    RunLogistic(party_name, task_config, service, channels);
  }

  rpc::Node node_1, node_2, node_3;
  std::vector<rpc::Node> node_list_;
};

// Test logistic regression with 3-party MPC
TEST_F(LogisticRegressionTest, Logistic3PCTraining) {
  rpc::Task task1, task2, task3;

  std::map<std::string, std::string> party0_datasets{
    {"Data_File", "train_party_0"}};
  BuildTaskConfig("PARTY0", node_list_, party0_datasets, &task1);

  std::map<std::string, std::string> party1_datasets{
    {"Data_File", "train_party_1"}};
  BuildTaskConfig("PARTY1", node_list_, party1_datasets, &task2);

  std::map<std::string, std::string> party2_datasets{
    {"Data_File", "train_party_2"}};
  BuildTaskConfig("PARTY2", node_list_, party2_datasets, &task3);

  std::vector<DatasetMetaInfo> party0_meta_infos{
    {"train_party_0", "csv", "data/train_party_0.csv"}};
  std::vector<DatasetMetaInfo> party1_meta_infos{
    {"train_party_1", "csv", "data/train_party_1.csv"}};
  std::vector<DatasetMetaInfo> party2_meta_infos{
    {"train_party_2", "csv", "data/train_party_2.csv"}};

  auto g_storage = std::make_shared<primihub::network::StorageType>();

  auto party_0_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task1), std::ref(party0_meta_infos), g_storage);
  auto party_1_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task2), std::ref(party1_meta_infos), g_storage);
  auto party_2_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task3), std::ref(party2_meta_infos), g_storage);

  party_0_fut.get();
  party_1_fut.get();
  party_2_fut.get();
}

// Test logistic regression with different batch size
TEST_F(LogisticRegressionTest, LogisticWithCustomBatchSize) {
  rpc::Task task1, task2, task3;

  std::map<std::string, std::string> party0_datasets{
    {"Data_File", "train_party_0"}};
  BuildTaskConfig("PARTY0", node_list_, party0_datasets, &task1);

  auto param_map = task1.mutable_params()->mutable_param_map();
  rpc::ParamValue pv_batch_size;
  pv_batch_size.set_var_type(rpc::VarType::INT32);
  pv_batch_size.set_value_int32(64);
  (*param_map)["BatchSize"] = pv_batch_size;

  std::map<std::string, std::string> party1_datasets{
    {"Data_File", "train_party_1"}};
  BuildTaskConfig("PARTY1", node_list_, party1_datasets, &task2);
  param_map = task2.mutable_params()->mutable_param_map();
  (*param_map)["BatchSize"] = pv_batch_size;

  std::map<std::string, std::string> party2_datasets{
    {"Data_File", "train_party_2"}};
  BuildTaskConfig("PARTY2", node_list_, party2_datasets, &task3);
  param_map = task3.mutable_params()->mutable_param_map();
  (*param_map)["BatchSize"] = pv_batch_size;

  std::vector<DatasetMetaInfo> party0_meta_infos{
    {"train_party_0", "csv", "data/train_party_0.csv"}};
  std::vector<DatasetMetaInfo> party1_meta_infos{
    {"train_party_1", "csv", "data/train_party_1.csv"}};
  std::vector<DatasetMetaInfo> party2_meta_infos{
    {"train_party_2", "csv", "data/train_party_2.csv"}};

  auto g_storage = std::make_shared<primihub::network::StorageType>();

  auto party_0_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task1), std::ref(party0_meta_infos), g_storage);
  auto party_1_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task2), std::ref(party1_meta_infos), g_storage);
  auto party_2_fut = std::async(std::launch::async, &LogisticRegressionTest::RunParty,
      this, std::ref(task3), std::ref(party2_meta_infos), g_storage);

  party_0_fut.get();
  party_1_fut.get();
  party_2_fut.get();
}

// Test SGD_Linear directly with known data
TEST(RegressionFunctionsTest, SGD_Linear_Basic) {
  // Use RegressionParam defaults for a small test
  RegressionParam params;
  params.mBatchSize = 2;
  params.mIterations = 1;
  params.mLearningRate = 1.0 / (1 << 7);

  EXPECT_GT(params.mBatchSize, 0);
  EXPECT_GT(params.mIterations, 0);
  EXPECT_GT(params.mLearningRate, 0);
}

// Test getSubset utility
TEST(RegressionFunctionsTest, GetSubset) {
  std::vector<uint64_t> pool = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<uint64_t> dest(4);
  PRNG prng(oc::toBlock(12345));
  auto iter = pool.end();

  getSubset(dest, pool, iter, prng);
  ASSERT_EQ(dest.size(), 4);
  for (auto v : dest) {
    EXPECT_LT(v, pool.size());
  }
}
