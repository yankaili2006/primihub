#include "gtest/gtest.h"
#include <nlohmann/json.hpp>
#include "src/primihub/cli/task_config_parser.h"
#include "src/primihub/protos/worker.pb.h"

using namespace primihub;

class TaskConfigParserTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TaskConfigParserTest, FillParamByScalar_String) {
  rpc::ParamValue pv;
  nlohmann::json j = "hello";
  fillParamByScalar("STRING", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_EQ(pv.value_string(), "hello");
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Int32) {
  rpc::ParamValue pv;
  nlohmann::json j = 42;
  fillParamByScalar("INT32", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_EQ(pv.value_int32(), 42);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Int64) {
  rpc::ParamValue pv;
  nlohmann::json j = 1234567890123LL;
  fillParamByScalar("INT64", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_EQ(pv.value_int64(), 1234567890123LL);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Bool) {
  rpc::ParamValue pv;
  nlohmann::json j = true;
  fillParamByScalar("BOOL", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_TRUE(pv.value_bool());
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Float) {
  rpc::ParamValue pv;
  nlohmann::json j = 3.14f;
  fillParamByScalar("FLOAT", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_FLOAT_EQ(pv.value_float(), 3.14f);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Double) {
  rpc::ParamValue pv;
  nlohmann::json j = 2.718281828;
  fillParamByScalar("DOUBLE", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_DOUBLE_EQ(pv.value_double(), 2.718281828);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Object) {
  rpc::ParamValue pv;
  nlohmann::json j = {{"key", "value"}, {"num", 1}};
  fillParamByScalar("OBJECT", j, &pv);
  EXPECT_FALSE(pv.is_array());
  EXPECT_FALSE(pv.value_string().empty());
  auto parsed = nlohmann::json::parse(pv.value_string());
  EXPECT_EQ(parsed["key"], "value");
}

TEST_F(TaskConfigParserTest, FillParamByArray_String) {
  rpc::ParamValue pv;
  nlohmann::json j = {"a", "b", "c"};
  fillParamByArray("STRING", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_string_array().value_string_array_size(), 3);
  EXPECT_EQ(pv.value_string_array().value_string_array(0), "a");
  EXPECT_EQ(pv.value_string_array().value_string_array(2), "c");
}

TEST_F(TaskConfigParserTest, FillParamByArray_Int32) {
  rpc::ParamValue pv;
  nlohmann::json j = {1, 2, 3};
  fillParamByArray("INT32", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_int32_array().value_int32_array_size(), 3);
  EXPECT_EQ(pv.value_int32_array().value_int32_array(1), 2);
}

TEST_F(TaskConfigParserTest, FillParamByArray_Bool) {
  rpc::ParamValue pv;
  nlohmann::json j = {true, false, true};
  fillParamByArray("BOOL", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_bool_array().value_bool_array_size(), 3);
  EXPECT_TRUE(pv.value_bool_array().value_bool_array(0));
  EXPECT_FALSE(pv.value_bool_array().value_bool_array(1));
}

TEST_F(TaskConfigParserTest, FillParamByArray_Double) {
  rpc::ParamValue pv;
  nlohmann::json j = {1.1, 2.2, 3.3};
  fillParamByArray("DOUBLE", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_double_array().value_double_array_size(), 3);
  EXPECT_DOUBLE_EQ(pv.value_double_array().value_double_array(0), 1.1);
}

TEST_F(TaskConfigParserTest, FillParamByArray_Empty) {
  rpc::ParamValue pv;
  nlohmann::json j = nlohmann::json::array();
  fillParamByArray("STRING", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_string_array().value_string_array_size(), 0);
}

TEST_F(TaskConfigParserTest, FillParamByArray_Object) {
  rpc::ParamValue pv;
  nlohmann::json j = {{{"x", 1}}, {{"y", 2}}};
  fillParamByArray("OBJECT", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_string_array().value_string_array_size(), 2);
}

TEST_F(TaskConfigParserTest, ParseDownFileConfig_Empty) {
  nlohmann::json j = {{"task_name", "test"}};
  DownloadFileListType files;
  auto ret = ParseDownFileConfig(j, &files);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_TRUE(files.empty());
}

TEST_F(TaskConfigParserTest, ParseDownFileConfig_Single) {
  nlohmann::json j = {
    {"download", {
      {{"remote_file_path", "/remote/data.csv"}, {"save_as", "./local/data.csv"}}
    }}
  };
  DownloadFileListType files;
  auto ret = ParseDownFileConfig(j, &files);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(files.size(), 1);
  EXPECT_EQ(files[0].remote_file_path, "/remote/data.csv");
  EXPECT_EQ(files[0].save_as, "./local/data.csv");
}

TEST_F(TaskConfigParserTest, ParseDownFileConfig_Multiple) {
  nlohmann::json j = {
    {"download", {
      {{"remote_file_path", "/a"}, {"save_as", "./a"}},
      {{"remote_file_path", "/b"}, {"save_as", "./b"}},
    }}
  };
  DownloadFileListType files;
  auto ret = ParseDownFileConfig(j, &files);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(files.size(), 2);
  EXPECT_EQ(files[1].remote_file_path, "/b");
}

TEST_F(TaskConfigParserTest, BuildFederatedRequest_NoComponentParams) {
  nlohmann::json j = {{"task_name", "test"}};
  rpc::Task task;
  auto ret = BuildFederatedRequest(j, &task);
  EXPECT_EQ(ret, retcode::SUCCESS);
}

TEST_F(TaskConfigParserTest, BuildFederatedRequest_WithRoleParams) {
  nlohmann::json j = {
    {"component_params", {
      {"roles", {{"provider", {"party0"}}, {"consumer", {"party1"}}}},
      {"role_params", {
        {"party0", {{"data_set", "ds0"}}},
        {"party1", {{"data_set", "ds1"}}}
      }}
    }},
    {"party_info", {
      {"party0", {{"ip", "127.0.0.1"}, {"port", 50050}, {"use_tls", false}}},
      {"party1", {{"ip", "127.0.0.1"}, {"port", 50051}, {"use_tls", false}}}
    }}
  };
  rpc::Task task;
  auto ret = BuildFederatedRequest(j, &task);
  EXPECT_EQ(ret, retcode::SUCCESS);
  auto& datasets = task.party_datasets();
  EXPECT_TRUE(datasets.contains("party0"));
  EXPECT_TRUE(datasets.contains("party1"));
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_BasicFields) {
  nlohmann::json j = {
    {"task_type", "ACTOR_TASK"},
    {"task_name", "test_task"},
    {"task_lang", "proto"},
    {"params", {
      {"epoch", {{"type", "INT32"}, {"value", 10}}},
      {"lr", {{"type", "DOUBLE"}, {"value", 0.01}}},
      {"names", {{"type", "STRING"}, {"value", "alice"}}}
    }},
    {"party_datasets", {
      {"party0", {{"data_set", "ds_0"}}}
    }},
    {"party_info", {
      {"party0", {{"ip", "127.0.0.1"}, {"port", 50050}, {"use_tls", false}}}
    }}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  auto& task = request.task();
  EXPECT_EQ(task.name(), "test_task");
  EXPECT_EQ(task.type(), rpc::TaskType::ACTOR_TASK);
  EXPECT_EQ(task.language(), rpc::Language::PROTO);
  auto& params = task.params().param_map();
  EXPECT_EQ(params.at("epoch").value_int32(), 10);
  EXPECT_DOUBLE_EQ(params.at("lr").value_double(), 0.01);
  EXPECT_EQ(params.at("names").value_string(), "alice");
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_DefaultTaskName) {
  nlohmann::json j = {{"task_lang", "proto"}};
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(request.task().name(), "demoTask");
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_InvalidTaskType) {
  nlohmann::json j = {
    {"task_type", "INVALID_TYPE"},
    {"task_lang", "proto"}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_InvalidLang) {
  nlohmann::json j = {
    {"task_lang", "invalid_lang"}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Int32_EdgeValues) {
  rpc::ParamValue pv;
  nlohmann::json j = INT32_MAX;
  fillParamByScalar("INT32", j, &pv);
  EXPECT_EQ(pv.value_int32(), INT32_MAX);

  rpc::ParamValue pv2;
  nlohmann::json j2 = INT32_MIN;
  fillParamByScalar("INT32", j2, &pv2);
  EXPECT_EQ(pv2.value_int32(), INT32_MIN);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Int64_EdgeValues) {
  rpc::ParamValue pv;
  nlohmann::json j = INT64_MAX;
  fillParamByScalar("INT64", j, &pv);
  EXPECT_EQ(pv.value_int64(), INT64_MAX);

  rpc::ParamValue pv2;
  nlohmann::json j2 = INT64_MIN;
  fillParamByScalar("INT64", j2, &pv2);
  EXPECT_EQ(pv2.value_int64(), INT64_MIN);
}

TEST_F(TaskConfigParserTest, FillParamByScalar_Bool_False) {
  rpc::ParamValue pv;
  nlohmann::json j = false;
  fillParamByScalar("BOOL", j, &pv);
  EXPECT_FALSE(pv.value_bool());
}

TEST_F(TaskConfigParserTest, FillParamByScalar_EmptyString) {
  rpc::ParamValue pv;
  nlohmann::json j = "";
  fillParamByScalar("STRING", j, &pv);
  EXPECT_TRUE(pv.value_string().empty());
}

TEST_F(TaskConfigParserTest, FillParamByArray_Int64) {
  rpc::ParamValue pv;
  nlohmann::json j = {1000000000000LL, 2000000000000LL};
  fillParamByArray("INT64", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_int64_array().value_int64_array_size(), 2);
  EXPECT_EQ(pv.value_int64_array().value_int64_array(1), 2000000000000LL);
}

TEST_F(TaskConfigParserTest, FillParamByArray_Float) {
  rpc::ParamValue pv;
  nlohmann::json j = {1.5f, 2.5f, 3.5f};
  fillParamByArray("FLOAT", j, &pv);
  EXPECT_TRUE(pv.is_array());
  EXPECT_EQ(pv.value_float_array().value_float_array_size(), 3);
  EXPECT_FLOAT_EQ(pv.value_float_array().value_float_array(0), 1.5f);
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_PsiTaskType) {
  nlohmann::json j = {
    {"task_type", "PSI_TASK"},
    {"task_lang", "proto"},
    {"party_info", {
      {"party0", {{"ip", "127.0.0.1"}, {"port", 50050}, {"use_tls", false}}}
    }}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(request.task().type(), rpc::TaskType::PSI_TASK);
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_EmptyJson) {
  nlohmann::json j = nlohmann::json::object();
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(request.task().name(), "demoTask");
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_PythonLang) {
  nlohmann::json j = {
    {"task_lang", "python"},
    {"task_code", {{"code_file_path", ""}, {"code", "print('hello')"}}}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(request.task().language(), rpc::Language::PYTHON);
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_CodeFromFile) {
  nlohmann::json j = {
    {"task_lang", "python"},
    {"task_code", {{"code_file_path", "/nonexistent/file.py"}}}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST_F(TaskConfigParserTest, BuildRequestWithTaskConfig_PirTaskType) {
  nlohmann::json j = {
    {"task_type", "PIR_TASK"},
    {"task_lang", "proto"},
    {"party_info", {
      {"party0", {{"ip", "127.0.0.1"}, {"port", 50050}, {"use_tls", false}}}
    }}
  };
  PushTaskRequest request;
  auto ret = BuildRequestWithTaskConfig(j, &request);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(request.task().type(), rpc::TaskType::PIR_TASK);
}
