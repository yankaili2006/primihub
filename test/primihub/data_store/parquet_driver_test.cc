#include "gtest/gtest.h"
#include <filesystem>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include "src/primihub/data_store/parquet/parquet_driver.h"
#include "src/primihub/data_store/factory.h"

using namespace primihub;

namespace fs = std::filesystem;

class ParquetDriverTest : public ::testing::Test {
 protected:
  std::string temp_dir_;
  std::string parquet_file_;

  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "primihub_parquet_test_XXXXXX";
    char tmpl[1024];
    strncpy(tmpl, temp_dir_.c_str(), sizeof(tmpl) - 1);
    auto dir = mkdtemp(tmpl);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
    parquet_file_ = temp_dir_ + "/test.parquet";
    CreateTestParquetFile();
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  std::shared_ptr<arrow::Array> MakeInt32Array(const std::vector<int32_t>& vals) {
    arrow::Int32Builder builder;
    for (auto v : vals) { builder.Append(v).ok(); }
    return builder.Finish().ValueOrDie();
  }

  std::shared_ptr<arrow::Array> MakeDoubleArray(const std::vector<double>& vals) {
    arrow::DoubleBuilder builder;
    for (auto v : vals) { builder.Append(v).ok(); }
    return builder.Finish().ValueOrDie();
  }

  std::shared_ptr<arrow::Array> MakeStringArray(const std::vector<std::string>& vals) {
    arrow::StringBuilder builder;
    for (auto& v : vals) { builder.Append(v).ok(); }
    return builder.Finish().ValueOrDie();
  }

  void CreateTestParquetFile() {
    auto schema = arrow::schema({
      arrow::field("a", arrow::int32()),
      arrow::field("b", arrow::float64()),
      arrow::field("c", arrow::utf8()),
    });
    auto a = MakeInt32Array({1, 2, 3});
    auto b = MakeDoubleArray({1.1, 2.2, 3.3});
    auto c = MakeStringArray({"x", "y", "z"});
    std::vector<std::shared_ptr<arrow::Array>> cols;
    cols.push_back(a);
    cols.push_back(b);
    cols.push_back(c);
    auto table = arrow::Table::Make(schema, cols, 3);
    auto outfile = arrow::io::FileOutputStream::Open(parquet_file_).ValueOrDie();
    auto wresult = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                               outfile, 65536);
    ASSERT_TRUE(wresult.ok());
  }
};

TEST_F(ParquetDriverTest, ReadParquet_Basic) {
  auto access_info = std::make_unique<ParquetAccessInfo>(parquet_file_);
  auto driver = std::make_shared<ParquetDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->readMeta();
  ASSERT_NE(dataset, nullptr);
}

TEST_F(ParquetDriverTest, ReadParquet_WithColumns) {
  auto access_info = std::make_unique<ParquetAccessInfo>(parquet_file_);
  auto driver = std::make_shared<ParquetDriver>("nodelet_addr", std::move(access_info));
  driver->read();
  std::vector<int> cols = {0, 1, 2};
  auto cursor = driver->GetCursor(cols);
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read();
  ASSERT_NE(dataset, nullptr);
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_rows(), 3);
  ASSERT_EQ(table->num_columns(), 3);
}

TEST_F(ParquetDriverTest, ReadParquet_SpecificColumns) {
  auto access_info = std::make_unique<ParquetAccessInfo>(parquet_file_);
  auto driver = std::make_shared<ParquetDriver>("nodelet_addr", std::move(access_info));
  driver->read();
  std::vector<int> col_index = {0, 2};
  auto cursor = driver->GetCursor(col_index);
  ASSERT_NE(cursor, nullptr);
}

TEST_F(ParquetDriverTest, ParquetDriverType) {
  auto driver = std::make_shared<ParquetDriver>("nodelet_addr");
  EXPECT_EQ(driver->getDriverType(), "PARQUET");
}

TEST_F(ParquetDriverTest, ParquetAccessInfo_Serialization) {
  ParquetAccessInfo info("/path/to/data.parquet");
  auto json_str = info.toString();
  EXPECT_FALSE(json_str.empty());
  EXPECT_NE(json_str.find("data_path"), std::string::npos);
}

TEST_F(ParquetDriverTest, FactoryCreate_Parquet) {
  auto driver = DataDirverFactory::getDriver("PARQUET", "nodelet_addr");
  ASSERT_NE(driver, nullptr);
  EXPECT_EQ(driver->getDriverType(), "PARQUET");
}
