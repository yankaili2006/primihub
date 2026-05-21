#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/file.h>
#include "src/primihub/data_store/csv/csv_driver.h"
#include "src/primihub/data_store/factory.h"

using namespace primihub;

namespace fs = std::filesystem;

class CsvDriverTest : public ::testing::Test {
 protected:
  std::string temp_dir_;
  std::string csv_file_;

  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "primihub_csv_test_XXXXXX";
    char tmpl[1024];
    strncpy(tmpl, temp_dir_.c_str(), sizeof(tmpl) - 1);
    auto dir = mkdtemp(tmpl);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
    csv_file_ = temp_dir_ + "/test.csv";
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  void CreateCsvFile(const std::string& content) {
    std::ofstream ofs(csv_file_);
    ofs << content;
    ofs.close();
  }
};

TEST_F(CsvDriverTest, ReadCsv_Basic) {
  CreateCsvFile("a,b,c\n1,2,3\n4,5,6\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read();
  ASSERT_NE(dataset, nullptr);
  auto table = std::get<std::shared_ptr<arrow::Table>>(dataset->data);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_rows(), 2);
  ASSERT_EQ(table->num_columns(), 3);
}

TEST_F(CsvDriverTest, ReadCsv_ReadMeta) {
  CreateCsvFile("x,y\n1.0,hello\n2.0,world\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto meta = cursor->readMeta();
  ASSERT_NE(meta, nullptr);
}

TEST_F(CsvDriverTest, ReadCsv_WithOffset) {
  CreateCsvFile("a,b\n1,2\n3,4\n5,6\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read(1, 2);
  ASSERT_NE(dataset, nullptr);
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_rows(), 2);
}

TEST_F(CsvDriverTest, ReadCsv_OffsetBeyondEnd) {
  CreateCsvFile("a,b\n1,2\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  auto dataset = cursor->read(10, 5);
  ASSERT_EQ(dataset, nullptr);
}

TEST_F(CsvDriverTest, ReadCsv_SpecificColumns) {
  CreateCsvFile("a,b,c\n1,2,3\n4,5,6\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  driver->read();
  std::vector<int> col_index = {0, 2};
  auto cursor = driver->GetCursor(col_index);
  ASSERT_NE(cursor, nullptr);
}

TEST_F(CsvDriverTest, CSVDriverType) {
  auto driver = std::make_shared<CSVDriver>("nodelet_addr");
  EXPECT_EQ(driver->getDriverType(), "CSV");
}

TEST_F(CsvDriverTest, CSVDriverNodeletAddr) {
  auto driver = std::make_shared<CSVDriver>("custom_addr");
  EXPECT_EQ(driver->getNodeletAddress(), "custom_addr");
}

TEST_F(CsvDriverTest, CSVAccessInfo_Serialization) {
  CSVAccessInfo info("/path/to/data.csv");
  auto json_str = info.toString();
  EXPECT_FALSE(json_str.empty());
  EXPECT_NE(json_str.find("data_path"), std::string::npos);
  EXPECT_NE(json_str.find("/path/to/data.csv"), std::string::npos);
}

TEST_F(CsvDriverTest, EmptyCsv) {
  CreateCsvFile("a,b\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read();
  ASSERT_NE(dataset, nullptr);
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_EQ(table->num_rows(), 0);
}

TEST_F(CsvDriverTest, SingleRow) {
  CreateCsvFile("h1,h2\nonly,row\n");
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  auto dataset = cursor->read();
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_EQ(table->num_rows(), 1);
}

TEST_F(CsvDriverTest, FactoryCreate_Csv) {
  auto driver = DataDirverFactory::getDriver("CSV", "nodelet_addr");
  ASSERT_NE(driver, nullptr);
  EXPECT_EQ(driver->getDriverType(), "CSV");
}

TEST_F(CsvDriverTest, FactoryCreate_Lowercase) {
  auto driver = DataDirverFactory::getDriver("csv", "nodelet_addr");
  ASSERT_NE(driver, nullptr);
  EXPECT_EQ(driver->getDriverType(), "CSV");
}

TEST_F(CsvDriverTest, FactoryCreate_Invalid) {
  EXPECT_THROW(
    DataDirverFactory::getDriver("INVALID_TYPE", "nodelet_addr"),
    std::runtime_error
  );
}

TEST_F(CsvDriverTest, WriteThenReadSimple) {
  auto access_info = std::make_unique<CSVAccessInfo>(csv_file_);
  auto driver = std::make_shared<CSVDriver>("nodelet_addr", std::move(access_info));
  CreateCsvFile("x,y\n10,1.5\n20,2.5\n30,3.5\n");
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read();
  ASSERT_NE(dataset, nullptr);
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_rows(), 3);
  ASSERT_EQ(table->num_columns(), 2);
}
