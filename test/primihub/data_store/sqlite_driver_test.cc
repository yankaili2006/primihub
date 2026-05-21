#include "gtest/gtest.h"
#include <filesystem>
#include <memory>
#include <SQLiteCpp/SQLiteCpp.h>
#include "src/primihub/data_store/sqlite/sqlite_driver.h"
#include "src/primihub/data_store/factory.h"

using namespace primihub;

namespace fs = std::filesystem;

class SqliteDriverTest : public ::testing::Test {
 protected:
  std::string temp_dir_;
  std::string db_path_;

  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "primihub_sqlite_test_XXXXXX";
    char tmpl[1024];
    strncpy(tmpl, temp_dir_.c_str(), sizeof(tmpl) - 1);
    auto dir = mkdtemp(tmpl);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
    db_path_ = temp_dir_ + "/test.db";
    CreateTestDb();
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  void CreateTestDb() {
    SQLite::Database db(db_path_, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    db.exec("CREATE TABLE test_table (id INTEGER, name TEXT, score DOUBLE)");
    db.exec("INSERT INTO test_table VALUES (1, 'alice', 95.5)");
    db.exec("INSERT INTO test_table VALUES (2, 'bob', 87.0)");
    db.exec("INSERT INTO test_table VALUES (3, 'carol', 92.3)");
  }
};

TEST_F(SqliteDriverTest, ReadSqlite_Basic) {
  auto access_info = std::make_unique<SQLiteAccessInfo>(db_path_, "test_table",
                                                         std::vector<std::string>());
  auto driver = std::make_shared<SQLiteDriver>("nodelet_addr", std::move(access_info));
  auto cursor = driver->read();
  ASSERT_NE(cursor, nullptr);
  auto dataset = cursor->read();
  ASSERT_NE(dataset, nullptr);
  auto& data_ref = dataset->data;
  auto table = std::get<std::shared_ptr<arrow::Table>>(data_ref);
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->num_rows(), 3);
}

TEST_F(SqliteDriverTest, SqliteDriverType) {
  auto driver = std::make_shared<SQLiteDriver>("nodelet_addr");
  EXPECT_EQ(driver->getDriverType(), "SQLITE");
}

TEST_F(SqliteDriverTest, FactoryCreate_Sqlite) {
  auto driver = DataDirverFactory::getDriver("SQLITE", "nodelet_addr");
  ASSERT_NE(driver, nullptr);
  EXPECT_EQ(driver->getDriverType(), "SQLITE");
}

TEST_F(SqliteDriverTest, SqliteAccessInfo_Serialization) {
  SQLiteAccessInfo info("/path/to/db.db", "table1", {"col1", "col2"});
  auto json_str = info.toString();
  EXPECT_FALSE(json_str.empty());
}
