#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include "src/primihub/util/file_util.h"

using namespace primihub;

namespace fs = std::filesystem;

class FileUtilTest : public ::testing::Test {
 protected:
  std::string temp_dir_;

  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "file_util_test_XXXXXX";
    char tmpl[1024];
    strncpy(tmpl, temp_dir_.c_str(), sizeof(tmpl) - 1);
    auto dir = mkdtemp(tmpl);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  void CreateFile(const std::string& name, const std::string& content = "") {
    std::ofstream ofs(temp_dir_ + "/" + name);
    ofs << content;
    ofs.close();
  }
};

TEST_F(FileUtilTest, CompletePath_Absolute) {
  auto result = CompletePath("/default", "/absolute/path");
  EXPECT_EQ(result, "/absolute/path");
}

TEST_F(FileUtilTest, CompletePath_Relative) {
  auto result = CompletePath("/default", "relative/path");
  EXPECT_EQ(result, "/default/relative/path");
}

TEST_F(FileUtilTest, CompletePath_EmptyDefault) {
  auto result = CompletePath("", "relative/path");
  EXPECT_EQ(result, "relative/path");
}

TEST_F(FileUtilTest, CompletePath_EmptyPath) {
  auto result = CompletePath("/default", "");
  EXPECT_TRUE(result.empty());
}

TEST_F(FileUtilTest, CompletePath_RelativeNoDefault) {
  auto result = CompletePath("", "path");
  EXPECT_EQ(result, "path");
}

TEST_F(FileUtilTest, FileExists_True) {
  CreateFile("test.txt", "hello");
  EXPECT_TRUE(FileExists(temp_dir_ + "/test.txt"));
}

TEST_F(FileUtilTest, FileExists_False) {
  EXPECT_FALSE(FileExists(temp_dir_ + "/nonexistent.txt"));
}

TEST_F(FileUtilTest, FileSize_Basic) {
  CreateFile("test.txt", "12345");
  auto size = FileSize(temp_dir_ + "/test.txt");
  EXPECT_EQ(size, 5);
}

TEST_F(FileUtilTest, FileSize_Empty) {
  CreateFile("empty.txt");
  auto size = FileSize(temp_dir_ + "/empty.txt");
  EXPECT_EQ(size, 0);
}

TEST_F(FileUtilTest, FileSize_NotExists) {
  auto size = FileSize(temp_dir_ + "/nope.txt");
  EXPECT_EQ(size, 0);
}

TEST_F(FileUtilTest, ValidateDir_Exists) {
  auto ret = ValidateDir(temp_dir_);
  EXPECT_EQ(ret, 0);
}

TEST_F(FileUtilTest, ReadFileContents_Basic) {
  CreateFile("data.txt", "test content");
  std::string content;
  auto ret = ReadFileContents(temp_dir_ + "/data.txt", &content);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(content, "test content");
}

TEST_F(FileUtilTest, ReadFileContents_NotExists) {
  std::string content;
  auto ret = ReadFileContents(temp_dir_ + "/nope.txt", &content);
  EXPECT_EQ(ret, retcode::FAIL);
}
