#include "gtest/gtest.h"
#include "src/primihub/util/util.h"

using namespace primihub;

TEST(UtilFuncTest, StrSplit_CharDelimiter) {
  std::vector<std::string> result;
  str_split("a:b:c", &result, ':');
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
  EXPECT_EQ(result[2], "c");
}

TEST(UtilFuncTest, StrSplit_CharDelimiter_NoDelimiter) {
  std::vector<std::string> result;
  str_split("hello", &result, ':');
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "hello");
}

TEST(UtilFuncTest, StrSplit_CharDelimiter_EmptyInput) {
  std::vector<std::string> result;
  str_split("", &result, ':');
  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0].empty());
}

TEST(UtilFuncTest, StrSplit_StringDelimiter) {
  std::vector<std::string> result;
  str_split("a::b::c", &result, "::");
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[2], "c");
}

TEST(UtilFuncTest, StrSplit_StringDelimiter_NoMatch) {
  std::vector<std::string> result;
  str_split("hello", &result, "::");
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "hello");
}

TEST(UtilFuncTest, StrToUpper_Basic) {
  EXPECT_EQ(strToUpper("hello"), "HELLO");
  EXPECT_EQ(strToUpper("Hello World"), "HELLO WORLD");
  EXPECT_EQ(strToUpper("123"), "123");
  EXPECT_EQ(strToUpper(""), "");
}

TEST(UtilFuncTest, StrToLower_Basic) {
  EXPECT_EQ(strToLower("HELLO"), "hello");
  EXPECT_EQ(strToLower("Hello World"), "hello world");
  EXPECT_EQ(strToLower("123"), "123");
  EXPECT_EQ(strToLower(""), "");
}

TEST(UtilFuncTest, TrimLeft) {
  std::string s1 = "  hello";
  TrimLeft(s1);
  EXPECT_EQ(s1, "hello");

  std::string s2 = "hello";
  TrimLeft(s2);
  EXPECT_EQ(s2, "hello");

  std::string s3 = "";
  TrimLeft(s3);
  EXPECT_EQ(s3, "");
}

TEST(UtilFuncTest, TrimRight) {
  std::string s1 = "hello  ";
  TrimRight(s1);
  EXPECT_EQ(s1, "hello");

  std::string s2 = "hello";
  TrimRight(s2);
  EXPECT_EQ(s2, "hello");

  std::string s3 = "";
  TrimRight(s3);
  EXPECT_EQ(s3, "");
}

TEST(UtilFuncTest, TrimAll) {
  std::string s1 = "  hello  ";
  TrimAll(s1);
  EXPECT_EQ(s1, "hello");

  std::string s2 = "hello";
  TrimAll(s2);
  EXPECT_EQ(s2, "hello");

  std::string s3 = "   ";
  TrimAll(s3);
  EXPECT_TRUE(s3.empty());
}

TEST(UtilFuncTest, ScopedTimer) {
  SCopedTimer timer;
  usleep(1000);
  auto elapsed = timer.timeElapse<std::chrono::milliseconds>();
  EXPECT_GE(elapsed, 0);
  auto elapsed_us = timer.timeElapse<std::chrono::microseconds>();
  EXPECT_GE(elapsed_us, 1000);
}

TEST(UtilFuncTest, BufToHexString) {
  uint8_t data[] = {0xde, 0xad, 0xbe, 0xef};
  auto hex = buf_to_hex_string(data, 4);
  EXPECT_EQ(hex, "deadbeef");

  uint8_t empty[] = {};
  auto empty_hex = buf_to_hex_string(empty, 0);
  EXPECT_TRUE(empty_hex.empty());
}

TEST(UtilFuncTest, SortPeers) {
  std::vector<std::string> peers = {"node3", "node1", "node2"};
  sort_peers(&peers);
  ASSERT_EQ(peers.size(), 3);
  EXPECT_EQ(peers[0], "node1");
  EXPECT_EQ(peers[1], "node2");
  EXPECT_EQ(peers[2], "node3");
}

TEST(UtilFuncTest, SortPeers_Empty) {
  std::vector<std::string> peers;
  sort_peers(&peers);
  EXPECT_TRUE(peers.empty());
}

TEST(UtilFuncTest, ParseToNode) {
  Node node;
  auto ret = parseToNode("node0:127.0.0.1:50050:0", &node);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(node.id_, "node0");
  EXPECT_EQ(node.ip_, "127.0.0.1");
  EXPECT_EQ(node.port_, 50050);
  EXPECT_FALSE(node.use_tls_);
}

TEST(UtilFuncTest, ParseToNode_TooFewFields) {
  Node node;
  auto ret = parseToNode("invalid", &node);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST(UtilFuncTest, GetAvailablePort) {
  uint32_t port = 0;
  int ret = getAvailablePort(&port);
  EXPECT_EQ(ret, 0);
  EXPECT_GT(port, 0);
  EXPECT_LE(port, 65535);
}

TEST(UtilFuncTest, StrToUpper_Mixed) {
  EXPECT_EQ(strToUpper("Hello123!@#"), "HELLO123!@#");
}

TEST(UtilFuncTest, StrToLower_Mixed) {
  EXPECT_EQ(strToLower("HELLO123!@#"), "hello123!@#");
}

TEST(UtilFuncTest, StrSplit_ConsecutiveDelimiters) {
  std::vector<std::string> result;
  str_split("a::b", &result, ':');
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_TRUE(result[1].empty());
  EXPECT_EQ(result[2], "b");
}

TEST(UtilFuncTest, StrSplit_StringDelimiter_PartialAtEnd) {
  std::vector<std::string> result;
  str_split("a::b::", &result, "::");
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "b");
}

TEST(UtilFuncTest, BufToHexString_AllZeros) {
  uint8_t data[] = {0x00, 0x00, 0x00};
  auto hex = buf_to_hex_string(data, 3);
  EXPECT_EQ(hex, "000000");
}

TEST(UtilFuncTest, BufToHexString_MaxValues) {
  uint8_t data[] = {0xFF, 0xAB, 0xCD};
  auto hex = buf_to_hex_string(data, 3);
  EXPECT_EQ(hex, "ffabcd");
}
