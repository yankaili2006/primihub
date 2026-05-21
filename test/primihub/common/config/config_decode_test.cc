#include "gtest/gtest.h"
#include <yaml-cpp/yaml.h>
#include "src/primihub/common/config/config.h"

using primihub::common::RedisConfig;
using primihub::common::CertificateConfig;
using primihub::common::DBInfo;
using primihub::common::Dataset;
using primihub::common::LocalKV;
using primihub::common::P2P;
using primihub::common::ServerInfo;
using primihub::common::NodeConfig;
using primihub::common::Tee;

TEST(ConfigDecodeTest, RedisConfig_Decode) {
  YAML::Node node;
  node["redis_addr"] = "127.0.0.1:6379";
  node["redis_password"] = "pass123";
  node["use_redis"] = true;
  RedisConfig cfg = node.as<RedisConfig>();
  EXPECT_EQ(cfg.redis_addr, "127.0.0.1:6379");
  EXPECT_EQ(cfg.redis_password, "pass123");
  EXPECT_TRUE(cfg.use_redis);
}

TEST(ConfigDecodeTest, RedisConfig_DefaultFalse) {
  YAML::Node node;
  node["redis_addr"] = "";
  node["redis_password"] = "";
  node["use_redis"] = false;
  RedisConfig cfg = node.as<RedisConfig>();
  EXPECT_FALSE(cfg.use_redis);
}

TEST(ConfigDecodeTest, DBInfo_Decode) {
  YAML::Node node;
  node["db_name"] = "test_db";
  node["table_name"] = "test_table";
  DBInfo info = node.as<DBInfo>();
  EXPECT_EQ(info.db_name, "test_db");
  EXPECT_EQ(info.table_name, "test_table");
}

TEST(ConfigDecodeTest, LocalKV_Decode) {
  YAML::Node node;
  node["model"] = "leveldb";
  node["path"] = "/data/kv";
  LocalKV lkv = node.as<LocalKV>();
  EXPECT_EQ(lkv.model, "leveldb");
  EXPECT_EQ(lkv.path, "/data/kv");
}

TEST(ConfigDecodeTest, P2P_Decode) {
  YAML::Node node;
  node["multi_addr"] = "/ip4/0.0.0.0/tcp/4001";
  node["bootstrap_nodes"].push_back("/ip4/1.2.3.4/tcp/4001");
  node["bootstrap_nodes"].push_back("/ip4/5.6.7.8/tcp/4001");
  P2P p2p = node.as<P2P>();
  EXPECT_EQ(p2p.multi_addr, "/ip4/0.0.0.0/tcp/4001");
  ASSERT_EQ(p2p.bootstrap_nodes.size(), 2);
  EXPECT_EQ(p2p.bootstrap_nodes[0], "/ip4/1.2.3.4/tcp/4001");
}

TEST(ConfigDecodeTest, Tee_Decode) {
  YAML::Node node;
  node["executor"] = true;
  node["sgx_enable"] = false;
  node["ra_server_addr"] = "ra.example.com:5000";
  node["cert_path"] = "/certs/tee.pem";
  Tee tee = node.as<Tee>();
  EXPECT_TRUE(tee.executor);
  EXPECT_FALSE(tee.sgx_enable);
  EXPECT_EQ(tee.ra_server_addr, "ra.example.com:5000");
  EXPECT_EQ(tee.cert_path, "/certs/tee.pem");
}

TEST(ConfigDecodeTest, Dataset_Sqlite) {
  YAML::Node node;
  node["description"] = "test dataset";
  node["model"] = "sqlite";
  node["source"] = "test.db";
  node["db_info"]["db_name"] = "main";
  node["db_info"]["table_name"] = "data";
  Dataset ds = node.as<Dataset>();
  EXPECT_EQ(ds.description, "test dataset");
  EXPECT_EQ(ds.model, "sqlite");
  EXPECT_EQ(ds.db_info.db_name, "main");
}

TEST(ConfigDecodeTest, Dataset_NonSqlite_NoDbInfo) {
  YAML::Node node;
  node["description"] = "csv dataset";
  node["model"] = "csv";
  node["source"] = "/data/file.csv";
  Dataset ds = node.as<Dataset>();
  EXPECT_EQ(ds.model, "csv");
  EXPECT_EQ(ds.source, "/data/file.csv");
}

TEST(ConfigDecodeTest, CertificateConfig_EmptyPath) {
  YAML::Node node;
  node["root_ca"] = "";
  node["key"] = "";
  node["cert"] = "";
  // Should decode without crash even though files don't exist
  EXPECT_NO_THROW({
    CertificateConfig cfg = node.as<CertificateConfig>();
  });
}

TEST(ConfigDecodeTest, ServerInfo_Decode) {
  YAML::Node node;
  node["ip"] = "192.168.1.1";
  node["port"] = 8080;
  node["use_tls"] = true;
  node["mode"] = "grpc";
  ServerInfo info = node.as<ServerInfo>();
  EXPECT_EQ(info.host_info.ip_, "192.168.1.1");
  EXPECT_EQ(info.host_info.port_, 8080);
  EXPECT_TRUE(info.host_info.use_tls_);
}

TEST(ConfigDecodeTest, NodeConfig_Basic) {
  YAML::Node node;
  node["node"] = "node0";
  node["location"] = "127.0.0.1";
  node["grpc_port"] = 50050;
  node["use_tls"] = false;
  NodeConfig nc = node.as<NodeConfig>();
  EXPECT_EQ(nc.server_config.id_, "node0");
  EXPECT_EQ(nc.server_config.ip_, "127.0.0.1");
  EXPECT_EQ(nc.server_config.port_, 50050);
  EXPECT_FALSE(nc.server_config.use_tls_);
}

TEST(ConfigDecodeTest, NodeConfig_WithOptionalFields) {
  YAML::Node node;
  node["node"] = "node0";
  node["location"] = "127.0.0.1";
  node["grpc_port"] = 50050;
  node["disable_report"] = true;
  node["storage_path"] = "/data/storage";
  NodeConfig nc = node.as<NodeConfig>();
  EXPECT_TRUE(nc.disable_report);
  EXPECT_EQ(nc.storage_info.path, "/data/storage");
}

TEST(ConfigDecodeTest, Tee_DefaultValues) {
  YAML::Node node;
  node["executor"] = false;
  node["sgx_enable"] = false;
  node["ra_server_addr"] = "";
  node["cert_path"] = "";
  Tee tee = node.as<Tee>();
  EXPECT_FALSE(tee.executor);
  EXPECT_FALSE(tee.sgx_enable);
  EXPECT_TRUE(tee.ra_server_addr.empty());
}

TEST(ConfigDecodeTest, P2P_EmptyBootstrap) {
  YAML::Node node;
  node["multi_addr"] = "/ip4/0.0.0.0/tcp/4001";
  P2P p2p = node.as<P2P>();
  EXPECT_EQ(p2p.multi_addr, "/ip4/0.0.0.0/tcp/4001");
  EXPECT_TRUE(p2p.bootstrap_nodes.empty());
}

TEST(ConfigDecodeTest, ServerInfo_WithTlsTrue) {
  YAML::Node node;
  node["ip"] = "10.0.0.1";
  node["port"] = 50050;
  node["use_tls"] = true;
  node["mode"] = "grpc";
  ServerInfo info = node.as<ServerInfo>();
  EXPECT_TRUE(info.host_info.use_tls_);
}

TEST(ConfigDecodeTest, RedisConfig_EmptyStrings) {
  YAML::Node node;
  node["redis_addr"] = "";
  node["redis_password"] = "";
  node["use_redis"] = false;
  RedisConfig cfg = node.as<RedisConfig>();
  EXPECT_TRUE(cfg.redis_addr.empty());
  EXPECT_TRUE(cfg.redis_password.empty());
}
