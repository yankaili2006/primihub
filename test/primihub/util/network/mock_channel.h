#ifndef TEST_PRIMIHUB_UTIL_NETWORK_MOCK_CHANNEL_H_
#define TEST_PRIMIHUB_UTIL_NETWORK_MOCK_CHANNEL_H_

#include "gmock/gmock.h"
#include "src/primihub/util/network/link_context.h"
#include "src/primihub/protos/worker.pb.h"

namespace primihub::network {

class MockChannel : public IChannel {
 public:
  MockChannel() : IChannel(nullptr) {}
  explicit MockChannel(LinkContext* link_ctx) : IChannel(link_ctx) {}

  MOCK_METHOD2(send_string, retcode(const std::string& key, const std::string& data));
  MOCK_METHOD2(send_view, retcode(const std::string& key, std::string_view sv_data));
  MOCK_METHOD2(send_wrapper_string, bool(const std::string& key, const std::string& data));
  MOCK_METHOD2(send_wrapper_view, bool(const std::string& key, std::string_view sv_data));
  MOCK_METHOD3(sendRecv_string, retcode(const std::string& key, const std::string& send_data, std::string* recv_data));
  MOCK_METHOD3(sendRecv_view, retcode(const std::string& key, std::string_view send_data, std::string* recv_data));
  MOCK_METHOD2(submitTask, retcode(const rpc::PushTaskRequest& request, rpc::PushTaskReply* reply));
  MOCK_METHOD2(executeTask, retcode(const rpc::PushTaskRequest& request, rpc::PushTaskReply* reply));
  MOCK_METHOD2(killTask, retcode(const rpc::KillTaskRequest& request, rpc::KillTaskResponse* reply));
  MOCK_METHOD2(updateTaskStatus, retcode(const rpc::TaskStatus& request, rpc::Empty* reply));
  MOCK_METHOD2(fetchTaskStatus, retcode(const rpc::TaskContext& request, rpc::TaskStatusReply* reply));
  MOCK_METHOD2(StopTask, retcode(const rpc::TaskContext& request, rpc::Empty* reply));
  MOCK_METHOD2(DownloadData, retcode(const rpc::DownloadRequest& request, std::vector<std::string>* data));
  MOCK_METHOD2(NewDataset, retcode(const rpc::NewDatasetRequest& request, rpc::NewDatasetResponse* reply));
  MOCK_METHOD1(forwardRecv, std::string(const std::string& key));
  MOCK_METHOD2(CheckSendCompleteStatus, retcode(const std::string& key, uint64_t expected_complete_num));

  retcode send(const std::string& key, const std::string& data) override {
    return send_string(key, data);
  }
  retcode send(const std::string& key, std::string_view sv_data) override {
    return send_view(key, sv_data);
  }
  bool send_wrapper(const std::string& key, const std::string& data) override {
    return send_wrapper_string(key, data);
  }
  bool send_wrapper(const std::string& key, std::string_view sv_data) override {
    return send_wrapper_view(key, sv_data);
  }
  retcode sendRecv(const std::string& key, const std::string& send_data, std::string* recv_data) override {
    return sendRecv_string(key, send_data, recv_data);
  }
  retcode sendRecv(const std::string& key, std::string_view send_data, std::string* recv_data) override {
    return sendRecv_view(key, send_data, recv_data);
  }
};

class MockLinkContext : public LinkContext {
 public:
  MOCK_METHOD1(getChannel, std::shared_ptr<IChannel>(const primihub::Node& node));
  MOCK_METHOD3(Send_str, retcode(const std::string& key, const Node& node, const std::string& buf));
  MOCK_METHOD3(Send_view, retcode(const std::string& key, const Node& node, std::string_view buf));
  MOCK_METHOD2(Recv_str, retcode(const std::string& key, std::string* buf));
  MOCK_METHOD3(Recv_from, retcode(const std::string& key, const Node& node, std::string* buf));
  MOCK_METHOD4(SendRecv_str, retcode(const std::string& key, const Node& node, const std::string& send_buf, std::string* recv_buf));
  retcode Send(const std::string& key, const Node& dest_node, const std::string& send_buf) override {
    return Send_str(key, dest_node, send_buf);
  }
  retcode Send(const std::string& key, const Node& dest_node, std::string_view send_buf) override {
    return Send_view(key, dest_node, send_buf);
  }
  retcode Recv(const std::string& key, std::string* recv_buf) override {
    return Recv_str(key, recv_buf);
  }
  retcode Recv(const std::string& key, const Node& dest_node, std::string* recv_buf) override {
    return Recv_from(key, dest_node, recv_buf);
  }
  retcode SendRecv(const std::string& key, const Node& dest_node, const std::string& send_buf, std::string* recv_buf) override {
    return SendRecv_str(key, dest_node, send_buf, recv_buf);
  }
};

}  // namespace primihub::network

#endif  // TEST_PRIMIHUB_UTIL_NETWORK_MOCK_CHANNEL_H_
