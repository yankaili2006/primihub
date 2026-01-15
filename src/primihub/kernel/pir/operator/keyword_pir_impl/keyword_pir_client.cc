/*
* Copyright (c) 2023 by PrimiHub
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      https://www.apache.org/licenses/
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "src/primihub/kernel/pir/operator/keyword_pir_impl/keyword_pir_client.h"
#include <fstream>
#include <algorithm>
#include <unordered_map>

#include "src/primihub/common/value_check_util.h"
#include "src/primihub/util/util.h"
#include "src/primihub/util/file_util.h"
#include "src/primihub/common/common.h"
#include "city.h"

namespace primihub::pir {
using Receiver = apsi::receiver::Receiver;
using MatchRecord = apsi::receiver::MatchRecord;
using PSIParams = apsi::PSIParams;

retcode KeywordPirOperatorClient::OnExecute(const PirDataType& input_data,
                                            PirDataType* result) {
  auto link_ctx = this->GetLinkContext();
  DataGroup grouped_data;
  // step 1
  {
    auto ret = AcquireQueryPolicy();
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);
    ret = ValidataQueryPolicy();
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);
    ret = ProcessDataByQueryPolicy(input_data, &grouped_data);
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);
    IndexType loop_to_execute{grouped_data.size()};
    ret = NotifyLoopToExecute(loop_to_execute);
  }

  // step 2
  for (auto& [gourp_index, group_data] : grouped_data) {
    VLOG(5) << "begin to request psi params";
    auto ret = RequestPSIParams(gourp_index);
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);

    std::vector<std::string> orig_item_total;
    std::vector<apsi::Item> items_vec;
    orig_item_total.reserve(group_data.size());
    items_vec.reserve(group_data.size());
    for (const auto& key : group_data) {
      apsi::Item item = key;
      items_vec.emplace_back(std::move(item));
      orig_item_total.emplace_back(key);
    }
    std::vector<HashedItem> oprf_items_total;
    std::vector<LabelKey> label_keys_total;
    VLOG(5) << "begin to Receiver::RequestOPRF";
    ret = RequestOprf(gourp_index,
                      items_vec, &oprf_items_total, &label_keys_total);
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);

    CHECK_TASK_STOPPED(retcode::FAIL);
    VLOG(5) << "Receiver::RequestOPRF end, begin to receiver.request_query";

    this->receiver_ = std::make_unique<Receiver>(*psi_params_);
    size_t query_size = group_data.size();
    auto table_size = this->psi_params_->table_params().table_size;
    table_size = static_cast<size_t>(table_size * PirConstant::table_size_factor);
    size_t block_num = query_size / table_size;
    size_t rem_size = query_size % table_size;
    std::vector<int64_t> block_item_info;
    for (size_t i = 0; i < block_num; i++) {
      block_item_info.push_back(table_size);
    }
    if (rem_size != 0) {
      block_item_info.push_back(rem_size);
    }
    if (VLOG_IS_ON(5)) {
      std::string block_item_info_str;
      for (const auto& item : block_item_info) {
        block_item_info_str.append(std::to_string(item)).append(" ");
      }
      LOG(INFO) << "block_item_info: " << block_item_info_str;
    }

    int64_t start_index{0};
    for (size_t i = 0; i < block_item_info.size(); i++) {
      int64_t size_per_query = block_item_info[i];
      if (i > 0) {
        start_index += block_item_info[i-1];
      }
      LOG(INFO) << "start batch group: " << i << " "
          << "start index: " << start_index << " "
          << "group size: " << size_per_query;
      std::vector<std::string> orig_item;
      std::vector<HashedItem> oprf_items;
      std::vector<LabelKey> label_keys;
      orig_item.reserve(size_per_query);
      oprf_items.reserve(size_per_query);
      label_keys.reserve(size_per_query);
      for (size_t j = 0; j < size_per_query; j++) {
        size_t index = start_index + j;
        orig_item.push_back(orig_item_total[index]);
        oprf_items.push_back(oprf_items_total[index]);
        label_keys.push_back(label_keys_total[index]);
      }
      // request query
      std::vector<MatchRecord> query_result;
      auto query = this->receiver_->create_query(oprf_items);
      // chl.send(move(query.first));
      auto request_query_data = std::move(query.first);
      std::ostringstream string_ss;
      request_query_data->save(string_ss);
      std::string serialized_data = string_ss.str();
      std::string query_data_str;
      query_data_str.reserve(sizeof(IndexType) + serialized_data.size());
      query_data_str.append(TO_CCHAR(&gourp_index), sizeof(gourp_index));
      query_data_str.append(serialized_data);
      auto itt = move(query.second);
      VLOG(5) << "query_data_str size: " << query_data_str.size();

      ret = link_ctx->Send(this->key_, PeerNode(), query_data_str);
      CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);

      // receive package count
      uint32_t package_count = 0;
      std::string pkg_count_key = this->PackageCountKey(link_ctx->request_id());
      ret = link_ctx->Recv(pkg_count_key,
                          this->PeerNode(),
                          reinterpret_cast<char*>(&package_count),
                          sizeof(package_count));
      CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);

      VLOG(5) << "received package count: " << package_count;
      std::vector<apsi::ResultPart> result_packages;
      for (size_t i = 0; i < package_count; i++) {
        std::string recv_data;
        ret = link_ctx->Recv(this->response_key_, this->PeerNode(), &recv_data);
        CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);
        VLOG(5) << "client received data length: " << recv_data.size();
        std::istringstream stream_in(recv_data);
        auto result_part = std::make_unique<apsi::network::ResultPackage>();
        auto seal_context = this->receiver_->get_seal_context();
        result_part->load(stream_in, seal_context);
        result_packages.push_back(std::move(result_part));
      }
      query_result = this->receiver_->process_result(label_keys, itt,
                                                    result_packages);
      VLOG(5) << "query_resultquery_resultquery_resultquery_result: "
              << query_result.size();
      ExtractResult(orig_item, query_result, result);
    }
  }

  {
    std::string task_end{"SUCCESS"};
    auto link_ctx = this->GetLinkContext();
    auto ret = link_ctx->Send(this->key_task_end_, PeerNode(), task_end);
    CHECK_RETCODE_WITH_RETVALUE(ret, retcode::FAIL);
  }
  return retcode::SUCCESS;
}

retcode KeywordPirOperatorClient::AcquireQueryPolicy() {
  CHECK_TASK_STOPPED(retcode::FAIL);
  RequestType type = RequestType::QueryPolicy;
  std::string request{TO_CHAR(&type), sizeof(type)};
  std::string response_str;
  auto link_ctx = this->GetLinkContext();
  CHECK_NULLPOINTER_WITH_ERROR_MSG(link_ctx, "LinkContext is empty");
  auto ret = link_ctx->Send(this->key_, PeerNode(), request);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "send AcquireQueryPolicy to peer: [" << PeerNode().to_string()
        << "] failed";
    return ret;
  }
  ret = link_ctx->Recv(this->response_key_, PeerNode(), &response_str);
  if (ret != retcode::SUCCESS || response_str.empty()) {
    LOG(ERROR) << "acquire query policy from server failed";
    return retcode::FAIL;
  }
  this->slot_size_ = *reinterpret_cast<const IndexType*>(response_str.c_str());
  LOG(INFO) << "query policy for server is: " << this->slot_size_;
  return retcode::SUCCESS;
}
retcode KeywordPirOperatorClient::ValidataQueryPolicy() {
  if (this->slot_size_ > 0) {
    return retcode::SUCCESS;
  } else {
    return retcode::FAIL;
  }
}
retcode KeywordPirOperatorClient::ProcessDataByQueryPolicy(
    const PirDataType& input, DataGroup* grouped_data_ptr) {
  auto& grouped_data = *grouped_data_ptr;
  if (this->slot_size_ == 1) {
    IndexType index{0};
    auto& items = grouped_data[index];
    items.reserve(input.size());
    for (auto& [key, val] : input) {
      items.push_back(key);
    }
  } else {
    for (auto& [key, val] : input) {
      IndexType index = CityHash64(key.c_str(), key.length()) % this->slot_size_;
      auto& items = grouped_data[index];
      items.push_back(key);
    }
  }
  return retcode::SUCCESS;
}

retcode KeywordPirOperatorClient::NotifyLoopToExecute(const IndexType loop_num) {
  CHECK_TASK_STOPPED(retcode::FAIL);
  std::string request{TO_CCHAR(&loop_num), sizeof(loop_num)};
  auto link_ctx = this->GetLinkContext();
  CHECK_NULLPOINTER_WITH_ERROR_MSG(link_ctx, "LinkContext is empty");
  auto ret = link_ctx->Send(this->loop_num_key_, PeerNode(), request);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "send AcquireQueryPolicy to peer: [" << PeerNode().to_string()
        << "] failed";
    return ret;
  }
  return retcode::SUCCESS;
}

// ------------------------Receiver----------------------------
retcode KeywordPirOperatorClient::RequestPSIParams(const IndexType slot_index) {
  CHECK_TASK_STOPPED(retcode::FAIL);
  std::string request{TO_CCHAR(&slot_index), sizeof(slot_index)};
  RequestType type = RequestType::PsiParam;
  request.append(TO_CHAR(&type), sizeof(type));
  VLOG(5) << "send_data length: " << request.length();
  std::string response_str;
  auto link_ctx = this->GetLinkContext();
  CHECK_NULLPOINTER_WITH_ERROR_MSG(link_ctx, "LinkContext is empty");
  auto ret = link_ctx->Send(this->key_, PeerNode(), request);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "send requestPSIParams to peer: [" << PeerNode().to_string()
        << "] failed";
    return ret;
  }
  ret = link_ctx->Recv(this->response_key_, PeerNode(), &response_str);
  if (VLOG_IS_ON(7)) {
    std::string tmp_str;
    for (const auto& chr : response_str) {
      tmp_str.append(std::to_string(static_cast<int>(chr))).append(" ");
    }
    VLOG(7) << "recv_data size: " << response_str.size() << " "
            << "data content: " << tmp_str;
  }

  // create psi params
  // static std::pair<PSIParams, std::size_t> Load(std::istream &in);
  std::istringstream stream_in(response_str);
  auto [parse_data, ret_size] = PSIParams::Load(stream_in);
  psi_params_ = std::make_unique<PSIParams>(parse_data);
  VLOG(5) << "parsed psi param, size: " << ret_size << " "
          << "content: " << psi_params_->to_string();
  return retcode::SUCCESS;
}

retcode KeywordPirOperatorClient::RequestOprf(const IndexType slot_index,
    const std::vector<Item>& items,
    std::vector<apsi::HashedItem>* res_items_ptr,
    std::vector<apsi::LabelKey>* res_label_keys_ptr) {
  CHECK_TASK_STOPPED(retcode::FAIL);

  RequestType type = RequestType::Oprf;
  std::string oprf_response;
  auto oprf_receiver = this->receiver_->CreateOPRFReceiver(items);
  auto& res_items = *res_items_ptr;
  auto& res_label_keys = *res_label_keys_ptr;
  res_items.resize(oprf_receiver.item_count());
  res_label_keys.resize(oprf_receiver.item_count());
  auto oprf_req = oprf_receiver.query_data();
  VLOG(5) << "oprf_request data length: " << oprf_req.size();
  // std::string_view oprf_request_sv{
  //     reinterpret_cast<char*>(const_cast<unsigned char*>(oprf_request.data())),
  //     oprf_request.size()};
  std::string oprf_request_str;
  size_t index_size = sizeof(slot_index);
  oprf_request_str.resize(index_size + oprf_req.size());
  memcpy(&oprf_request_str[0], TO_CCHAR(&slot_index), index_size);
  memcpy(&oprf_request_str[index_size], oprf_req.data(), oprf_req.size());
  auto link_ctx = this->GetLinkContext();
  CHECK_NULLPOINTER_WITH_ERROR_MSG(link_ctx, "LinkContext is empty");
  auto ret = link_ctx->Send(this->key_, PeerNode(), oprf_request_str);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "requestOprf to peer: [" << PeerNode().to_string()
        << "] failed";
    return ret;
  }
  ret = link_ctx->Recv(this->response_key_, this->PeerNode(), &oprf_response);
  if (ret != retcode::SUCCESS || oprf_response.empty()) {
    LOG(ERROR) << "receive oprf_response from peer: ["
               << PeerNode().to_string() << "] failed";
    return retcode::FAIL;
  }
  VLOG(0) << "received oprf response length: " << oprf_response.size() << " ";
  oprf_receiver.process_responses(oprf_response, res_items, res_label_keys);
  return retcode::SUCCESS;
}

retcode KeywordPirOperatorClient::RequestQuery(const IndexType slot_index) {
  RequestType type = RequestType::Query;
  std::string send_data{reinterpret_cast<char*>(&type), sizeof(type)};
  VLOG(5) << "send_data length: " << send_data.length();
  return retcode::SUCCESS;
}

retcode KeywordPirOperatorClient::ExtractResult(
    const std::vector<std::string>& orig_items,
    const std::vector<MatchRecord>& intersection,
    PirDataType* result) {
  CHECK_TASK_STOPPED(retcode::FAIL);
  for (size_t i = 0; i < orig_items.size(); i++) {
    if (!intersection[i].found) {
      VLOG(0) << "no match result found for query: [" << orig_items[i] << "]";
      continue;
    }
    auto& key = orig_items[i];
    if (intersection[i].label.has_data()) {
      std::string label_info = intersection[i].label.to_string();
      std::vector<std::string> labels;
      std::string sep = DATA_RECORD_SEP;
      str_split(label_info, &labels, sep);
      auto pair_info = result->emplace(key, std::vector<std::string>());
      auto& it = std::get<0>(pair_info);
      for (const auto& lable_ : labels) {
        it->second.push_back(lable_);
      }
    } else {
      LOG(WARNING) << "no value found for query key: " << key;
    }
  }
  return retcode::SUCCESS;
}

}  // namespace primihub::pir
