#include "src/primihub/kernel/pir/operator/id_pir.h"
#include <glog/logging.h>
#include <sstream>
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

constexpr const char* kIdPirKey = "pir_id_query";

retcode IdPirOperator::OnExecute(const PirDataType& input, PirDataType* result) {
  if (role() == Role::CLIENT) {
    return ClientSendRecv(input, result);
  } else if (role() == Role::SERVER) {
    return ServerSendRecv(input, result);
  }
  LOG(ERROR) << "unknown role: " << PartyName();
  return retcode::FAIL;
}

retcode IdPirOperator::ClientSendRecv(const PirDataType& input, PirDataType* result) {
  std::string query_str;
  for (const auto& [key, _] : input) {
    if (!query_str.empty()) query_str.append(",");
    query_str.append(key);
  }
  VLOG(5) << "CLIENT sending query: " << query_str;
  std::string reply_str;
  auto ret = GetLinkContext()->SendRecv(kIdPirKey, PeerNode(), query_str, &reply_str);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "CLIENT SendRecv failed";
    return retcode::FAIL;
  }
  std::istringstream stream(reply_str);
  std::string line;
  while (std::getline(stream, line, '|')) {
    if (line.empty()) continue;
    auto pos = line.find(',');
    if (pos == std::string::npos) continue;
    std::string key = line.substr(0, pos);
    std::string val = line.substr(pos + 1);
    (*result)[key].push_back(val);
  }
  VLOG(5) << "CLIENT received " << result->size() << " results";
  return retcode::SUCCESS;
}

retcode IdPirOperator::ServerSendRecv(const PirDataType& input, PirDataType* result) {
  std::string query_str;
  auto ret = GetLinkContext()->Recv(kIdPirKey, PeerNode(), &query_str);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "SERVER Recv failed";
    return retcode::FAIL;
  }
  VLOG(5) << "SERVER received query: " << query_str;
  std::string reply_str;
  std::istringstream stream(query_str);
  std::string key;
  while (std::getline(stream, key, ',')) {
    if (key.empty()) continue;
    auto it = input.find(key);
    if (it != input.end()) {
      for (const auto& val : it->second) {
        if (!reply_str.empty()) reply_str.append("|");
        reply_str.append(key).append(",").append(val);
      }
    }
  }
  LOG(INFO) << "SERVER sending reply (" << reply_str.size() << " bytes)";
  ret = GetLinkContext()->Send(kIdPirKey, PeerNode(), reply_str);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "SERVER Send failed";
    return retcode::FAIL;
  }
  return retcode::SUCCESS;
}

namespace {
// IdPirOperator is a plaintext ID-lookup passthrough; it provides no FHE-based
// PIR privacy. Registered under "id_pir" so legacy PirType::ID_PIR routes here
// via the factory shim. Real SealPIR will register separately under "seal_pir"
// when the pir-acc/ module is migrated.
PirCapabilities IdPirCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = false;
  caps.hint_per_database = false;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 1000000;  // plain lookup; trivial scale
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 0;  // proportional to query size
  caps.typical_hint_size_bytes = 0;
  caps.is_real = true;  // real SealPIR / passthrough impl
  return caps;
}

PirRegistrar<IdPirOperator> id_pir_registrar_("id_pir", IdPirCaps());
}  // namespace

}  // namespace primihub::pir
