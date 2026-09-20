//  "Copyright [2023] <PrimiHub>"
#ifndef SRC_PRIMIHUB_KERNEL_PIR_COMMON_H_
#define SRC_PRIMIHUB_KERNEL_PIR_COMMON_H_
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <cstdint>

namespace primihub::pir {
using PirDataType = std::unordered_map<std::string, std::vector<std::string>>;

// Legacy PIR scheme selector preserved for backward compatibility.
// New code SHOULD pass an algorithm name string to PirRegistry::Create instead.
enum class PirType {
  ID_PIR = 0,
  KEY_PIR,
};

enum class QueryType {
  Index = 0,
  Keyword,
  Semantic,
};

enum class ThreatModel {
  SemiHonest = 0,
  SemiHonestNonColluding,
  Malicious,
};

enum class PerfClass {
  Ms = 0,
  SubSecond,
  Seconds,
  Tens,
};

enum class Backend {
  CPU = 0,
  AVX2,
  CUDA,
  AUTO,
};

enum class LatencyBudget {
  Any = 0,
  Ms,
  SubSecond,
  Seconds,
};

inline const char* ToString(PirType t) {
  switch (t) {
    case PirType::ID_PIR: return "ID_PIR";
    case PirType::KEY_PIR: return "KEY_PIR";
  }
  return "UNKNOWN";
}

inline const char* ToString(QueryType q) {
  switch (q) {
    case QueryType::Index: return "Index";
    case QueryType::Keyword: return "Keyword";
    case QueryType::Semantic: return "Semantic";
  }
  return "UNKNOWN";
}

inline const char* ToString(ThreatModel m) {
  switch (m) {
    case ThreatModel::SemiHonest: return "SemiHonest";
    case ThreatModel::SemiHonestNonColluding: return "SemiHonestNonColluding";
    case ThreatModel::Malicious: return "Malicious";
  }
  return "UNKNOWN";
}

inline const char* ToString(PerfClass p) {
  switch (p) {
    case PerfClass::Ms: return "Ms";
    case PerfClass::SubSecond: return "SubSecond";
    case PerfClass::Seconds: return "Seconds";
    case PerfClass::Tens: return "Tens";
  }
  return "UNKNOWN";
}

inline const char* ToString(Backend b) {
  switch (b) {
    case Backend::CPU: return "CPU";
    case Backend::AVX2: return "AVX2";
    case Backend::CUDA: return "CUDA";
    case Backend::AUTO: return "AUTO";
  }
  return "UNKNOWN";
}

inline const char* LegacyNameFor(PirType t) {
  switch (t) {
    case PirType::ID_PIR: return "id_pir";
    case PirType::KEY_PIR: return "apsi";
  }
  return "";
}

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_COMMON_H_
