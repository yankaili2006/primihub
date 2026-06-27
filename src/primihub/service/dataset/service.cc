/*
 Copyright 2022 PrimiHub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include <glog/logging.h>

#include <future>
#include <thread>
#include <chrono>
#include <vector>
#include <utility>

#include "src/primihub/service/dataset/service.h"
#include "src/primihub/data_store/factory.h"
#include "src/primihub/common/common.h"
#include "src/primihub/common/config/config.h"
#include "src/primihub/service/dataset/util.hpp"
#include "src/primihub/util/redis_helper.h"
#include "src/primihub/common/config/server_config.h"
#include "src/primihub/service/dataset/meta_service/grpc_impl.h"

using DataSetAccessInfoPtr = std::unique_ptr<primihub::DataSetAccessInfo>;

namespace primihub::service {

DatasetService::DatasetService(
    std::unique_ptr<DatasetMetaService> meta_service) :
    meta_service_(std::move(meta_service)) {
  Init();
}

retcode DatasetService::Init() {
  auto& ins = ServerConfig::getInstance();
  auto& node_cfg = ins.getNodeConfig();

  nodelet_addr_ = node_cfg.server_config.to_string();
  return retcode::SUCCESS;
}

/**
 * @brief Construct a new Dataset object
 * 1. Read data using driver for get dataset & datameta
 * 2. Save datameta in local storage.
 * 3. Publish dataset meta on libp2p network.
 *
 * @param driver [input]: Data driver
 * @param name [input]: Dataset name
 * @param dataset_id [input]: Dataset description
 * @param dataset_access_info [input]: dataset access_info
 * @param meta [output]: Dataset meta
 * @return std::shared_ptr<primihub::Dataset>
 */
std::shared_ptr<primihub::Dataset>
DatasetService::newDataset(std::shared_ptr<primihub::DataDriver> driver,
                          const std::string& dataset_id,
                          const std::string& dataset_access_info,
                          DatasetMeta* meta) {
  // Read data using driver for get dataset & datameta
  // just get meta info from dataset
  // auto dataset = driver->getCursor()->read();
  auto cursor = driver->read();
  if (cursor == nullptr) {
    LOG(ERROR) << "get cursor for dataset: " << dataset_id << " failed";
    return nullptr;
  }
  auto dataset = cursor->readMeta();
  if (dataset == nullptr) {
    LOG(ERROR) << "get dataset meta for dataset id: "
               << dataset_id << " failed";
    return nullptr;
  }
  *meta = DatasetMeta(dataset,
                      dataset_id,
                      DatasetVisbility::PUBLIC,
                      dataset_access_info);
  // Save datameta in local storage.& Publish dataset meta on libp2p network.
  auto ret = MetaService()->PutMeta(*meta);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "Put Meta data to meta service failed";
    return nullptr;
  }
  return dataset;
}

/**
 * @brief write dataset to local storage
 * @param dataset [input]: Dataset to be written with own driver
 * @param description [input]: Dataset description
 * @param dataset_access_info [input]: Dataset access info
 * @param meta [output]: Dataset metadata
 */
void DatasetService::writeDataset(const std::shared_ptr<primihub::Dataset> &dataset,
              const std::string &description,
              const std::string& dataset_access_info,
              DatasetMeta &meta /*output*/) {
    // dataset.write();
    // dataset->getDataDriver()->getCursor()->write(dataset);     // TODO(fix in future)
    meta = DatasetMeta(dataset, description, DatasetVisbility::PUBLIC, dataset_access_info);
    MetaService()->PutMeta(meta);
}


/**
 * @brief Register dataset use meta
 *
 * @param meta [input]: Dataset meta
 */
void DatasetService::regDataset(const DatasetMeta& meta) {
    MetaService()->PutMeta(meta);
}

/**
 * @brief Read dataset by dataset uri
 * TODO(chenhongbo) Only support local dataset now. !!!
 * @param id [input]: Dataset id
 * @param handler [input]: Read dataset handler
 * @return outcome::result<void>
 *  TODO async style or callback style ?
 */
retcode DatasetService::readDataset(const DatasetId& id,
                                    ReadDatasetHandler handler) {
  MetaService()->GetMeta(id, [&](std::shared_ptr<DatasetMeta> meta) {
    if (meta) {
      // Construct dataset from meta
      auto driver = DataDirverFactory::getDriver(meta->getDriverType(), nodelet_addr_);
      auto cursor = driver->read(meta->getDataURL());  // only support Local file path now.
      auto dataset = cursor->read();
      handler(dataset);
      return retcode::SUCCESS;
    }
    return retcode::FAIL;
  });
  return retcode::SUCCESS;
}

/**
 * @brief Find dataset by dataset uri
 *
 */
retcode DatasetService::findDataset(const DatasetId& id,
                                    FoundMetaHandler handler) {
  MetaService()->GetMeta(id, handler);
  return retcode::SUCCESS;
}


/**
 * @brief TODO Delete dataset
 * 1. Delete dataset meta from local storage.
 * 2. Delete dataset from libp2p network.
 *
 * @param id [input]: Dataset id
 * @return int
 */
retcode DatasetService::deleteDataset(const DatasetId& id) {
  return retcode::SUCCESS;
}

/**
 * @brief Consture datasets from yaml datasets configuration.
 * @param config_file_path [input]: Datasets configuration file path
 *
 */
void DatasetService::loadDefaultDatasets(const std::string& config_file_path) {
  LOG(INFO) << "📃 Load default datasets from config: " << config_file_path;
  YAML::Node config = YAML::LoadFile(config_file_path);
  auto& server_cfg = ServerConfig::getInstance();
  auto& nodelet_cfg = server_cfg.PublicServiceConfig();
  std::string nodelet_addr = nodelet_cfg.to_string();
  if (!config["datasets"]) {
    LOG(WARNING) << "no datasets found in config file, ignore....";
    return;
  }
  for (const auto& dataset : config["datasets"]) {
    try {
      auto dataset_type = dataset["model"].as<std::string>();
      auto dataset_uid = dataset["description"].as<std::string>();
      auto access_info = createAccessInfo(dataset_type, dataset);
      if (access_info == nullptr) {
        LOG(WARNING) << "create access info for "
                     << dataset_uid << " failed, to skip";
        continue;
      }
      std::string access_info_str = access_info->toString();
      auto driver = DataDirverFactory::getDriver(dataset_type,
                                                 nodelet_addr,
                                                 std::move(access_info));
      this->registerDriver(dataset_uid, driver);
      auto cursor = driver->read();
      if (cursor == nullptr) {
        LOG(WARNING) << "get cursor for " << dataset_uid << " failed";
        continue;
      }

      DatasetMeta meta;
      newDataset(driver, dataset_uid, access_info_str, &meta);
    } catch (std::exception& e) {
      LOG(ERROR) << e.what() << " ignore .....";
    }
  }
}

// Load dataset from local meta storage.
//
// IMPORTANT (2026-06-04 fix): Skip cross-node mirrored rows.
//   GetAllMetas returns ALL rows in the local fusion DB, which after cross-org
//   mirror (see offline-deploy/mirror-datasets.sh) includes datasets whose data
//   files live on remote nodes. Previously this function unconditionally called
//   meta.setServerInfo(nodelet_addr_) + PutMeta on every row, which rewrote
//   remote datasets' addresses to point at the LOCAL node — breaking cli PSI
//   cross-node dispatch because both parties would resolve to the same node.
//
//   We now only re-register and re-publish metas whose server address is empty
//   or already matches the local node. Cross-node rows are left alone — they
//   stay queryable via meta lookups but we don't pretend to own their data.
void DatasetService::restoreDatasetFromLocalStorage() {
  LOG(INFO) << "💾 Restore dataset from local storage...";
  std::vector<DatasetMeta> metas;
  MetaService()->GetAllMetas(&metas);
  size_t restored = 0, skipped = 0;
  for (auto& meta : metas) {
    try {
      auto server_info = meta.getServerInfo();
      // Skip rows owned by other nodes (mirrored from sibling fusion DBs).
      if (!server_info.empty() && server_info != nodelet_addr_) {
        VLOG(5) << "skip restore for cross-node dataset "
                << meta.getDescription()
                << " (server=" << server_info
                << " local=" << nodelet_addr_ << ")";
        ++skipped;
        continue;
      }
      auto meta_info = meta.toJSON();
      VLOG(5) << "meta_info: " << meta_info;
      std::string driver_type = meta.getDriverType();
      std::string fid = meta.getDescription();
      auto access_info = this->createAccessInfo(driver_type, meta_info);
      if (access_info == nullptr) {
        std::string err_msg = "create access info failed";
        continue;
      }
      auto driver = DataDirverFactory::getDriver(driver_type,
                                                nodelet_addr_,
                                                std::move(access_info));
      this->registerDriver(fid, driver);
      // meta.setDataURL(nodelet_addr_ + ":" + dataset_path);
      meta.setServerInfo(nodelet_addr_);
      // Publish dataset meta on public network.
      MetaService()->PutMeta(meta);
      ++restored;
    } catch (std::exception& e) {
      LOG(ERROR) << e.what();
    }
  }
  LOG(INFO) << "💾 Restore done: restored=" << restored
            << " skipped_cross_node=" << skipped;
}

void DatasetService::setMetaSearchTimeout(unsigned int timeout) {
  MetaService()->SetMetaSearchTimeout(timeout);
}

std::string DatasetService::getNodeletAddr(void) {
  return nodelet_addr_;
}

primihub::retcode DatasetService::registerDriver(
    const std::string& dataset_id,
    std::shared_ptr<primihub::DataDriver> driver) {
  std::lock_guard<std::shared_mutex> lck(driver_mtx_);
  auto it = driver_manager_.find(dataset_id);
  if (it != driver_manager_.end()) {
      LOG(WARNING) << "update driver for dataset: " << dataset_id;
      it->second = std::move(driver);
  } else {
      driver_manager_.insert({dataset_id, std::move(driver)});
  }
  VLOG(5) << "dataset uid: " << dataset_id << " regiseter success";
  return primihub::retcode::SUCCESS;
}

std::map<std::string, Node> DatasetService::getLocalPartyAccessInfo(
    const std::vector<DatasetWithParamTag>& datasets_with_tag) {
  std::map<std::string, Node> party_access_info;
  Node local_node;
  local_node.fromString(nodelet_addr_);
  for (const auto& pair : datasets_with_tag) {
    const auto& dataset_id = std::get<0>(pair);
    const auto& party_name = std::get<1>(pair);
    VLOG(5) << "looking up local dataset: " << dataset_id
            << " for party: " << party_name;
    {
      std::shared_lock<std::shared_mutex> lck(driver_mtx_);
      auto it = driver_manager_.find(dataset_id);
      if (it != driver_manager_.end()) {
        party_access_info[party_name] = local_node;
        LOG(INFO) << "found local dataset: " << dataset_id
                  << " for party: " << party_name;
      } else {
        LOG(WARNING) << "dataset: " << dataset_id
                     << " not found in local driver manager, "
                     << "driver_manager_ size: " << driver_manager_.size();
      }
    }
  }
  return party_access_info;
}

std::shared_ptr<primihub::DataDriver>
DatasetService::getDriver(const std::string& dataset_id, bool is_acces_info) {
  {
    std::shared_lock<std::shared_mutex> lck(driver_mtx_);
    auto it = driver_manager_.find(dataset_id);
    if (it != driver_manager_.end()) {
      return it->second;
    }
  }
  if (is_acces_info) {
    DatasetMetaInfo meta_info;
    VLOG(5) << dataset_id;
    nlohmann::json oJson = nlohmann::json::parse(dataset_id);
    meta_info.driver_type = oJson["type"].get<std::string>();
    VLOG(5) << meta_info.driver_type;
    meta_info.access_info = dataset_id;
    auto& schema = meta_info.schema;
    nlohmann::json filed_list =
        nlohmann::json::parse(oJson["schema"].get<std::string>());
    for (const auto& filed : filed_list) {
      for (const auto& [key, value] : filed.items()) {
        schema.push_back(std::make_tuple(key, value));
      }
    }
    try {
      auto access_info = createAccessInfo(meta_info.driver_type, meta_info);
      return DataDirverFactory::getDriver(meta_info.driver_type,
                                          DatasetLocation(),
                                          std::move(access_info));
    } catch (std::exception& e) {
      LOG(ERROR) << e.what();
      return nullptr;
    }
  }

  // get Meta using meta service
  std::shared_ptr<primihub::DataDriver> driver{nullptr};
  MetaService()->GetMeta(dataset_id,
                        [&](std::shared_ptr<DatasetMeta> meta) -> retcode {
    if (meta) {
      DatasetMetaInfo meta_info;
      meta_info.id = meta->id;
      meta_info.driver_type = meta->getDriverType();
      meta_info.access_info = meta->getAccessInfo();
      auto& schema = meta_info.schema;
      auto table_schema = dynamic_cast<TableSchema*>(meta->getSchema().get());
      for (const auto& field : table_schema->ArrowSchema()->fields()) {
        auto name = field->name();
        int type = field->type()->id();
        schema.push_back(std::make_tuple(name, type));
      }
      VLOG(5) << "driver_type: " << meta_info.driver_type << " "
          << "access_info: " << meta_info.access_info;
      try {
        auto access_info = createAccessInfo(meta_info.driver_type, meta_info);
        driver = DataDirverFactory::getDriver(meta_info.driver_type,
                                            DatasetLocation(),
                                            std::move(access_info));
      } catch (std::exception& e) {
        LOG(ERROR) << e.what();
        return retcode::FAIL;
      }
      return retcode::SUCCESS;
    } else {
      LOG(ERROR) << "get dataset meta failed";
      return retcode::FAIL;
    }
  });
  return driver;
}

primihub::retcode DatasetService::unRegisterDriver(
    const std::string& dataset_id) {
    std::lock_guard<std::shared_mutex> lck(driver_mtx_);
  auto it = driver_manager_.find(dataset_id);
  if (it != driver_manager_.end()) {
      LOG(WARNING) << "erase driver for dataset: " << dataset_id;
      driver_manager_.erase(dataset_id);
  } else {  // do nothing
  }
  return primihub::retcode::SUCCESS;
}

DataSetAccessInfoPtr DatasetService::createAccessInfo(
    const std::string& driver_type,
    const YAML::Node& meta_info) {
  return DataDirverFactory::createAccessInfo(driver_type, meta_info);
}

DataSetAccessInfoPtr DatasetService::createAccessInfo(
    const std::string& driver_type,
    const std::string& meta_info) {
  return DataDirverFactory::createAccessInfo(driver_type, meta_info);
}

std::unique_ptr<DataSetAccessInfo> DatasetService::createAccessInfo(
    const std::string& driver_type,
    const DatasetMetaInfo& meta_info) {
  return DataDirverFactory::createAccessInfo(driver_type, meta_info);
}

}  // namespace primihub::service
