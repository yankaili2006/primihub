//
// Created by scwang on 2023/11/9.
//

#include "configmanager.h"

#include <fstream>

namespace util {

    std::string config_path = "../data/config.json";

    ConfigManager ConfigManager::singleton;

    ConfigManager::ConfigManager() {
        std::ifstream infile(config_path.c_str(), std::ios::binary);
        if (!infile.is_open()) {
            throw std::invalid_argument("Open config json file failed!");
        }

        Json::Reader json_reader;
        bool success = json_reader.parse(infile, configValue);
        if (!success) {
            throw std::invalid_argument("Parse config json file failed!");
        }
        infile.close();
    }

    int64_t ConfigManager::int64ValueForKey(const std::string& key) {
        return configValue[key].asInt64();
    }

    std::vector<int> ConfigManager::intVectorValueForKey(const std::string& key) {
        std::vector<int> numbers;
        const Json::Value numbersJson = configValue[key];
        for (const auto& number : numbersJson) {
            numbers.push_back(number.asInt());
        }
        return numbers;
    }

}
