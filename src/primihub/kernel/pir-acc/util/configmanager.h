//
// Created by scwang on 2023/11/9.
//

#ifndef CUSEAL_CONFIGMANAGER_H
#define CUSEAL_CONFIGMANAGER_H

#include "../extern/jsoncpp/json/json.h"

#include <cstdint>
#include <string>

#define ConfigUtil util::ConfigManager::singleton

namespace util {

    class ConfigManager {

    public:

        static ConfigManager singleton;

        int64_t int64ValueForKey(const std::string& key);

        std::vector<int> intVectorValueForKey(const std::string& key);

    private:

        ConfigManager();

        bool initialized = false;
        Json::Value configValue;

    };

}


#endif //CUSEAL_CONFIGMANAGER_H
