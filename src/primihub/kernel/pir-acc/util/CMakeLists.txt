# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Source files in this directory
set(CUSEAL_UTIL_FILES ${CUSEAL_UTIL_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/configmanager.h
        ${CMAKE_CURRENT_LIST_DIR}/configmanager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/vectorutil.h
        ${CMAKE_CURRENT_LIST_DIR}/vectorutil.cu
        ${CMAKE_CURRENT_LIST_DIR}/keyutil.h
        ${CMAKE_CURRENT_LIST_DIR}/keyutil.cpp
        ${CMAKE_CURRENT_LIST_DIR}/safequeue.h
)

set(CUSEAL_UTIL_FILES ${CUSEAL_UTIL_FILES} PARENT_SCOPE)