# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Source files in this directory
set(SIGMA_SOURCE_FILES ${SIGMA_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/batchencoder.cu
        ${CMAKE_CURRENT_LIST_DIR}/ciphertext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ckks.cu
        ${CMAKE_CURRENT_LIST_DIR}/context.cpp
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptor.cu
        ${CMAKE_CURRENT_LIST_DIR}/evaluator.cu
        ${CMAKE_CURRENT_LIST_DIR}/keygenerator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kswitchkeys.cpp
        ${CMAKE_CURRENT_LIST_DIR}/memorymanager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/modulus.cpp
        ${CMAKE_CURRENT_LIST_DIR}/plaintext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/randomgen.cpp
        ${CMAKE_CURRENT_LIST_DIR}/serialization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/valcheck.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernelprovider.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernelutils.cu
)

# Add header files for installation
install(
        FILES
        ${CMAKE_CURRENT_LIST_DIR}/batchencoder.h
        ${CMAKE_CURRENT_LIST_DIR}/ciphertext.h
        ${CMAKE_CURRENT_LIST_DIR}/ckks.cuh
        ${CMAKE_CURRENT_LIST_DIR}/modulus.h
        ${CMAKE_CURRENT_LIST_DIR}/context.h
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.h
        ${CMAKE_CURRENT_LIST_DIR}/dynarray.h
        ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.h
        ${CMAKE_CURRENT_LIST_DIR}/encryptor.h
        ${CMAKE_CURRENT_LIST_DIR}/evaluator.cuh
        ${CMAKE_CURRENT_LIST_DIR}/galoiskeys.h
        ${CMAKE_CURRENT_LIST_DIR}/keygenerator.h
        ${CMAKE_CURRENT_LIST_DIR}/kswitchkeys.h
        ${CMAKE_CURRENT_LIST_DIR}/memorymanager.h
        ${CMAKE_CURRENT_LIST_DIR}/plaintext.cuh
        ${CMAKE_CURRENT_LIST_DIR}/publickey.h
        ${CMAKE_CURRENT_LIST_DIR}/randomgen.h
        ${CMAKE_CURRENT_LIST_DIR}/randomtostd.h
        ${CMAKE_CURRENT_LIST_DIR}/relinkeys.h
        ${CMAKE_CURRENT_LIST_DIR}/sigma.h
        ${CMAKE_CURRENT_LIST_DIR}/secretkey.h
        ${CMAKE_CURRENT_LIST_DIR}/serializable.h
        ${CMAKE_CURRENT_LIST_DIR}/serialization.h
        ${CMAKE_CURRENT_LIST_DIR}/valcheck.h
        ${CMAKE_CURRENT_LIST_DIR}/version.h
        ${CMAKE_CURRENT_LIST_DIR}/kernelprovider.cuh
        ${CMAKE_CURRENT_LIST_DIR}/kernelutils.cuh
        DESTINATION
        ${SIGMA_INCLUDES_INSTALL_DIR}/sigma
)

add_subdirectory(util)
set(SIGMA_SOURCE_FILES ${SIGMA_SOURCE_FILES} PARENT_SCOPE)
