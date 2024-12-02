cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

project(cuSEAL  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


add_subdirectory(SIGMA)
add_subdirectory(extern/cnpy)
add_subdirectory(extern/jsoncpp)


set(CUSEAL_UTIL_FILES "")
add_subdirectory(util)


add_library(sealpir pir.hpp pir.cu pir_client.hpp pir_client.cpp pir_server.cuh
        pir_server.cu)
target_link_libraries(sealpir SIGMA::sigma)
#[[target_link_libraries(sealpir PRIVATE cnpy)
target_link_libraries(sealpir PRIVATE jsoncpp)
target_link_libraries(sealpir PRIVATE -lcurand)]]

add_executable(main main.cpp ${CUSEAL_UTIL_FILES})
target_link_libraries(main PRIVATE sealpir)
target_link_libraries(main PRIVATE cnpy)
target_link_libraries(main PRIVATE jsoncpp)
target_link_libraries(main PRIVATE -lcurand)

#[[add_executable(cuSEAL
        main.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(cuSEAL PRIVATE SIGMA::sigma)
target_link_libraries(cuSEAL PRIVATE cnpy)
#target_include_directories(cuSEAL PRIVATE cnpy)
target_link_libraries(cuSEAL PRIVATE jsoncpp)



add_executable(keygen
        keygen.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(keygen PRIVATE SIGMA::sigma)
target_link_libraries(keygen PRIVATE cnpy)
target_link_libraries(keygen PRIVATE jsoncpp)
target_link_libraries(keygen PRIVATE -lcurand)


add_executable(encrypt
        encrypt.cu
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(encrypt PRIVATE SIGMA::sigma)
target_link_libraries(encrypt PRIVATE cnpy)
target_link_libraries(encrypt PRIVATE jsoncpp)
target_link_libraries(encrypt PRIVATE -lcurand)


add_executable(calculate
        calculate.cu
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(calculate PRIVATE SIGMA::sigma)
target_link_libraries(calculate PRIVATE cnpy)
target_link_libraries(calculate PRIVATE jsoncpp)
target_link_libraries(calculate PRIVATE -lcurand)

add_executable(decrypt
        decrypt.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(decrypt PRIVATE SIGMA::sigma)
target_link_libraries(decrypt PRIVATE cnpy)
target_link_libraries(decrypt PRIVATE jsoncpp)
target_link_libraries(decrypt PRIVATE -lcurand)]]
