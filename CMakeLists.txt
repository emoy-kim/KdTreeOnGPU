cmake_minimum_required(VERSION 3.8)

project("KdTreeOnGPU")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		source/kdtree.cpp
)

configure_file(include/project_constants.h.in ${PROJECT_BINARY_DIR}/project_constants.h @ONLY)

include_directories("include")
include_directories("${CMAKE_SOURCE_DIR}/3rd_party/glm")

add_executable(KdTreeOnGPU ${SOURCE_FILES})

target_link_libraries(
     KdTreeOnGPU
        pthread
        dl
)

target_include_directories(KdTreeOnGPU PUBLIC ${CMAKE_BINARY_DIR})