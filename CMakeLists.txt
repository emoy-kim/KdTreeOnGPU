cmake_minimum_required(VERSION 3.8)

project("KdTreeOnGPU")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		source/text.cpp
		source/light.cpp
		source/camera.cpp
		source/shader.cpp
		source/kdtree_shader.cpp
		source/kdtree.cpp
		source/object.cpp
		source/kdtree_object.cpp
		source/renderer.cpp
)

configure_file(include/project_constants.h.in ${PROJECT_BINARY_DIR}/project_constants.h @ONLY)

find_package(CUDA)

include_directories("include")
include(cmake/add-libraries-linux.cmake)

if(CUDA_FOUND)
	set(
      CUDA_SOURCE_FILES
			source/cuda/kdtree.cu
	)
   cuda_add_executable(KdTreeOnGPU ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
	target_compile_definitions(KdTreeOnGPU PRIVATE USE_CUDA)
else()
	add_executable(KdTreeOnGPU ${SOURCE_FILES})
endif()

include(cmake/target-link-libraries-linux.cmake)

target_include_directories(KdTreeOnGPU PUBLIC ${CMAKE_BINARY_DIR})