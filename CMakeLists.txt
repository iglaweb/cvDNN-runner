cmake_minimum_required(VERSION 2.8)
project(dnn_executor)

if(NOT APPLE)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")
endif(NOT APPLE)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Release)


if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -frtti -fexceptions -D WITH_OPENCL=ON")
else(APPLE) # Nano
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -fPIC")
endif(APPLE)


set(SOURCE_FILES main.cpp)


SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# detect operating system and host processor
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
message(STATUS "The host processor is ${CMAKE_HOST_SYSTEM_PROCESSOR}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}" )
message(STATUS "Configuration types: ${CMAKE_CONFIGURATION_TYPES}" )

# Detect 64 or 32 bit
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64-bit project
    SET(64_BIT_OS TRUE)
    message(STATUS "A 64-bit OS detected" )
else()
    SET(64_BIT_OS FALSE)
    message(STATUS "A 32-bit OS detected" )
endif()


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


file(GLOB srcs *.cpp *.c *.cc)
file(GLOB hdrs *.hpp *.h)
file(GLOB_RECURSE source_list)


include_directories("${CMAKE_CURRENT_LIST_DIR}")

include_directories(/usr/X11/lib)
include_directories(/usr/local/lib)
include_directories(/usr/local/include)


# eval dnn executor
add_executable(eval_dnn_executor_exec eval_dnn_executor/main.cpp)
target_link_libraries(eval_dnn_executor_exec ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} ${LIBRARIES}))
