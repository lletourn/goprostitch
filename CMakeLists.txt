cmake_minimum_required(VERSION 3.16)
project(goprostitch)

find_package(spdlog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/framestitcher.cpp ${CMAKE_SOURCE_DIR}/src/inputprocessor.cpp ${CMAKE_SOURCE_DIR}/src/outputencoder.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)
