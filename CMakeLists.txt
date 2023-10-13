cmake_minimum_required(VERSION 3.16)
project(goprostitch)

add_compile_options("-fstack-protector-all")

find_package(spdlog REQUIRED)
find_package(RapidJSON REQUIRED)
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

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/framestitcher.cpp ${CMAKE_SOURCE_DIR}/src/compositing.cpp ${CMAKE_SOURCE_DIR}/src/inputprocessor.cpp ${CMAKE_SOURCE_DIR}/src/outputencoder.cpp ${CMAKE_SOURCE_DIR}/src/readers.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)

add_executable(memtst ${CMAKE_SOURCE_DIR}/src/memtst.cpp ${CMAKE_SOURCE_DIR}/src/framestitcher.cpp ${CMAKE_SOURCE_DIR}/src/compositing.cpp ${CMAKE_SOURCE_DIR}/src/inputprocessor.cpp ${CMAKE_SOURCE_DIR}/src/readers.cpp)
set_property(TARGET memtst PROPERTY CXX_STANDARD 17)
target_link_libraries(memtst PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)

add_executable(calibrate ${CMAKE_SOURCE_DIR}/src/calibrate.cpp)
set_property(TARGET calibrate PROPERTY CXX_STANDARD 17)
target_link_libraries(calibrate PRIVATE ${OpenCV_LIBS} spdlog::spdlog)

add_executable(manualkeypoints ${CMAKE_SOURCE_DIR}/src/manualkeypoints.cpp ${CMAKE_SOURCE_DIR}/src/pointselector.cpp ${CMAKE_SOURCE_DIR}/src/readers.cpp)
set_property(TARGET manualkeypoints PROPERTY CXX_STANDARD 17)
target_link_libraries(manualkeypoints PRIVATE ${OpenCV_LIBS} spdlog::spdlog)

add_executable(viewstitch ${CMAKE_SOURCE_DIR}/src/viewstitch.cpp ${CMAKE_SOURCE_DIR}/src/framestitcher.cpp ${CMAKE_SOURCE_DIR}/src/compositing.cpp ${CMAKE_SOURCE_DIR}/src/inputprocessor.cpp ${CMAKE_SOURCE_DIR}/src/readers.cpp)
set_property(TARGET viewstitch PROPERTY CXX_STANDARD 17)
target_link_libraries(viewstitch PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)

add_executable(paramextractor ${CMAKE_SOURCE_DIR}/src/paramextractor.cpp ${CMAKE_SOURCE_DIR}/src/framestitcher.cpp ${CMAKE_SOURCE_DIR}/src/compositing.cpp)
set_property(TARGET paramextractor PROPERTY CXX_STANDARD 17)
target_link_libraries(paramextractor PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)

add_executable(seamfinder ${CMAKE_SOURCE_DIR}/src/seamfinder.cpp ${CMAKE_SOURCE_DIR}/src/compositing.cpp ${CMAKE_SOURCE_DIR}/src/readers.cpp)
set_property(TARGET seamfinder PROPERTY CXX_STANDARD 17)
target_link_libraries(seamfinder PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)

add_executable(orig_stitching_detailed ${CMAKE_SOURCE_DIR}/src/stitching_detailed.orig.cpp)
set_property(TARGET orig_stitching_detailed PROPERTY CXX_STANDARD 17)
target_link_libraries(orig_stitching_detailed PRIVATE PkgConfig::LIBAV ${OpenCV_LIBS} spdlog::spdlog)
