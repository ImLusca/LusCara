cmake_minimum_required(VERSION 3.10)
project(FaceRecProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Diretório do ONNX Runtime: pode ser sobrescrito via -DONNX_DIR=/caminho/para/onnxruntime
set(ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/onnxruntime-linux-x64-1.17.0" CACHE PATH "ONNX Runtime root directory")
include_directories(${ONNX_DIR}/include)
link_directories(${ONNX_DIR}/lib)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(facenet_embedder ${SOURCES})

target_link_libraries(facenet_embedder
  ${OpenCV_LIBS}
  onnxruntime
)

set_target_properties(facenet_embedder PROPERTIES INSTALL_RPATH "$ORIGIN/lib")

# Benchmark de acurácia/latência no LFW
add_executable(benchmark_lfw
  src/FacenetEngine.cpp
  src/ImageProcessor.cpp
  src/MathUtils.cpp
  tools/benchmark_lfw.cpp)

target_link_libraries(benchmark_lfw
  ${OpenCV_LIBS}
  onnxruntime
)

set_target_properties(benchmark_lfw PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
