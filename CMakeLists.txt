cmake_minimum_required(VERSION 3.27)

project(dla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

message("Build type: " ${CMAKE_BUILD_TYPE})
message("Compiler: " ${CMAKE_CXX_COMPILER})

add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
  nlohmann
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(nlohmann)
include_directories(${nlohmann_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -DDEBUG -DSERIAL")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

set(GLAD_PATH "/Users/eliotheinrich/Projects/glad")
set(GLFW_PATH "/opt/homebrew/Cellar/glfw/3.4")
include_directories(src "${GLAD_PATH}/include" "${GLFW_PATH}/include")
link_directories("${GLFW_PATH}/lib")

# --- MAIN --- #
add_executable(main src/glad.cpp src/main.cpp)
target_include_directories(main PRIVATE src)

set(CMAKE_DEBUG_POSTFIX "d")
set_target_properties(main PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_link_libraries(main PRIVATE glfw3 pthread)
