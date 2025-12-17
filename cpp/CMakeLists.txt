cmake_minimum_required(VERSION 3.10)
project(FatigueMonitor)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    main.cpp
    InputMonitor.cpp
    FaceMonitor.cpp
    FatigueApp.cpp
)

# Executable
add_executable(FatigueMonitor ${SOURCES})

# Link libraries
target_link_libraries(FatigueMonitor ${OpenCV_LIBS})

# Windows specific libraries for Hooks and Sound
if(WIN32)
    target_link_libraries(FatigueMonitor User32.lib Gdi32.lib)
endif()
