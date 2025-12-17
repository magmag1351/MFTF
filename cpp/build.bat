@echo off
setlocal

echo Checking for CMake...
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake is not found in your PATH.
    echo Please install CMake: https://cmake.org/download/
    echo And ensure you have a C++ compiler installed (e.g., Visual Studio 2022 with C++ workload).
    pause
    exit /b 1
)

echo Creating build directory...
if not exist build mkdir build
cd build

echo Configuring with CMake...
cmake ..
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed.
    echo You might need to specify the OpenCV path if it's not found automatically.
    echo Example: cmake .. -DOpenCV_DIR="C:/path/to/opencv/build"
    pause
    exit /b 1
)

echo Building project...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo Build successful!
echo You can now run the application from the build/Release directory.
pause
