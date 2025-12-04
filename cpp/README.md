# C++ Fatigue Monitor

This is a C++ rewrite of the Fatigue Monitor application, designed for better performance.

## Prerequisites

1.  **C++ Compiler**: Visual Studio (MSVC) or MinGW/GCC.
2.  **CMake**: Version 3.10 or higher.
3.  **OpenCV**: Version 4.x with `opencv_contrib` modules (specifically `face` module).
    *   If using `vcpkg`: `vcpkg install opencv[contrib]:x64-windows`

## Required Model Files

You must place the following files in the same directory as the executable (or the working directory):

1.  **`haarcascade_frontalface_alt2.xml`**
    *   Found in OpenCV `data/haarcascades` folder.
    *   [Download from OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml)

2.  **`lbfmodel.yaml`**
    *   Required for Face Landmark detection.
    *   [Download from OpenCV 3rdparty](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml)

## Build Instructions

1.  Open a terminal in this `cpp` directory.
2.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake:
    ```bash
    cmake ..
    ```
    *   If OpenCV is not found automatically, specify the path: `-DOpenCV_DIR=C:/path/to/opencv/build`
4.  Build:
    ```bash
    cmake --build . --config Release
    ```

## Running

1.  Ensure the model files are in the `Release` folder (or wherever the `.exe` is).
2.  Run `FatigueMonitor.exe`.
