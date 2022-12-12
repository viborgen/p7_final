cmake_minimum_required(VERSION 3.13)

# Add FetchContent functionality
include(FetchContent)

project(P1SdkProject)

# SDK Major version
set(SDK_MAJOR_VERSION "2")

# System name (lower case)
string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME_LC)


# File format extension
if(WIN32)
set(SDK_PACKAGE_EXT ".zip")
else()
set(SDK_PACKAGE_EXT ".tgz")
endif()

# Account for ARM based linux systems
if(NOT APPLE AND UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
set(LINUX_ARCH "-arm64")
else()
set(LINUX_ARCH "")
endif()

# Setup CameraSDK Download
FetchContent_Declare(CameraSDK 
    URL https://developer.phaseone.com/sdk/releases/camerasdk/${SDK_MAJOR_VERSION}/p1camerasdk-${SYSTEM_NAME_LC}${LINUX_ARCH}${SDK_PACKAGE_EXT}
    SOURCE_DIR CameraSDK
)
# Download CameraSDK for the current system
message(STATUS "Downloading CameraSDK...")
FetchContent_Populate(CameraSDK)
find_package(CameraSDK CONFIG REQUIRED HINTS ${CMAKE_CURRENT_BINARY_DIR}/CameraSDK)

# Setup ImageSDK Download
FetchContent_Declare(ImageSDK 
    URL https://developer.phaseone.com/sdk/releases/imagesdk/${SDK_MAJOR_VERSION}/p1imagesdk-${SYSTEM_NAME_LC}${LINUX_ARCH}${SDK_PACKAGE_EXT}
    SOURCE_DIR ImageSDK
)
# Download ImageSDK for the current system
message(STATUS "Downloading ImageSDK...")
FetchContent_Populate(ImageSDK)
find_package(ImageSDK CONFIG REQUIRED HINTS ${CMAKE_CURRENT_BINARY_DIR}/ImageSDK)


#find opencv SDK
find_package(OpenCV REQUIRED HINTS 
    /home/mathias/opencv/include
    /home/mathias/opencv/include
)

## Add sample executable
add_executable(SDK main.cpp)

## Link static with SDKs
target_link_libraries(SDK PRIVATE
    CameraSDK::CameraSdkCppStatic
    ImageSDK::ImageSdkCppStatic
    ${OpenCV_LIBS}
)