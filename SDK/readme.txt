main.cpp loads all images in the IIQ folder
Creates an array for the RAW data and saves it as .bin files in bin folder
Converts to RGB and stores in jpg folder



build on cmake and gcc, and uses Phase One ImageSDK and openCV.

getting started terminal commands
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .