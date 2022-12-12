// C++ STL
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <glob.h>
#include <string>
#include <libgen.h>

// Phase One SDKs
#include <P1Camera.hpp>
#include <P1Image.hpp>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//using
using namespace std; //making the code simpler. 
using std::cout;
using P1::CameraSdk::Camera;
using P1::CameraSdk::IIQImageFile;
using P1::ImageSdk::RawImage;
using P1::ImageSdk::BitmapImage;
using P1::ImageSdk::ConvertConfig;
using P1::ImageSdk::BitmapBase;
using P1::ImageSdk::SensorBayerOutput;
using P1::ImageSdk::BayerFormat;
using P1::ImageSdk::DecodeConfig;

//paths for where to save the .bin and .jpg files
string binPath = "../bin/ ";
string binType = ".bin";
string jpgPath = "../jpg/";
string jpgType = ".jpg";

int main(int argc, const char* argv[])
{
glob_t glob_result; //using glob to locate all the IIQ files that needs to be loaded.
int k = 0; //counting the number of IIQ files.
glob("../IIQ/*",GLOB_TILDE,NULL,&glob_result); //loads the file names in given folder.
for(unsigned int i=0; i<glob_result.gl_pathc; ++i){ //printing loaded files for debugging and counting images.
  cout << glob_result.gl_pathv[i] << "\n";
  cout << basename(glob_result.gl_pathv[i]) << "\n";
  k = i;
}


#pragma region ImageSDK Processing
    P1::ImageSdk::SetSensorProfilesLocation("../ImageSDK/SensorProfiles"); // Tell ImageSDK where it can find SensorProfiles
    //First creating .bin files for each image
    cout << "creating .bin files \n";
    for(unsigned int i=0; i<=k; ++i){
        RawImage iiqImage(glob_result.gl_pathv[i]); //load image file from glob list
        SensorBayerOutput bayerOutput = iiqImage.Decode(DecodeConfig::Defaults); //decode image
        uint8_t *pixels = bayerOutput.Data().get(); // pointer to the bayer pixels
        // write the pixel data to disk in .bin file.
        string binOutputName = binPath + basename(glob_result.gl_pathv[i]) + binType; 
        std::ofstream file(binOutputName, std::ios::binary | std::ios::trunc);
        file.write((char*)pixels, bayerOutput.ByteSize());
    }
    cout << "Done .bin files \n";

    //then creating .jpg files for each image
    cout << "Creating .jpg files \n";
    for(unsigned int i=0; i<=k; ++i){
        RawImage iiqImage(glob_result.gl_pathv[i]); //load image file from glob list
        // create possibility for exporting a JPG by converting to RGB matrix
        float scaleFactor = 1;
        ConvertConfig config;
        config.SetOutputScale(scaleFactor); // scale of original size
        config.SetOutputFormat(P1::ImageSdk::BitmapFormat::Rgb24); //defining to use 8 bit RGB, 8 bit pr. channel equals 24.
        BitmapImage rgbBitmap = iiqImage.Convert(config); //Convert (Raw) Bayer data into an RGB bitmap.
        cv::Mat rgbImage(cv::Size(rgbBitmap.Width(), rgbBitmap.Height()), CV_8UC3, rgbBitmap.Data().get()); //creating the opencv mat with details for image to later export.
        cv::Mat bgrImage;//Convert from RGB to OpenCV's BGR style colorspace...
        cv::cvtColor(rgbImage, bgrImage, cv::COLOR_RGB2BGRA);
        std::stringstream outputFilename; //defining filename
        string jpgOutputName = jpgPath + basename(glob_result.gl_pathv[i]) + jpgType;
        cv::imwrite(jpgOutputName, bgrImage); //saving image
    }
    cout << "Done .jpg files \n";
#pragma endregion
    return 0;
}