#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#endif

#include "pathFounder.h"

std::string loadKernelSource(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.good()) {
    std::cerr << "Error loading kernel source from file: " << filename << std::endl;
    exit(1);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

using json = nlohmann::json;

int main(){

  // Initialize OpenCL
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  auto platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  auto device = devices.front();

  for (auto& d : devices) {
    if (d.getInfo<CL_DEVICE_VENDOR>().find("NVIDIA") != std::string::npos) {
      device = d;
      break;
    }
  }

  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  std::string kernelSource = loadKernelSource("/home/ugurcan/Documents/yl/include/brightSpotKernel.cl");
  cl::Program::Sources sources;
  sources.push_back({kernelSource.c_str(), kernelSource.length()});

  cl::Program program(context, sources);
  auto buildResult = program.build({device});
  if (buildResult != CL_SUCCESS) {
    std::cerr << "Error building kernel: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    exit(1);
  }

  std::string path_string = "/home/ugurcan/Documents/yl/bruhaga/AhmetV2";
  std::vector<std::filesystem::path> path_vector = visua::PathFounder::getSubDirectories(path_string);

  std::vector<std::filesystem::path> sub_catagory_names = {"LCC", "LMLO", "RCC", "RMLO"};

  std::filesystem::path json_path = path_vector.at(1) / sub_catagory_names.at(2);
  json_path += ".json";

  std::filesystem::path image_path = path_vector.at(1) / sub_catagory_names.at(2);
  image_path += ".png";

  std::ifstream jsonFile(json_path);
  json jsonData;
  jsonFile >> jsonData;

  cv::Mat image = cv::imread(image_path);
  cv::Mat copy_image;
  image.copyTo(copy_image);
  cv::resize(copy_image, copy_image, cv::Size(800, 600));

  cv::imshow("Original", copy_image);


  if (image.empty()) {
    std::cout << "Could not load image: " << image_path << std::endl;
    return -1;
  }

  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

  cv::resize(grayImage, grayImage, cv::Size(800, 600));

  float i = 1.0;
  int halfWindow = 5;
  bool stopper = false;
  while(true){

    if (i > 3.0){
      i = 1.0;
    }

    cl::Kernel kernel(program, "brightSpotsKernel");

    // Create buffers for the input and output
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            grayImage.total() * grayImage.elemSize(), grayImage.data);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, 
                            grayImage.total() * grayImage.elemSize());

    // Set the kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<int>(grayImage.cols));
    kernel.setArg(3, static_cast<int>(grayImage.rows));
    kernel.setArg(4, halfWindow);  // Example for halfWindow
    kernel.setArg(5, i);  // Example for thresholdFactor
    

    // Run the kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grayImage.cols, grayImage.rows), cl::NullRange);
    queue.finish();

    // Read back the result
    cv::Mat outputImage(grayImage.size(), grayImage.type());
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, 
                            grayImage.total() * grayImage.elemSize(), outputImage.data);


    cv::putText(outputImage, "Thr. Factor: " + std::to_string(i), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));
    cv::putText(outputImage, "Half Window: " + std::to_string(halfWindow), cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));


    // Display or save the result
    cv::imshow("Output Image", outputImage);
    char c=(char) cv::waitKey(10);
    if(c==27) break;
    else if (c == 32) stopper = !stopper;
    else if (c == 119) halfWindow++;
    else if (c == 115) halfWindow--;
    if (!stopper){
      i += 0.01;
    }
  }

  /*

  // Iterate over shapes in the JSON and draw polygons
    for (const auto& shape : jsonData["shapes"]) {
      // Check if the shape is a polygon
      if (shape["shape_type"] == "polygon") {
        std::vector<cv::Point> points;
        for (const auto& pt : shape["points"]) {
          points.push_back(cv::Point(pt[0], pt[1]));
        }

        // Draw the polygon
        const cv::Point* pts = (const cv::Point*) cv::Mat(points).data;
        int npts = cv::Mat(points).rows;

        cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 3);
      }
    }

    int histSize = 256; // number of bins
    float range[] = { 1, 256 }; // exclude 0
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Compute the average of histogram values excluding the zero bin
    double sum = 0;
    int count = 0;
    for (int i = 1; i < histSize; ++i) { // start from 1 to exclude the zero bin
        sum += hist.at<float>(i) * i;
        count += hist.at<float>(i);
    }
    double avg = count > 0 ? sum / count : 0;

    cv::Mat resizedImage_org;
    cv::resize(image, resizedImage_org, cv::Size(800, 600));

    cv::Mat resizedImage_gray;
    cv::resize(grayImage, resizedImage_gray, cv::Size(800, 600));

    cv::Mat output = cv::Mat::zeros(resizedImage_gray.size(), resizedImage_gray.type());

    int windowSize = 30; // Define the local window size
    int halfWindow = windowSize / 2;
    double thresholdFactor = 2.0; // Factor to determine how much brighter the pixel should be compared to the local average

    for (int y = halfWindow; y < resizedImage_gray.rows - halfWindow; ++y) {
      for (int x = halfWindow; x < resizedImage_gray.cols - halfWindow; ++x) {
        // Define the local region
        cv::Rect localRect(x - halfWindow, y - halfWindow, windowSize, windowSize);
        cv::Mat localArea = resizedImage_gray(localRect);

        // Compute the mean of the local area excluding zeros
        double sum = 0;
        int count = 0;
        for (int ly = 0; ly < localArea.rows; ++ly) {
          for (int lx = 0; lx < localArea.cols; ++lx) {
            uchar val = localArea.at<uchar>(ly, lx);
            if (val != 0) {
              sum += val;
              count++;
            }
          }
        }

        double localMean = count > 0 ? sum / count : 0;

        // Compare the current pixel to the local mean
        uchar currentPixelValue = resizedImage_gray.at<uchar>(y, x);
        if (currentPixelValue > localMean * thresholdFactor && currentPixelValue != 0) {
          output.at<uchar>(y, x) = 255; // Mark as bright spot
        }
      }
    }

    cv::Mat resizedImage_out;
    cv::resize(output, resizedImage_out, cv::Size(800, 600));

    cv::imshow("Bright Spot Image", resizedImage_out);

    // Set a threshold based on the average
    cv::imshow("Original Image", resizedImage_org);

    double thresholdValue = avg;
    cv::Mat binaryImage;
    int i = 0;
    bool stopper = false;
    while(true){
      if (i == 255){
        i = 0;
      }

      cv::threshold(grayImage, binaryImage, i, 255, cv::THRESH_BINARY);

      cv::Mat resizedImage;
      cv::resize(binaryImage, resizedImage, cv::Size(800, 600));

      cv::putText(resizedImage, std::to_string(i), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));
      cv::putText(resizedImage, "Hist Avr: " + std::to_string(thresholdValue), cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));

      // Display the image
      cv::imshow("Image with Polygons", resizedImage);


      // Press  ESC on keyboard to exit
      if (i == int (thresholdValue)){
        char c=(char) cv::waitKey(500);
        if(c==27) break;
      }
      else{
        char c=(char) cv::waitKey(10);
        if(c==27) break;
        else if (c == 32) stopper = !stopper;
      }

      if (!stopper){
        i++;
      }
    }
    
    cv::waitKey(0);

    */

  return 0;
}