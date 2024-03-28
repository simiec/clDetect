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

// Function to load OpenCL kernel source from a file
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        std::cerr << "Error loading kernel source from file: " << filename << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Alias for the JSON library
using json = nlohmann::json;

int main(){

  // Initialize OpenCL by selecting the first available platform and device (preferably NVIDIA GPU)
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

  // Load and build the OpenCL program from the kernel source file
  std::string kernelSource = loadKernelSource("/home/ugurcan/Documents/yl/include/brightSpotKernel.cl");
  cl::Program::Sources sources;
  sources.push_back({kernelSource.c_str(), kernelSource.length()});

  cl::Program program(context, sources);
  auto buildResult = program.build({device});
  if (buildResult != CL_SUCCESS) {
    std::cerr << "Error building kernel: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    exit(1);
  }

  // Load image and JSON data from the specified directory path
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

  // Create Image variable from json given path
  cv::Mat image = cv::imread(image_path);

  // Check image
  if (image.empty()) {
    std::cout << "Could not load image: " << image_path << std::endl;
    return -1;
  }

  // Convert the image to grayscale and resize for processing
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

  cv::resize(grayImage, grayImage, cv::Size(800, 600));

  float i = 1.0;
  int halfWindow = 40;
  bool stopper = false;
  while(true){

    if (i > 3.0){
      i = 1.0;
    }

    /* Create buffers and set kernel arguments for edge detection */
    cl::Kernel kernel_edge_detection(program, "deleteNearZeroPointsKernel");

    cl::Buffer inputBuffer2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            grayImage.total() * grayImage.elemSize(), grayImage.data);

    cl::Buffer outputBuffer2(context, CL_MEM_WRITE_ONLY, 
                            grayImage.total() * grayImage.elemSize());


    // Set the kernel arguments
    kernel_edge_detection.setArg(0, inputBuffer2);
    kernel_edge_detection.setArg(1, outputBuffer2);
    kernel_edge_detection.setArg(2, static_cast<int>(grayImage.cols));
    kernel_edge_detection.setArg(3, static_cast<int>(grayImage.rows));
    int neighborDistance = 30;
    int threshold = 5;
    kernel_edge_detection.setArg(4, neighborDistance);
    kernel_edge_detection.setArg(5, threshold);

    queue.enqueueNDRangeKernel(kernel_edge_detection, cl::NullRange, cl::NDRange(grayImage.cols, grayImage.rows), cl::NullRange);
    queue.finish();

    // Read the processed image from the buffer
    cv::Mat outputImage2(grayImage.size(), grayImage.type());
    queue.enqueueReadBuffer(outputBuffer2, CL_TRUE, 0, 
                            grayImage.total() * grayImage.elemSize(), outputImage2.data);

    /* Bright spot detection kernel setup and execution */
    cl::Kernel kernel(program, "brightSpotsKernel");

    // Create buffers for the bright spot detection
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            outputImage2.total() * outputImage2.elemSize(), outputImage2.data);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, 
                            outputImage2.total() * outputImage2.elemSize());

    // Set the kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<int>(grayImage.cols));
    kernel.setArg(3, static_cast<int>(grayImage.rows));
    kernel.setArg(4, halfWindow);  // Example for halfWindow
    kernel.setArg(5, i);  // Example for thresholdFactor
    

    // Run the kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(outputImage2.cols, outputImage2.rows), cl::NullRange);
    queue.finish();

    // Read back the result
    cv::Mat outputImage(outputImage2.size(), outputImage2.type());
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, 
                            outputImage2.total() * outputImage2.elemSize(), outputImage.data);

    /* Setup and execute bounding box kernel to find areas of interest */
    cl::Kernel kernel_bounding_box(program, "findGlobalBoundingBoxKernel");

    // Create buffers
    cl::Buffer inputBufferBB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           outputImage.total() * outputImage.elemSize(), outputImage.data);
    // Initialize the bounding box values: [INT_MAX, 0, INT_MAX, 0] for [minX, maxX, minY, maxY]
    int bbox[4] = {INT_MAX, 0, INT_MAX, 0};
    cl::Buffer bboxBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                          sizeof(bbox), bbox);

    kernel_bounding_box.setArg(0, inputBufferBB);
    kernel_bounding_box.setArg(1, bboxBuffer);
    kernel_bounding_box.setArg(2, static_cast<int>(outputImage.cols));
    kernel_bounding_box.setArg(3, static_cast<int>(outputImage.rows));
    uchar brightness_threshold = 20; // Brightness threshold
    kernel_bounding_box.setArg(4, brightness_threshold);

    // Execute the kernel
    queue.enqueueNDRangeKernel(kernel_bounding_box, cl::NullRange, cl::NDRange(outputImage.cols, outputImage.rows), cl::NullRange);
    queue.finish();

    // Read the bounding box result
    queue.enqueueReadBuffer(bboxBuffer, CL_TRUE, 0, sizeof(bbox), bbox);


    // Output and visualization of the bounding box and image processing parameters
    std::cout << "Global bounding box: "
              << "minX: " << bbox[0] << ", "
              << "maxX: " << bbox[1] << ", "
              << "minY: " << bbox[2] << ", "
              << "maxY: " << bbox[3] << std::endl;

    cv::putText(outputImage, "Thr. Factor: " + std::to_string(i), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));
    cv::putText(outputImage, "Half Window: " + std::to_string(halfWindow), cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255));

    cv::rectangle(outputImage, cv::Rect2i(bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2]), cv::Scalar(255), 1);


    // Display or save the result
    cv::imshow("Output Image", outputImage);
    cv::imshow("Output 2: ", outputImage2);

    
     // Handle user inputs to control the processing loop
    char c = (char) cv::waitKey(10);
    if (c == 27) break;                     // Exit on ESC
    else if (c == 32) stopper = !stopper;   // Space to pause
    else if (c == 119) halfWindow++;        // 'w' to increase halfWindow
    else if (c == 115) halfWindow--;        // 's' to decrease halfWindow

    if (!stopper) {
      i += 0.01;  // Increment the threshold factor to adjust brightness detection
    }
  }

  return 0;
}