
// OpenCL kernel for removing near-zero points in an image. 
// This operation helps in noise reduction and focusing on significant features.
__kernel void deleteNearZeroPointsKernel(
    __global uchar* image,           // Input image buffer
    __global uchar* output,          // Output image buffer
    const int width,                 // Image width
    const int height,                // Image height
    const int neighborDistance,      // Distance within which to search for zero neighbors
    const int zeroCountThreshold) {  // Threshold of zero-valued neighbors to decide on deletion
    
    // Get the current working element's x and y coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure the thread is within image boundaries
    if (x >= 0 && y >= 0 && x < width && y < height) {
        int zeroCount = 0;

        // Iterate over the neighborhood of the current pixel
        for (int dy = -neighborDistance; dy <= neighborDistance; dy++) {
            for (int dx = -neighborDistance; dx <= neighborDistance; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                // Check if the neighbor is within image boundaries
                if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                    // Increment zeroCount if a zero-valued neighbor is found
                    if (image[ny * width + nx] == 0) {
                        zeroCount++;
                    }
                }
            }
        }

        // Set the pixel value to zero if zeroCount exceeds the threshold, otherwise retain the original value
        output[y * width + x] = (zeroCount > zeroCountThreshold) ? 0 : image[y * width + x];
    }
}
// OpenCL kernel to highlight bright spots in the image based on local neighborhood analysis.
__kernel void brightSpotsKernel(
    __global uchar* image,           // Input image buffer
    __global uchar* output,          // Output image buffer
    const int width,                 // Image width
    const int height,                // Image height
    const int halfWindow,            // Half of the window size for local analysis
    const float thresholdFactor) {   // Factor to scale the pixel value for brightness detection
    
    // Get the current working element's x and y coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Process only the elements that have sufficient surrounding pixels for analysis
    if (x >= halfWindow && y >= halfWindow && x < (width - halfWindow) && y < (height - halfWindow)) {
        double sum = 0;
        int count = 0;
        int brightCount = 0;
        uchar currentPixelValue = image[y * width + x];

        // Analyze the local neighborhood of the current pixel
        for (int ly = -halfWindow; ly <= halfWindow; ++ly) {
            for (int lx = -halfWindow; lx <= halfWindow; ++lx) {
                uchar val = image[(y + ly) * width + (x + lx)];
                if (val != 0) {
                    sum += val;
                    count++;
                    if (val > currentPixelValue * thresholdFactor) {
                        brightCount++;
                    }
                }
            }
        }

        // Calculate local mean brightness
        double localMean = count > 0 ? sum / count : 0;

        // Highlight bright spots based on the comparison between current pixel value and local mean
        if (currentPixelValue > localMean * thresholdFactor && currentPixelValue != 0) {
            double brightnessScore = currentPixelValue - localMean;
            double areaFactor = brightCount / (double)(count > 0 ? count : 1);
            double score = brightnessScore * areaFactor;
            output[y * width + x] = (uchar)(255.0 * score);
        } else {
            output[y * width + x] = 0;
        }
    }
}

// OpenCL kernel to find the global bounding box of bright areas in the image.
__kernel void findGlobalBoundingBoxKernel(
    __global const uchar* image,     // Input image buffer
    __global int* bbox,              // Output bounding box [minX, maxX, minY, maxY]
    const int width,                 // Image width
    const int height,                // Image height
    const uchar brightnessThreshold) // Brightness threshold to consider for bounding box
{
    // Get the current working element's x and y coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Check only the relevant pixels within the image bounds
    if (x < width && y < height) {
        uchar pixelValue = image[y * width + x];
        
        // Update bounding box if the pixel is brighter than the threshold
        if (pixelValue > brightnessThreshold) {
            atomic_min(&bbox[0], x); // Update minX
            atomic_max(&bbox[1], x); // Update maxX
            atomic_min(&bbox[2], y); // Update minY
            atomic_max(&bbox[3], y); // Update maxY
        }
    }
}