__kernel void brightSpotsKernel(__global uchar* image, __global uchar* output, 
                                const int width, const int height, 
                                const int halfWindow, const float thresholdFactor) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= halfWindow && y >= halfWindow && x < (width - halfWindow) && y < (height - halfWindow)) {
        double sum = 0;
        int count = 0;
        int brightCount = 0;
        uchar currentPixelValue = image[y * width + x];

        // Compute the sum, count, and brightCount for the local area excluding zeros
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

        double localMean = count > 0 ? sum / count : 0;

        // Determine scoring based on brightness and local area brightness
        if (currentPixelValue > localMean * thresholdFactor && currentPixelValue != 0) {
            double brightnessScore = currentPixelValue - localMean;
            double areaFactor = brightCount / (double)(count > 0 ? count : 1);
            
            // Combine the brightness score and area factor to determine the final score
            // Here we assume brightnessScore and areaFactor are normalized to [0, 1]
            double score = brightnessScore * areaFactor;
            output[y * width + x] = (uchar)(255.0 * score);
        } else {
            output[y * width + x] = 0;
        }
    }
}

__kernel void labelingKernel(__global const uchar* image, __global int* labels, 
                             const int width, const int height, const uchar threshold) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x;

    if (x < width && y < height) {
        uchar pixelValue = image[index];
        if (pixelValue > threshold) {
            // Simple labeling: label is the pixel's index
            labels[index] = index;
        } else {
            labels[index] = -1;  // Mark as background
        }
    }
}


__kernel void scoringKernel(__global const int* labels, __global uchar* output, 
                            const int width, const int height, __global const int* sizes, 
                            const int maxSize) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x;

    if (x < width && y < height && labels[index] != -1) {
        int label = labels[index];
        int size = sizes[label]; // Size of the component
        
        // Normalize the score based on the size of the component
        uchar score = (uchar)(255.0f * size / maxSize);
        output[index] = score;
    } else {
        output[index] = 0; // Background or non-bright spot
    }
}