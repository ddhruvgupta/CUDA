
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
//#include "opencv2/cudawarping.hpp"
#include <stdio.h>
#include <helper_cuda.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#define BLUR_SIZE 2

using namespace cv;
using namespace std;

cudaError_t blurWithCuda(uchar* in, uchar* out, int w, int h);
//void blurKernel(uchar * in, uchar * out, int w, int h);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*
__global__ void blurKernel(uchar* in, uchar* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);

    }
}
*/

__global__ 
void blurKernel(uchar* in, uchar* out, int w, int h) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < h*BLUR_SIZE && j < w* BLUR_SIZE)
    {
        int iIn = (i / BLUR_SIZE);
        int jIn = (j / BLUR_SIZE);
        out[i * w*BLUR_SIZE + j] = in[iIn * w + jIn];
    }
}

int main(int argc, char* argv[])
{
    Mat image = imread("lena512.bmp", IMREAD_GRAYSCALE);   // Read the file
    //Mat image = imread("lena512.bmp");
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);
    //waitKey(0);


    // import image

    int rows = image.rows;
    int cols = image.cols;

    //float scale = 5.0f;
    uchar* in = image.data;


    uchar * out = (uchar *) malloc(rows * cols * BLUR_SIZE * BLUR_SIZE+1);


    // Add vectors in parallel.

    cudaError_t cudaStatus = blurWithCuda(in, out, cols, rows);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    Mat out_mat  = Mat(rows* BLUR_SIZE, cols* BLUR_SIZE, CV_8UC1, out);
    
    namedWindow("Display window2", WINDOW_AUTOSIZE);
    imshow("Display window2", out_mat);
    waitKey(0);
    /*
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    */
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.



cudaError_t blurWithCuda(uchar* in, uchar* out, int w, int h)
{
    uchar * dPin;
    uchar * dPout;
    
    cudaError_t cudaStatus;

    //dim3 dimGrid(ceil(h / 16.0), ceil(w / 16.0), 1);
    //dim3 dimBlock(16, 16, 1);
    dim3 dimBlock(16, 16);
    dim3 dimGrid( ((w* BLUR_SIZE)/16)+1 , ((h * BLUR_SIZE)/16) +1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dPin, w * h * sizeof(uchar));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dPout, BLUR_SIZE * BLUR_SIZE *w * h * sizeof(uchar));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dPin, in, w * h * sizeof(uchar), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dPout, out, w * h * sizeof(uchar), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    blurKernel<<< dimGrid, dimBlock >>> (dPin, dPout, w, h);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dPout, w * h * BLUR_SIZE * BLUR_SIZE* sizeof(uchar), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dPin);
    cudaFree(dPout);
    
    
    return cudaStatus;
}
