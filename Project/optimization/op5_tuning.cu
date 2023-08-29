#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output - y
    input - input - x
    mask - convolution kernel- k
    Batch - batch_size (number of images in x) - B
    Map_out - number of output feature maps - M
    Channel - number of input feature maps - C
    Height - input height dimension - H
    Width - input width dimension - W
    K - kernel height and width (K x K)
    */

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if((h < H_out) && (w < W_out)){
        for(int c = 0; c < Channel; c++){
            acc += in_4d(b,c,h+0,w+0) * mask_4d(m,c,0,0);
            acc += in_4d(b,c,h+0,w+1) * mask_4d(m,c,0,1);
            acc += in_4d(b,c,h+0,w+2) * mask_4d(m,c,0,2);
            acc += in_4d(b,c,h+0,w+3) * mask_4d(m,c,0,3);
            acc += in_4d(b,c,h+0,w+4) * mask_4d(m,c,0,4);
            acc += in_4d(b,c,h+0,w+5) * mask_4d(m,c,0,5);
            acc += in_4d(b,c,h+0,w+6) * mask_4d(m,c,0,6);

            acc += in_4d(b,c,h+1,w+0) * mask_4d(m,c,1,0);
            acc += in_4d(b,c,h+1,w+1) * mask_4d(m,c,1,1);
            acc += in_4d(b,c,h+1,w+2) * mask_4d(m,c,1,2);
            acc += in_4d(b,c,h+1,w+3) * mask_4d(m,c,1,3);
            acc += in_4d(b,c,h+1,w+4) * mask_4d(m,c,1,4);
            acc += in_4d(b,c,h+1,w+5) * mask_4d(m,c,1,5);
            acc += in_4d(b,c,h+1,w+6) * mask_4d(m,c,1,6);

            acc += in_4d(b,c,h+2,w+0) * mask_4d(m,c,2,0);
            acc += in_4d(b,c,h+2,w+1) * mask_4d(m,c,2,1);
            acc += in_4d(b,c,h+2,w+2) * mask_4d(m,c,2,2);
            acc += in_4d(b,c,h+2,w+3) * mask_4d(m,c,2,3);
            acc += in_4d(b,c,h+2,w+4) * mask_4d(m,c,2,4);
            acc += in_4d(b,c,h+2,w+5) * mask_4d(m,c,2,5);
            acc += in_4d(b,c,h+2,w+6) * mask_4d(m,c,2,6);  

            acc += in_4d(b,c,h+3,w+0) * mask_4d(m,c,3,0);
            acc += in_4d(b,c,h+3,w+1) * mask_4d(m,c,3,1);
            acc += in_4d(b,c,h+3,w+2) * mask_4d(m,c,3,2);
            acc += in_4d(b,c,h+3,w+3) * mask_4d(m,c,3,3);
            acc += in_4d(b,c,h+3,w+4) * mask_4d(m,c,3,4);
            acc += in_4d(b,c,h+3,w+5) * mask_4d(m,c,3,5);
            acc += in_4d(b,c,h+3,w+6) * mask_4d(m,c,3,6);

            acc += in_4d(b,c,h+4,w+0) * mask_4d(m,c,4,0);
            acc += in_4d(b,c,h+4,w+1) * mask_4d(m,c,4,1);
            acc += in_4d(b,c,h+4,w+2) * mask_4d(m,c,4,2);
            acc += in_4d(b,c,h+4,w+3) * mask_4d(m,c,4,3);
            acc += in_4d(b,c,h+4,w+4) * mask_4d(m,c,4,4);
            acc += in_4d(b,c,h+4,w+5) * mask_4d(m,c,4,5);
            acc += in_4d(b,c,h+4,w+6) * mask_4d(m,c,4,6);

            acc += in_4d(b,c,h+5,w+0) * mask_4d(m,c,5,0);
            acc += in_4d(b,c,h+5,w+1) * mask_4d(m,c,5,1);
            acc += in_4d(b,c,h+5,w+2) * mask_4d(m,c,5,2);
            acc += in_4d(b,c,h+5,w+3) * mask_4d(m,c,5,3);
            acc += in_4d(b,c,h+5,w+4) * mask_4d(m,c,5,4);
            acc += in_4d(b,c,h+5,w+5) * mask_4d(m,c,5,5);
            acc += in_4d(b,c,h+5,w+6) * mask_4d(m,c,5,6);

            acc += in_4d(b,c,h+6,w+0) * mask_4d(m,c,6,0);
            acc += in_4d(b,c,h+6,w+1) * mask_4d(m,c,6,1);
            acc += in_4d(b,c,h+6,w+2) * mask_4d(m,c,6,2);
            acc += in_4d(b,c,h+6,w+3) * mask_4d(m,c,6,3);
            acc += in_4d(b,c,h+6,w+4) * mask_4d(m,c,6,4);
            acc += in_4d(b,c,h+6,w+5) * mask_4d(m,c,6,5);
            acc += in_4d(b,c,h+6,w+6) * mask_4d(m,c,6,6);
        }
        out_4d(b, m, h, w) = acc;
    }
    
#undef out_4d
#undef in_4d
#undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;


    dim3 Dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(Map_out, Y, Batch);

    conv_forward_kernel<<<DimGrid, Dimblock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
