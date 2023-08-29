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

    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // const int Width_grid = ceil(1.0 * Width_out / TILE_WIDTH);
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_grid;

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
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                        acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
        }
        out_4d(b, m, h, w) = acc;
    }
    
#undef out_4d
#undef in_4d
#undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
#define STREAM_NUM 10
    // Allocate memory and copy over the relevant data structures to the GPU

// const float *host_y, host_output
// const float *host_x, host_input
// const float *host_k, host_mask
// float **device_y_ptr, device_output_ptr
// float **device_x_ptr, device_input_ptr
// float **device_k_ptr, device_mask_ptr
// const int B, Batch
// const int M, Map_out
// const int C, Channel
// const int H, Height
// const int W, Width
// const int K

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    float* host_y_temp = (float*)host_output;
    int x_batch_size = (Batch * Channel * Height * Width) / STREAM_NUM;
    int y_batch_size = (Batch * Map_out * H_out * W_out) / STREAM_NUM;

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;

    dim3 Dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(Map_out, Y, Batch/STREAM_NUM);

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamCreate(&stream[i]);

    cudaMemcpyAsync(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for (int i = 0; i < STREAM_NUM; i++){
        int x_offset = x_batch_size * i;
        int y_offset = y_batch_size * i;
        cudaMemcpyAsync((*device_input_ptr) + x_offset, host_input + x_offset, x_batch_size * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<DimGrid, Dimblock, 0, stream[i]>>>((*device_output_ptr) + y_offset, (*device_input_ptr) + x_offset, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync(host_y_temp + y_offset, (*device_output_ptr) + y_offset, y_batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamDestroy(stream[i]);

    // Free device memory
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);

#undef STREAM_NUM

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
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
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
