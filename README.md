# ECE408 Spring 2023

Welcome to my ECE408 repo! The project folder contains the design and implementation of the forward-pass of a convolutional layer using CUDA. I Used a series of methods such as
tiled shared memory, streaming, tuning, etc. to optimize and accelerate the forward propagation. I've made the following optimization to the forward-pass:
- Tiled shared memory convolution; 
- Weight matrix (kernel values) in constant memory; 
- Tuning with restrict and loop unrolling; 
- Using Streams to overlap computation with data transfer

If you are not familiar with RAI/RAID, please checkout the [Introduction to RAID](https://drive.google.com/file/d/1t6-uPgbCxi5zx0FKKG15nanXt7NX8zCP/view?usp=sharing) PDF. This document explains the RAI/RAID organization and options.

Download the RAI binary for your platform below.

[Linux and Darwin RAI binaries](https://drive.google.com/drive/folders/1Pp84x3So9OEHUwRHQVZcRP441wRsO-UV)

### Windows

****
On Windows, you'll need to install WSL and a virtual linux OS. Several Linux versions are available
through the Microsoft Store.
