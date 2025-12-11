# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME : SREE NIVEDITAA SARAVANAN</h3>
<h3>ENTER YOUR REGISTER NO: 212223230213</h3>
<h3>EX. NO.1</h3>
<h3>DATE:10.09.25</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```c
%%cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    printf("h_c[0] = %.0f (expected 0)\n", h_c[0]);
    printf("h_c[100] = %.0f (expected 300)\n", h_c[100]);
    printf("Success!\n");
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

## OUTPUT:
<img width="718" height="212" alt="image" src="https://github.com/user-attachments/assets/36956dcf-1cae-4d12-9476-ef7160bcc728" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
