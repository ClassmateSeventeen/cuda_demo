#include <stdio.h>
#include "../tools/common.cuh"


__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}


int main(void)
{
    //1、设置GPU设备
    setGPU();

    //2、分配主机内存和设备内存，并初始化
    int nx = 16;
    int ny = 8;                               //设置元素数量
    int nxy = nx * ny;
    size_t stBytesCount = nxy * sizeof(int);   //字节数

    //（1） 分配主机内存，并初始化
    int *ipHost_A, *ipHost_B, *ipHost_C;
    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);
    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        //主机内存初始化为0
        for (int i = 0; i < nxy; i++)
        {
            ipHost_A[i] = i;
            ipHost_B[i] = i + 1;
        }
        memset(ipHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }   

    //（2） 分配设备内存，并初始化
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;
    cudaMalloc((int**)&ipDevice_A, stBytesCount);
    cudaMalloc((int**)&ipDevice_B, stBytesCount);
    cudaMalloc((int**)&ipDevice_C, stBytesCount);
    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        //设备内存初始化为0
        cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice);
    }
    else
    {
        printf("Fail to allocate device memory!\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(-1);
    }

    //5、调用核函数在设备中进行计算
    dim3 block(4, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);

    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);    //调用核函数
    cudaDeviceSynchronize();    //同步

    //6、将计算得到的数据从设备传回给主机
    cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)    //打印
    {
        printf("idx=%d\tmatrix_A:%d\tmatrix_B:%d\tresult=%d\n", i+1, ipHost_A[i], ipHost_B[i], ipHost_C[i]);
    }

    //7、释放主机与设备内存
    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);
    cudaFree(ipDevice_A);
    cudaFree(ipDevice_B);
    cudaFree(ipDevice_C);

    cudaDeviceReset();
    return 0;
}

