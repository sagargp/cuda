#include <iostream>

class cuComplex
{
  private:
    float r;
    float i;

  public:
    __device__ cuComplex(float a, float b) : r(a), i(b) { }

    __device__ float norm(void)
    {
      return r*r + i*i;
    }

    __device__ cuComplex operator*(const cuComplex &a)
    {
      return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ cuComplex operator+(const cuComplex &a)
    {
      return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int k_julia(int DIM, int x, int y)
{
  const float scale = 1.5;

  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);

  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);

  int i = 0;
  for (i = 0; i < 200; i++)
  {
    a = a * a + c;
    if (a.norm() > 1000)
      return 0;
  }
  return 1;
}

__global__ void d_kernel(int DIM, unsigned char *ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int offset = x + y * blockDim.x * gridDim.x;
  int julia_value = k_julia(DIM, x, y);

  ptr[offset * 3 + 0] = 255 * julia_value;
  ptr[offset * 3 + 1] = 0;
  ptr[offset * 3 + 2] = 0;
}

void run_dat_kernel(int DIM, unsigned char *host_buffer)
{
  unsigned char *dev_buffer;
  cudaMalloc((void**)&dev_buffer, 3*DIM*DIM*sizeof(unsigned char));

  int N = 16;
  dim3 blocks(DIM/N, DIM/N);
  dim3 threads(N, N);
  d_kernel<<<blocks,threads>>>(DIM, dev_buffer);

  cudaMemcpy(host_buffer, dev_buffer, 3*DIM*DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaFree(dev_buffer);
}
