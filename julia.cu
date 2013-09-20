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
    if (a.norm() > 200)
      return 0;
  }
  return 1;
}

__global__ void d_kernel(int DIM, unsigned char *ptr)
{
  int x = blockIdx.x;
  int y = blockIdx.y;

  int offset      = x + y * gridDim.x;
  int julia_value = k_julia(DIM, x, y);

  ptr[offset * 3 + 0] = 255 * julia_value;
  ptr[offset * 3 + 1] = 255 * julia_value;
  ptr[offset * 3 + 2] = 255 * julia_value;
}

void run_dat_kernel(int DIM, unsigned char *host_buffer)
{
  unsigned char *dev_buffer;
  cudaMalloc((void**)&dev_buffer, 3*DIM*DIM*sizeof(unsigned char));

  dim3 grid(DIM, DIM);
  d_kernel<<<grid,1>>>(DIM, dev_buffer);

  cudaMemcpy(host_buffer, dev_buffer, 3*DIM*DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(dev_buffer);
}
