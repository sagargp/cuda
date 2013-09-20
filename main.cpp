#include <chrono>
#include <iostream>
#include <complex>
#include "sdl_manager.h"

extern void run_dat_kernel(int DIM, unsigned char *host_buffer);

int h_julia(int DIM, int x, int y)
{
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);

  std::complex<float> c(-0.8, 0.156);
  std::complex<float> a(jx, jy);

  int i = 0;
  for (i = 0; i < 200; i++)
  {
    a = a * a + c;
    if (std::norm(a) > 1000)
      return 0;
  }
  return 1;
}

void h_kernel(int DIM, unsigned char *ptr)
{
  for (int y = 0; y < DIM; y++)
  {
    for (int x = 0; x < DIM; x++)
    {
      int offset = x + y * DIM;
      int julia_value = h_julia(DIM, x, y);

      ptr[offset * 3 + 0] = 255 * julia_value;
      ptr[offset * 3 + 1] = 255 * julia_value;
      ptr[offset * 3 + 2] = 255 * julia_value;
    }
  }
}

int main(int argv, char **argc)
{
  const int DIM = 1000;

  std::cout << "Initializing SDL" << std::endl;
  SDL_Manager manager(DIM);
  manager.init();

  unsigned char *buffer = manager.get_draw_buffer();

  std::cout << "Computing Julia fractal on the CPU" << std::endl;
  auto start = std::chrono::steady_clock::now();
  {
    h_kernel(DIM, buffer);
    manager.redraw();
    manager.wait_for_keypress();
  }
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  std::cout << "Took " << duration.count() << "ms." << std::endl;

  std::cout << "Computing Julia fractal on the GPU" << std::endl;
  start = std::chrono::steady_clock::now();
  {
    run_dat_kernel(DIM, buffer);
    manager.redraw();
    manager.wait_for_keypress();
  }
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  std::cout << "Took " << duration.count() << "ms." << std::endl;

  return 0;
}
