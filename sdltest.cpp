#include <iostream>
#include <complex>
#include "sdl_manager.h"

const int DIM = 1000;

int julia(int x, int y)
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

void kernel(unsigned char *ptr)
{
  for (int y = 0; y < DIM; y++)
  {
    for (int x = 0; x < DIM; x++)
    {
      int offset = x + y * DIM;
      int julia_value = julia(x, y);

      ptr[offset * 3 + 0] = 255 * julia_value;
      ptr[offset * 3 + 1] = 255 * julia_value;
      ptr[offset * 3 + 2] = 255 * julia_value;
    }
  }
}

int main(int argc, char* argv[])
{
  SDL_Manager manager(DIM);

  manager.init();

  unsigned char *buffer = manager.get_draw_buffer();

  kernel(buffer);

  manager.redraw();

  manager.wait_for_keypress();
  return 0;
}
