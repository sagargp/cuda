#include "sdl_manager.h"

SDL_Manager::SDL_Manager(int dim) : 
  m_dim(dim)
{
  m_buffer_rptr = (unsigned char*) malloc(3 * m_dim * m_dim * sizeof(unsigned char));
}

SDL_Manager::~SDL_Manager()
{
  SDL_Quit();
}

bool SDL_Manager::init()
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
    return false;
  
  if ( !(m_screen = SDL_SetVideoMode(m_dim, m_dim, 32, SDL_HWSURFACE)) )
  {
    SDL_Quit();
    return false;
  }

  return true;
}

void SDL_Manager::redraw()
{
  if (SDL_MUSTLOCK(m_screen))
    if (SDL_LockSurface(m_screen) < 0)
      return;

  for (int y = 0; y < m_dim; y++)
  {
    for (int x = 0; x < m_dim; x++)
    {
      int offset = x + y * m_dim;

      unsigned char r = m_buffer_rptr[offset * 3 + 0];
      unsigned char g = m_buffer_rptr[offset * 3 + 1];
      unsigned char b = m_buffer_rptr[offset * 3 + 2];

      set_pixel(x, y, r, g, b);
    }
  }

  if(SDL_MUSTLOCK(m_screen))
    SDL_UnlockSurface(m_screen);

  SDL_Flip(m_screen);
}

unsigned char* SDL_Manager::get_draw_buffer()
{
  return m_buffer_rptr;
}

void SDL_Manager::wait_for_keypress()
{
  SDL_Event event;
  int keypress = 0;
  while(!keypress) 
  {
    while(SDL_PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL_QUIT:
          keypress = 1;
          break;
        case SDL_KEYDOWN:
          keypress = 1;
          break;
      }
    }
  }
}

void SDL_Manager::set_pixel(int x, int y, Uint8 r, Uint8 g, Uint8 b)
{
  Uint8 *pixel = (Uint8*)m_screen->pixels;
  pixel += (y * m_screen->pitch) + (x*sizeof(Uint32));

  Uint32 color = SDL_MapRGB(m_screen->format, r, g, b);
  *((Uint32*)pixel) = color;
}
