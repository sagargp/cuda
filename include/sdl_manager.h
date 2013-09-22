#ifndef SDL_MANAGER_H
#define SDL_MANAGER_H

#include <SDL/SDL.h>

class SDL_Manager
{
  private:
    SDL_Surface *m_screen;

    int m_dim;

    unsigned char *m_buffer_rptr;

    void set_pixel(int x, int y, Uint8 r, Uint8 g, Uint8 b);

  public:
    SDL_Manager(int dim);

    ~SDL_Manager();

    bool init();

    void redraw();

    void wait_for_keypress();

    unsigned char *get_draw_buffer();
};

#endif
