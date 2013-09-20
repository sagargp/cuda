all: utils
	clang++ sdl_manager.o sdltest.cpp -o sdltest `sdl-config --libs`

utils:
	clang++ -g -std=c++11 -c sdl_manager.cpp `sdl-config --cflags`

clean:
	rm -rf sdltest sdltest.dSYM sdl_manager.o
