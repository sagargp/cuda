CC=g++-4.7
LIBS=-L/usr/local/cuda/lib -lcuda -lcudart `sdl-config --libs`
INCLUDES=`sdl-config --cflags`

all: main

main: julia.o sdl_manager.o
	${CC} -std=c++11 -o main main.cpp julia.o sdl_manager.o ${LIBS} ${INCLUDES}

julia.o:
	nvcc -c -m64 julia.cu

sdl_manager.o:
	${CC} -g -std=c++11 -c sdl_manager.cpp ${INCLUDES}

clean:
	rm -rf sdltest main
	rm -rf *.dSYM *.o
