CC=g++-4.7
LIBS=-L/usr/local/cuda/lib -lcuda -lcudart `sdl-config --libs`
INCLUDES=`sdl-config --cflags` -I../include

all: main

main: julia.o
	${CC} -std=c++11 -o main main.cpp julia.o ../lib/sdl_manager.o ${LIBS} ${INCLUDES}

julia.o:
	nvcc -c -m64 julia.cu

clean:
	rm -rf main
	rm -rf *.o
