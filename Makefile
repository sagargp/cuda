CC=g++-4.7
LIBS=`sdl-config --libs`
INCLUDES=`sdl-config --cflags`

all: sdl_manager.o

sdl_manager.o:
	${CC} -g -std=c++11 -c include/sdl_manager.cpp ${INCLUDES} -o lib/sdl_manager.o

clean:
	rm -rf lib/*.o
