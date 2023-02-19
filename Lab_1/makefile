CC=g++
CFLAGS=-O2 -Wall -g
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
VFLAGS=-O2 -c -g -Wall -mavx -ftree-vectorize #-funroll-loops#-fopt-info-vec-optimized  
NVFLAGS=-O2 -fopt-info-vec-optimized -g -c -Wall 

run_simple: simple
	./simple

simple: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $@

vec: vecobj 
	$(CC) $(OBJECTS) -o $@
	objdump -d -M amd64 -S main.o > vecDump.txt

novec: novecobj
	$(CC) $(OBJECTS) -o $@
	objdump -d -M amd64 -S main.o > novecDump.txt

vecobj: $(SOURCES)
	$(CC) $(SOURCES) $(VFLAGS) 

novecobj: $(SOURCES)
	$(CC) $(SOURCES) $(NVFLAGS) 

clear:
	rm main.o vec vecDump.txt novec novecDump.txt simple