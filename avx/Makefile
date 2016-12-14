CFLAGS=-wd588 -ansi-alias
OPT=-O2 -xavx -g
REPORT=-opt-report=5

N=1000
DT=0.001f
STEPS=1000

PARAMS=-DN=$(N) -DDT=$(DT) -DSTEPS=$(STEPS)

INPUT=../examples/example_input.dat
OUTPUT=../examples/example_output.dat

all:
	icpc $(CFLAGS) $(OPT) $(REPORT) $(PARAMS) -c nbody.cpp
	icpc $(CFLAGS) $(OPT) $(PARAMS) -S -fsource-asm -c nbody.cpp
	icpc $(CFLAGS) $(PARAMS) nbody.o main.cpp -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody $(INPUT) $(OUTPUT)
