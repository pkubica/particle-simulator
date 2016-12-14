N=8192
DT=0.001f
STEPS=1000
THREADS_PER_BLOCK=128

INPUT=../examples/example_input.dat
OUTPUT=../examples/example_output.dat

all:
	nvcc -Xptxas="-v" nbody.cu main.cu -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody $(N) $(DT) $(STEPS) $(THREADS_PER_BLOCK) $(INPUT) $(OUTPUT)

profile:
	nvprof \
		--devices 0 \
		--metrics all \
		./nbody $(N) $(DT) 1 $(THREADS_PER_BLOCK) $(INPUT) $(OUTPUT)
