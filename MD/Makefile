NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations -fopenmp"

all: MD

MD: MD.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lgomp

clean:
	rm -f MD