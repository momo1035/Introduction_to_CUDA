NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: VectorAdd

VectorAdd: VectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f VectorAdd