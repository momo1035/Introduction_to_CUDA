NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: SpMV 

SpMV: SpMV.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<


clean:
	rm -f SpMV 
