NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: SumReduc

SumReduc: SumReduc.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f SumReduc