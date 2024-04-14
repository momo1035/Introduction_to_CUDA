NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: Stencil

Stencil: Stencil.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f Stencil