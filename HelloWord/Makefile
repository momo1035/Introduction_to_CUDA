NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: helloworld

helloworld: helloworld.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f helloworld