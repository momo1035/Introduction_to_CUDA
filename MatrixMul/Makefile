NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-deprecated-declarations"

all: MatrixMul MatrixMulTile

MatrixMul: MatrixMul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

MatrixMulTile: MatrixMulTile.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f MatrixMul MatrixMulTile
