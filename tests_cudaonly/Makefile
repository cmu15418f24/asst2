# Makefile for compiling and running cuda_tests.cu

CUDA_TEST_EXECUTABLE := cudaTest

CU_FILES   := cuda_tests.cu

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

CUDA_TEST_OBJ=$(OBJDIR)/cuda_tests.o

.PHONY: dirs clean all cudaTest

# Default target is to build the CUDA test executable
all: $(CUDA_TEST_EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(CUDA_TEST_EXECUTABLE)

# Rule to build cuda_tests.cu executable
$(CUDA_TEST_EXECUTABLE): dirs $(CUDA_TEST_OBJ)
		$(NVCC) $(CUDA_TEST_OBJ) $(NVCCFLAGS) -o $(CUDA_TEST_EXECUTABLE) $(LDFLAGS)

# Rule to compile cuda_tests.cu into an object
$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
