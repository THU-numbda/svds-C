#The settings of system, core and compiler with the path and version of oneapi and mkl
CC = gcc

OPENBLAS_PATH = /home/xufeng/test/openblas/OpenBLAS_openmp/

INCLUDE = $(OPENBLAS_PATH)include
COMPILER_FLAGS = -O3 -m64 -I$(INCLUDE)

OPENBLAS_LIB = $(OPENBLAS_PATH)lib/libopenblas.a
LINK_FLAGS = -Wl,--start-group $(OPENBLAS_LIB) -Wl,--end-group -lgfortran -lpthread -ldl -lm -fopenmp

DEMO = svdstest
SOURCE = svdstest.c svds.c matrix_funcs.c

$(DEMO): $(SOURCE)
	$(CC) $(SOURCE) $(COMPILER_FLAGS) $(LINK_FLAGS) -o $(DEMO)

clean:
	rm -rf $(DEMO)
