# hpc_lecture
20M38222 ZHENG Wenru

# file

0_mpi.cpp

0_mpi_result.txt

1_mpi_openmpi.cpp

1_mpi_openmpi_result.txt

2_mpi_SIMD.cpp

2_mpi_SIMD_result.txt

3_mpi_openmpi_SIMD.cpp

3_mpi_openmpi_SIMD_result.txt

4_mpi_cuda.cu

N＝256,1024,2048の結果は各txtファイルに書いています

3_mpi_openmpi_SIMD.cppの中で、

N＝256の時、kc=128,nc=16,mc=64,nr=16,mr=8

N＝1024,2048の時,kc=256,nc=32,mc=128,nr=32,mr=16

4_mpi_cuda.cuに対しては、実行はできますが、結果がおかしいですが、一応載せました

# sh for 0&1&2&3

qrsh -g tga-hpc-lecture -l f_node=1 -l h_rt=1:00:00

module load intel-mpi gcc

mpicxx 3_mpi_openmpi_SIMD.cpp -fopenmp -fopt-info-vec-optimized -march=native -O3

mpirun -np 4 ./a.out

# sh for 4_mpi_cuda.cu

module load cuda/11.2.146 openmpi gcc

nvcc mpi_cuda.cu -lmpi -std=c++11

mpirun -np 4 ./a.out



