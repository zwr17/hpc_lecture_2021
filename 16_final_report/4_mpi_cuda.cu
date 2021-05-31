#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
using namespace std;

__global__ void matrix(float *a,float *b,float *c,int N, int offset,int size){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N/size ){
    for (int i=0; i<N/size; i++)
      for (int k=0; k<N; k++)
        c[N*i+j+offset] += a[N*i+k] * b[N/size*k+j];
  }
}

int main(int argc, char** argv) {

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int gpusize, gpurank;
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  cudaGetDevice(&gpurank);

  const int N = 256; //change N here example.256,1024,2048
  const int block =256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);

  float *subA, *subB, *subC, *recv;
  int sub_size = N * N /size;
  int size_cuda = N * N /size * sizeof(float);
  subA = new float [sub_size];
  subB = new float [sub_size];
  subC = new float [sub_size];
  recv = new float [sub_size];

  float *a, *b, *c;
  cudaMalloc((void **) &a, size_cuda);
  cudaMalloc((void **) &b, size_cuda);
  cudaMalloc((void **) &c, size_cuda);

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }

  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  cudaMemcpy(a,subA,size_cuda,cudaMemcpyHostToDevice);
  cudaMemcpy(b,subB,size_cuda,cudaMemcpyHostToDevice);

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
    matrix<<<(N/size+block-1)/block,block>>>(a,b,c,N,offset,size);
    cudaDeviceSynchronize();
    
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();

    MPI_Request request[2];
    MPI_Isend(&subB[0], sub_size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], sub_size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    for (int i=0; i<sub_size; i++)
      subB[i] = recv[i];

    cudaMemcpy(b, subB, size_cuda, cudaMemcpyHostToDevice);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];

  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);

  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }

  MPI_Finalize();

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}