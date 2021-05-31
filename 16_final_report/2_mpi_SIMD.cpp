#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <immintrin.h>
using namespace std;
//typedef vector<vector<float>> matrix;


int main(int argc, char **argv) {

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 256;
  vector<float> A(N * N);
  vector<float> B(N * N);
  vector<float> C(N * N, 0);
  vector<float> subA(N * N / size);
  vector<float> subB(N * N / size);
  vector<float> subC(N * N / size, 0);
  vector<float> recv(N * N / size);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N * i + j] = drand48();
      B[N * i + j] = drand48();
    }
  }

  int offset = N / size * rank;
  for (int i = 0; i < N / size; i++)
      for (int j = 0; j < N; j++)
        subA[N * i + j] = A[N * (i + offset) + j];
  for (int i = 0; i < N; i++)
      for (int j = 0; j < N / size; j++)
        subB[N / size * i + j] = B[N * i + j + offset];
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;  
  
  const int m = N, n = N, k = N;
  const int kc = 128;
  const int nc = 16;
  const int mc = 64;
  const int nr = 16;
  const int mr = 8;

  double comp_time = 0, comm_time = 0;
  for (int irank = 0; irank < size; irank++){
    auto tic = chrono::steady_clock::now();
    offset = N / size * ((rank + irank) % size);

  for (int jc=0; jc<N/size; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      float Bc[kc*nc];
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p*nc+j] = subB[(N/size)*(p+pc)+j+jc];//???
        }
      }
      for (int ic=0; ic<N/size; ic+=mc) {
        float Ac[mc*kc],Cc[mc*nc];
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = subA[(N/size)*(i+ic)+p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                __m256 Avec = _mm256_broadcast_ss(Ac+i*kc+kr);
                for (int j=jr; j<jr+nr; j+=8) {
                  __m256 Bvec = _mm256_load_ps(Bc+kr*nc+j);
                  __m256 Cvec = _mm256_load_ps(Cc+i*nc+j);
                  Cvec = _mm256_fmadd_ps(Avec, Bvec, Cvec);
                  _mm256_store_ps(Cc+i*nc+j, Cvec); //store result
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            subC[N*(i+ic)+j+jc+offset] += Cc[i*nc+j];
          }
        }
      }
    }
  }
  auto toc = chrono::steady_clock::now();
  comp_time += chrono::duration<double>(toc - tic).count();

  MPI_Request request[2];
  MPI_Isend(&subB[0], N * N / size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
  MPI_Irecv(&recv[0], N * N / size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
  MPI_Waitall(2, request, MPI_STATUS_IGNORE);
  for (int i = 0; i < N * N / size; i++) 
    subB[i] = recv[i];

  tic = chrono::steady_clock::now();
  comm_time += chrono::duration<double>(tic - toc).count();
}

MPI_Allgather(&subC[0], N * N / size, MPI_FLOAT, &C[0], N * N / size, MPI_FLOAT, MPI_COMM_WORLD);
  
//printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);

  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N * i + j] -= A[N * i + k] * B[N * k + j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N * i + j]);

  if (rank == 0)
    {
        double time = comp_time + comm_time;
        printf("N    : %d\n", N);
        printf("comp : %lf s\n", comp_time);
        printf("comm : %lf s\n", comm_time);
        printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
        printf("error: %lf\n", err / N / N);
    }

  MPI_Finalize();
}
