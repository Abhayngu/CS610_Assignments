// compile : nvcc -std=c++11 22111001-prob4.cu -o prob4
// Execute : ./prob4

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <cstdlib>

const uint64_t N = (64);
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

// TODO: Edit the function definition as required
__global__ void kernel1(double* d_A, double* d_B) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if((i>=1) && (i<N-1) && (j>=1) && (j<N-1) && (k>=1) && (k<N-1)){
    d_B[i*N*N+j*N+k]=0.8 * (d_A[(i-1) * N * N + j * N + k]+d_A[(i+1) * N + j * N + k] + d_A[i * N * N+ (j-1) * N + k] +d_A[i * N * N + (j+1) * N + k] + d_A[i * N * N + j * N + k - 1] + d_A[i * N * N + j * N + k + 1]);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel2(double* d_A, double* d_B) {
  const int width = 8;
  
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;


  if((i>=1) && (i<N-1) && (j>=1) && (j<N-1)){
    int k;
    for(k = 1; k<(N-1); k+=width){

      

      for(int x = k; x<k+width && x < (N - 1); x++){
        d_B[i * N * N + j * N + x]= 0.8 * (d_A[(i-1) * N * N + j * N + x]+d_A[(i+1) * N + j * N + x] + d_A[i * N * N+ (j-1) * N + x] +d_A[i * N * N + (j+1) * N + x] + d_A[i * N * N + j * N + x - 1] + d_A[i * N * N + j * N + x + 1]);
      }
    }
    k-=width;
    if(k <= 0){
      k = 1;
    }
    while( k < N-1){
      d_B[i*N*N+j*N+k]= 0.8 * (d_A[(i-1) * N * N + j * N + k]+d_A[(i+1) * N + j * N + k] + d_A[i * N * N+ (j-1) * N + k] +d_A[i * N * N + (j+1) * N + k] + d_A[i * N * N + j * N + k - 1] + d_A[i * N * N + j * N + k + 1]);
      k++;
    }
  }
}

// TODO: Edit the function definition as required
__host__ void stencil(double* in, double* out) {
  for(int i = 1; i<N-1; i++){
    for(int j = 1; j<N-1; j++){
      for(int k = 1; k<N-1; k++){
        out[i*N*N+j*N+k]=0.8 * (in[(i-1)*N*N+j*N+k]+in[ (i+1)*N+j*N+k] + in [i*N*N+(j -1)*N+ k] +in [ i*N*N+ (j +1)*N+ k ] + in [ i*N*N+ j*N+ k -1] + in [ i*N*N+ j*N + k +1]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt, const uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;
  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;
  double* serial_in = new double[SIZE];
  double* serial_out = new double[SIZE];
  double *h_A = new double[SIZE], *h_B = new double[SIZE];
  double *temp = new double[SIZE];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        serial_in[i*N*N+j*N+k] = rand()%8;
        serial_out[i*N*N+j*N+k] = 0;
        h_A[i*N*N+j*N+k] =  serial_in[i*N*N+j*N+k];
        h_B[i*N*N+j*N+k] = 0;
        temp[i*N*N+j*N+k] = 0;
      }
    }
  }
  double clkbegin = rtclock();
  stencil(serial_in, serial_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  double *d_A, *d_B, *d_C;
  float k1_time;
  status = cudaMalloc(&d_A, SIZE * sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_B, SIZE * sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_C, SIZE * sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  dim3 grid((1<<3), (1<<3), (1<<3));  
  dim3 block((1<<3), (1<<3), (1<<3));
  kernel1<<<grid, block>>>(d_A, d_B);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_time, start, end);
  cudaMemcpy(h_B, d_B, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
  check_result(serial_out, h_B, N);
  std::cout << "Kernel 1 time (ms): " << k1_time << "\n";

  double* d_B2;
  float k2_time;
  status = cudaMalloc(&d_B2, SIZE * sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_B2, h_B, SIZE * sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  cudaEventRecord(start);
  dim3 grid2((1<<5), (1<<5), 1);  
  dim3 block2((1<<4), (1<<4), 1);
  kernel2<<<grid2, block2>>>(d_A, d_B2);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k2_time, start, end);
  cudaMemcpy(temp, d_B2, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
  check_result(serial_out, temp, N);
  std::cout << "Kernel 2 time (ms): " << k2_time << "\n";

  delete[] serial_in;
  delete[] serial_out;
  delete[] h_A;
  delete[] h_B;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // TODO: Free memory

  return EXIT_SUCCESS;
}
