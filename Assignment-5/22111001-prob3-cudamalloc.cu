// compile : nvcc -std=c++11 22111001-prob3-cudamalloc.cu -o prob3-1
// Execute : ./prob3-1

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);

using std::cerr;
using std::cout;
using std::endl;

__global__ void opt_matmul_prob2(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
  const int width = 16;
  __shared__ uint64_t left[width][width];
  __shared__ uint64_t right[width][width];

  uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if((i < N) && (j < N)){
    uint64_t x = 0;
    for(int k = 0; k < (N/width); k++){
      left[threadIdx.y][threadIdx.x] = d_A[i*N + k*width + threadIdx.x];
      right[threadIdx.y][threadIdx.x] = d_B[(k * width + threadIdx.y) * N + j];
      __syncthreads();
      for(int z = 0; z<width; z++){
        x += left[threadIdx.y][z] * right[z][threadIdx.x];
      __syncthreads();

      }
    }
    d_C[i*N + j] = x;
  }
}

__host__ void cpumatMul(const uint64_t* h_A, const uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      float sum = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(const uint64_t* w_ref, const uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
  } else {
    cout << "No differences found between base and test versions\n";
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
  const uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec\n" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  uint64_t *d_A, *d_B, *d_C1;
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_B, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  float k1_htd_time, k1_kernel_time, k1_dth_time;

  // Host To Device
  cudaEventRecord(start);
  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_htd_time, start, end);

  // Kernel
  cudaEventRecord(start);
  dim3 grid((1<<6), (1<<6), 1);
  dim3 block((1<<4), (1<<4), 1);
  opt_matmul_prob2<<<grid, block>>>(d_A, d_B, d_C1);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_kernel_time, start, end);
  
  // Device to Host
  cudaEventRecord(start);
  cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_dth_time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  
  // Checking result
  check_result(h_cpu_C, h_gpu1_C);
  cout << "Timings for kernel 1 :\n";
  cout << "Host To Device Copy(ms) : " << k1_htd_time << endl;
  cout << "Kernel(ms) : " << k1_kernel_time << endl;
  cout << "Device To Host Copy(ms) : " << k1_dth_time << endl << endl;
  
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);
  
  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);

  return EXIT_SUCCESS;
}
