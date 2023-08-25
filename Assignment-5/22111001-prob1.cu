// compile : nvcc -std=c++11 22111001-prob1.cu -o prob1
// Execute : ./prob1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(const double *d_k1_in, double *d_k1_out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= 1 && i + 1 < SIZE1 && j>=0 && j+1 < SIZE1){
    for(int itr = 0; itr < ITER; itr++){
      d_k1_out[(i * SIZE1) + j+1] = d_k1_in[(i-1) * SIZE1 + j+1] + d_k1_in[(i * SIZE1) + j+1] + d_k1_in[(i+1) * SIZE1 + j+1];
    }
  }
}

__global__ void kernel2(double *d_k2_in, double *d_k2_out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= 1 && i + 1 < SIZE2 && j>=0 && j+1 < SIZE2){
    for(int itr = 0; itr < ITER; itr++){
      d_k2_out[(i * SIZE2) + j+1] = d_k2_in[(i-1) * SIZE2 + j+1] + d_k2_in[(i * SIZE2) + j+1] + d_k2_in[(i+1) * SIZE2 + j+1];
    }
  }
}

__global__ void kernel3(double *d_k3_in, double *d_k3_out) {
  __shared__ double *n;
  n = d_k3_in;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= 1 && i + 1 < SIZE2 && j>=0 && j+1 < SIZE2){
    // for(int itr = 0; itr < ITER; itr++){
      d_k3_out[(i * SIZE2) + j+1] = n[(i-1) * SIZE2 + j+1] + n[(i * SIZE2) + j+1] + n[(i+1) * SIZE2 + j+1];
    // }
  }
}


__host__ void serial(const double *h_ser_in, double *h_ser_out) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser_out[i * SIZE1 + j + 1] =
            (h_ser_in[(i - 1) * SIZE1 + j + 1] + h_ser_in[i * SIZE1 + j + 1] +
             h_ser_in[(i + 1) * SIZE1 + j + 1]);
      }
    }
  }
}

__host__ void serial2(const double *h_ser_in, double *h_ser_out) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE2 - 1); i++) {
      for (int j = 0; j < (SIZE2 - 1); j++) {
        h_ser_out[i * SIZE2 + j + 1] =
            (h_ser_in[(i - 1) * SIZE2 + j + 1] + h_ser_in[i * SIZE2 + j + 1] +
             h_ser_in[(i + 1) * SIZE2 + j + 1]);
      }
    }
  }
}

__host__ void check_result(const double *w_ref, const double *w_opt, const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      double this_diff = w_ref[i * size + j] - w_opt[i * size + j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
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
  double *h_ser_in = new double[SIZE1 * SIZE1];
  double *h_ser_out = new double[SIZE1 * SIZE1];
  double *h_ser_in_2 = new double[SIZE2 * SIZE2];
  double *h_ser_out_2 = new double[SIZE2 * SIZE2];

  double *h_k1_in = new double[SIZE1 * SIZE1];
  double *h_k1_out = new double[SIZE1 * SIZE1];

  double *h_k2_in = new double[SIZE2 * SIZE2];
  double *h_k2_out = new double[SIZE2 * SIZE2];
  
  double *h_k3_in = new double[SIZE2 * SIZE2];
  double *h_k3_out = new double[SIZE2 * SIZE2];

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser_in[i * SIZE1 + j] = 1;
      h_ser_out[i * SIZE1 + j] = 1;
      h_k1_in[i * SIZE1 + j] = 1;
      h_k1_out[i * SIZE1 + j] = 1;
    }
  }
  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_ser_in_2[i * SIZE2 + j] = 1;
      h_ser_out_2[i * SIZE2 + j] = 0;
      h_k2_in[i * SIZE2 + j] = 1;
      h_k2_out[i * SIZE2 + j] = 0;
      h_k3_in[i * SIZE2 + j] = 1;
      h_k3_out[i * SIZE2 + j] = 0;
    }
  }

  double clkbegin = rtclock();
  serial(h_ser_in, h_ser_out);
  double clkend = rtclock();
  double time = clkend - clkbegin; 
  cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time) << " GFLOPS; Time = " << time * 1000 << " msec" << endl << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float k1_time; 

  double *d_k1_in;
  double *d_k1_out;
  status = cudaMalloc(&d_k1_in, SIZE1*SIZE1*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_k1_out, SIZE1*SIZE1*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k1_in, h_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k1_out, h_k1_out, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }


  dim3 grid1((1<<8), (1<<8), 1);
  dim3 block1((1<<5), (1<<5), 1);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  kernel1<<<grid1, block1>>>(d_k1_in, d_k1_out);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaMemcpy(h_k1_out, d_k1_out, SIZE1 * SIZE1 * sizeof(double), cudaMemcpyDeviceToHost);
  
  check_result(h_ser_out, h_k1_out, SIZE1);
  cout << "Kernel 1 on GPU(SIZE1): " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3)) << " GFLOPS; Time = " << k1_time << " msec" << endl << endl;




  float k2_time; 
  double *d_k2_in;
  double *d_k2_out;
  status = cudaMalloc(&d_k2_in, SIZE2*SIZE2*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_k2_out, SIZE2*SIZE2*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k2_in, h_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k2_out, h_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  clkbegin = rtclock();
  serial2(h_ser_in_2, h_ser_out_2);
  clkend = rtclock();
  time = clkend - clkbegin; // seconds
  cout << "Second Serial code on CPU(SIZE2): " << ((2.0 * SIZE2 * SIZE2 * ITER) / time) << " GFLOPS; Time = " << time * 1000 << " msec" << endl << endl;

  dim3 grid2((1<<9), (1<<9), 1);
  dim3 block2((1<<5), (1<<5), 1);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  kernel2<<<grid2, block2>>>(d_k2_in, d_k2_out);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k2_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaMemcpy(h_k2_out, d_k2_out, SIZE2 * SIZE2 * sizeof(double), cudaMemcpyDeviceToHost);
  
  check_result(h_ser_out_2, h_k2_out, SIZE2);
  cout << "Kernel 2 on GPU (SIZE2): "
       << ((2.0 * SIZE2 * SIZE2 * ITER) / (k2_time * 1.0e-3))
       << " GFLOPS; Time = " << k2_time << " msec" << endl << endl;



  float k3_time;
  double *d_k3_in;
  double *d_k3_out;
  status = cudaMalloc(&d_k3_in, SIZE2*SIZE2*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_k3_out, SIZE2*SIZE2*sizeof(double));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k3_in, h_k3_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_k3_out, h_k3_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  dim3 grid3((1<<9), (1<<9), 1);
  dim3 block3((1<<5), (1<<5), 1);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  kernel3<<<grid3, block3>>>(d_k3_in, d_k3_out);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k3_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaMemcpy(h_k3_out, d_k3_out, SIZE2 * SIZE2 * sizeof(double), cudaMemcpyDeviceToHost);
  
  check_result(h_ser_out_2, h_k3_out, SIZE2);
  cout << "Kernel 3 on GPU (With optimizations): "
      << ((2.0 * SIZE2 * SIZE2 * ITER) / (k3_time * 1.0e-3))
      << " GFLOPS; Time = " << k3_time << " msec" << endl << endl;

  cudaFree(d_k1_in);
  cudaFree(d_k1_out);

  cudaFree(d_k2_in);
  cudaFree(d_k2_out);

  cudaFree(d_k3_in);
  cudaFree(d_k3_out);

  delete[] h_ser_in;
  delete[] h_ser_out;
  delete[] h_ser_in_2;
  delete[] h_ser_out_2;

  delete[] h_k1_in;
  delete[] h_k1_out;

  delete[] h_k2_in;
  delete[] h_k2_out;

  delete[] h_k3_in;
  delete[] h_k3_out;

  return EXIT_SUCCESS;
}
