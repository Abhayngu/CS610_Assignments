// g++ -msse4 -mavx -march=native -O3 -fopt-info-vec-optimized -fopt-info-vec-missed -o problem5 problem5.cpp

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 16)
#define SSE_WIDTH 128
#define AVX2_WIDTH 256

void print_array(int* array);

__attribute__((optimize("no-tree-vectorize"))) int ref_version(int* __restrict__ source,
                                                               int* __restrict__ dest) {
  __builtin_assume_aligned(source, 64);
  __builtin_assume_aligned(dest, 64);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(int* source, int* dest) { return 0; }

int sse4_version(int* source, int* dest) { return 0; }

int avx2_version(int* source, int* dest) { return 0; }

int* array = nullptr;
int* ref_res = nullptr;
int* omp_res = nullptr;
int* sse_res = nullptr;
int* avx2_res = nullptr;

__attribute__((optimize("no-tree-vectorize"))) int main() {
  array = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  ref_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  omp_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  sse_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  avx2_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));

  for (int i = 0; i < N; i++) {
    array[i] = 1;
    ref_res[i] = 0;
    omp_res[i] = 0;
    sse_res[i] = 0;
    avx2_res[i] = 0;
  }

  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;

  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE4 result is wrong!\n"));
  cout << "SSE4 version: " << val_sse << " time: " << duration << endl;

  start = HR::now();
  int val_avx = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx << " time: " << duration << endl;

  return EXIT_SUCCESS;
}

void print_array(int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}
