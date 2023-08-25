// Compile: g++ -O2 -mavx -march=native -o problem2 problem2.cpp
// Execute: ./problem2

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::ios;

const int N = (1 << 13);
const int Niter = 10;
const double THRESHOLD = 0.000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_ref, double* __restrict__ z_ref) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THE CODE
// You can create multiple versions of the optimized() function to test your changes

void optimized(double** A, double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

void optimized2(double** A, double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j+=8) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

       y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i];

       y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i];

       y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i];

      y_opt[j+4] = y_opt[j+4] + A[i][j+4] * x[i];
      z_opt[j+4] = z_opt[j+4] + A[j+4][i] * x[i];

      y_opt[j+5] = y_opt[j+5] + A[i][j+5] * x[i];
      z_opt[j+5] = z_opt[j+5] + A[j+5][i] * x[i];

      y_opt[j+6] = y_opt[j+6] + A[i][j+6] * x[i];
      z_opt[j+6] = z_opt[j+6] + A[j+6][i] * x[i];

      y_opt[j+7] = y_opt[j+7] + A[i][j+7] * x[i];
      z_opt[j+7] = z_opt[j+7] + A[j+7][i] * x[i];
    }
  }
}

void optimized3(double** A, double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i+=16) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i+1][j] * x[i+1];
      z_opt[j] = z_opt[j] + A[j][i+1] * x[i+1];

      y_opt[j] = y_opt[j] + A[i+2][j] * x[i+2];
      z_opt[j] = z_opt[j] + A[j][i+2] * x[i+2];

      y_opt[j] = y_opt[j] + A[i+3][j] * x[i+3];
      z_opt[j] = z_opt[j] + A[j][i+3] * x[i+3];

      y_opt[j] = y_opt[j] + A[i+4][j] * x[i+4];
      z_opt[j] = z_opt[j] + A[j][i+4] * x[i+4];

      y_opt[j] = y_opt[j] + A[i+5][j] * x[i+5];
      z_opt[j] = z_opt[j] + A[j][i+5] * x[i+5];

      y_opt[j] = y_opt[j] + A[i+6][j] * x[i+6];
      z_opt[j] = z_opt[j] + A[j][i+6] * x[i+6];

      y_opt[j] = y_opt[j] + A[i+7][j] * x[i+7];
      z_opt[j] = z_opt[j] + A[j][i+7] * x[i+7];

       y_opt[j] = y_opt[j] + A[i+8][j] * x[i+8];
      z_opt[j] = z_opt[j] + A[j][i+8] * x[i+8];

       y_opt[j] = y_opt[j] + A[i+9][j] * x[i+9];
      z_opt[j] = z_opt[j] + A[j][i+9] * x[i+9];

       y_opt[j] = y_opt[j] + A[i+10][j] * x[i+10];
      z_opt[j] = z_opt[j] + A[j][i+10] * x[i+10];

       y_opt[j] = y_opt[j] + A[i+11][j] * x[i+11];
      z_opt[j] = z_opt[j] + A[j][i+11] * x[i+11];

      y_opt[j] = y_opt[j] + A[i+12][j] * x[i+12];
      z_opt[j] = z_opt[j] + A[j][i+12] * x[i+12];

      y_opt[j] = y_opt[j] + A[i+13][j] * x[i+13];
      z_opt[j] = z_opt[j] + A[j][i+13] * x[i+13];

      y_opt[j] = y_opt[j] + A[i+14][j] * x[i+14];
      z_opt[j] = z_opt[j] + A[j][i+14] * x[i+14];

      y_opt[j] = y_opt[j] + A[i+15][j] * x[i+15];
      z_opt[j] = z_opt[j] + A[j][i+15] * x[i+15];
    }
  }
}

void optimized4(double**  A, double*  x, double*   y_opt, double*  z_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j+=4) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
    }
  }
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i+=4) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      z_opt[j] = z_opt[j] + A[j][i+1] * x[i+1];
      z_opt[j] = z_opt[j] + A[j][i+2] * x[i+2];
      z_opt[j] = z_opt[j] + A[j][i+3] * x[i+3];
    }
  }
}


void avx_version(double** A, double* x, double* y_opt, double* z_opt) {
    __m256d rY, rX, rA, rZ;
    for(int i = 0; i<N; i++){
      for(int j = 0; j<N; j+=4){
        rY = _mm256_set_pd(y_opt[j+3], y_opt[j+2], y_opt[j+1], y_opt[j]);
        rX = _mm256_set_pd(x[i], x[i], x[i], x[i]);
        rA = _mm256_set_pd(A[i][j+3], A[i][j+2], A[i][j+1], A[i][j]);
        rY = _mm256_add_pd(rY, _mm256_mul_pd(rA, rX));
        _mm256_storeu_pd(y_opt+j, rY);
      }
    }

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i+=4) {
        rZ = _mm256_set_pd(0, 0, 0, z_opt[j]);
        rX = _mm256_set_pd(x[i+3], x[i+2], x[i+1], x[i]);
        rA = _mm256_set_pd(A[j][i+3], A[j][i+2], A[j][i+1], A[j][i]);
        rZ = _mm256_add_pd(rZ, _mm256_mul_pd(rA, rX));
        _mm256_storeu_pd(z_opt+i, rZ);
      }
    }
}

int main() {
  double clkbegin, clkend;
  double t;
  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";
       cout << endl;



   clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version 1 (Distributing inner loop as well outer loop): Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly


   clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version 2 (Unrolling inner loop 8 times): Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

   clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized3(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version 3 (Unrolling and jamming outer loop then unrolling inner loop): Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

   clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized4(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version 4 (Distributing both loops then interchanging loops of second loop and unrolling): Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }



  // Version with intinsics

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    avx_version(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  return EXIT_SUCCESS;
}
