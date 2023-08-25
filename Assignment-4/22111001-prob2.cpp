// Compile: g++ -std=c++11 -fopenmp 22111001-prob2.cpp -o quicksort
// Execute: ./quicksort

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// Make sure to test with other sizes
#define N (1 << 20)

void swap(int* x, int* y) {
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

int partition(int* arr, int low, int high) {
  int pivot, last;
  pivot = arr[low];
  swap(arr + low, arr + high);
  last = low;
  for (int i = low; i < high; i++) {
    if (arr[i] <= pivot) {
      swap(arr + last, arr + i);
      last += 1;
    }
  }
  swap(arr + last, arr + high);
  return last;
}

void serial_quicksort(int* arr, int start, int end) {
  int part;
  if (start < end) {
    part = partition(arr, start, end);

    serial_quicksort(arr, start, part - 1);
    serial_quicksort(arr, part + 1, end);
  }
}

void par_quicksort(int* arr, int start, int end) {
  // If elements are remaining
  if (start < end) {
    // ind is index at which we have place element which would be there if array was sorted 
    // we have to partition our array in two sub arrays and call this function in those sub arrays to sort 
    // those partitions
    int ind = partition(arr, start, end);
    // cout << "start : " << start << " ind : " << ind << " end : " << end << endl;

    // Open MP parallel block started
    #pragma omp parallel
    {
      // Single make sure that we make do not make duplicate tasks
      #pragma omp single
      {
        // This block defines the task which is to be executed
        #pragma omp task
        {
          // If number of elements in subarray 1 is greater than or equal to (1<<10) then we will call
          // parallel quicksort, but if not then serial quicksort will be called
          // As making and destroying task is an overhead, and we can not afford overhead which 
          // dominates the computation in our task
          if(ind - start - 2 >= (1<<10))
            par_quicksort(arr, start, ind - 1);
          else
            serial_quicksort(arr, start, ind - 1);
        }
        #pragma omp task
        {
          // Same logic with subarray 2 as was in subarray 1 
          if(end - ind - 2 >= (1<<10))
            par_quicksort(arr, ind + 1, end);
          else
            serial_quicksort(arr, ind + 1, end);
        }
      }
    }
  }
}

int main() {
  int* ser_arr = nullptr;
  int* par_arr = nullptr;
  ser_arr = new int[N];
  par_arr = new int[N];
  for (int i = 0; i < N; i++) {
    ser_arr[i] = rand() % 1000;
    par_arr[i] = ser_arr[i];
  }

//   cout << "Unsorted array: " << endl;
  // for (int i = 0; i < N; i++) {
  //   cout << ser_arr[i] << "\t";
  // }
  //cout << endl << endl;

  HRTimer start = HR::now();
  serial_quicksort(ser_arr, 0, N - 1);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial quicksort time: " << duration << " us" << endl;

  // cout << "Sorted array: " << endl;
  // for (int i = 0; i < N; i++) {
  //   cout << ser_arr[i] << "\t";
  // }
  // cout << endl << endl;

  start = HR::now();
  par_quicksort(par_arr, 0, N - 1);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OpenMP quicksort time: " << duration << " us" << endl;

  for (int i = 0; i < N; i++) {
    assert(ser_arr[i] == par_arr[i]);
  }

  return EXIT_SUCCESS;
}
