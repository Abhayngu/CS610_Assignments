// Compile: g++ -std=c++11 22111001-prob4.cpp -o find-max -ltbb
// Execute: ./find-max

#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <climits>
#include <cstdlib>
#include <tbb/tbb.h>
#include <time.h>

using std::cout;
using std::min;
using std::endl;
using namespace tbb;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 26)

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = std::numeric_limits<uint32_t>::min();
  uint32_t index_of_max = 0;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

class FindMax{
        const uint32_t* arr;
      public:
        uint32_t maxEl;
        uint32_t index;

        // Initialized maxEl with min value of datatype used and index -1
        FindMax(const uint32_t* a) : arr(a), maxEl(std::numeric_limits<uint32_t>::min()), index(-1) {}

        // Constructor which is called when splitting of work happens between threads
        FindMax( FindMax& tempObj, tbb::split ) : arr(tempObj.arr), maxEl(std::numeric_limits<uint32_t>::min()), index(-1) {}

        // Method which combines the solution of threads
        void join( const FindMax& obj ) {
            // If current object maxEl and obj.maxEl are same then update index to be minimum of them
            if(maxEl == obj.maxEl){
                index = min(index, obj.index);
            // Else if current object maxEl is less than the obj.maxEl, update both maxEl and index
            }else if(maxEl < obj.maxEl){
              index = obj.index;
              maxEl = obj.maxEl;
            }
        }

        // Overloaded method which is computing index having max element in the array
        void operator()(blocked_range<size_t>& r){
            for(size_t i = r.begin(); i<r.end(); i++){
              // if current element is greater than maxEl then update index and maxEl
              if(maxEl < arr[i]){  
                index = i;
                maxEl = arr[i];
              } 
            }
        }
    };

uint32_t tbb_find_max(const uint32_t* a) {
    // Create an object of FindMax class 
    FindMax obj(a);
    // Call parallel_reduce by giving it range and the object in the arguement
    parallel_reduce(blocked_range<size_t>(0, N), obj);
    return obj.index;  
}

int main() {
  uint32_t* a = new uint32_t[N];
  srand(time(0));
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
