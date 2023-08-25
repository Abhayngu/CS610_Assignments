// Compile: g++ -std=c++11 -fopenmp 22111001-prob1.cpp -o fibonacci -ltbb
// Execute: ./fibonacci

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>
#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include <tbb/task.h>
#define N 40

using std::cout;
using std::vector;
using std::endl;
using std::chrono::duration_cast;
using namespace tbb;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// Serial Fibonacci
long ser_fib(int n) {
  if (n < 2) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

long omp_fib_v1(int n) {
  // If n = 0 or n = 1 then return n
  // since fibonacci for negatve n does not make sense hence for that case I have returned -1
  if(n < 0){
    return -1;
  }
  if (n < 2) {
    return (n);
  }
  long x, y, ans;
  // Open MP parallel block starts from here
  #pragma omp parallel shared(n)
  {
    // single is used so that duplicate tasks are not created
    #pragma omp single
    {
      // Defining task block 
      #pragma omp task untied 
      {
        // Setting offset that if n <= 18 then we will use serial function
        // Or else we will use parallel version 
        if(n > 18)
          x = omp_fib_v1(n - 1);  
        else
          x = ser_fib(n-1);
      }
      // Defining task block
      #pragma omp task untied 
      {
        // Setting offset that if n <= 18 then we will use serial function
        // Or else we will use parallel version 
        if(n > 18)
          y = omp_fib_v1(n - 2);  
        else
          y = ser_fib(n-2);
      }
      // using taskwait so that ans is computed only when x and y are calculated
      #pragma omp taskwait
        ans = x + y;  
    }
  }
  return ans;
}

long omp_fib_v2(int n) {
  // If n = 0 or n = 1 then return n
  // since fibonacci for negatve n does not make sense hence for that case I have returned -1
  if(n < 0){
    return -1;
  }
  if (n == 0 || n == 1) {
    return (n);
  }
  long x=0, y=1, ans; 
  // Open MP parallel block starts from here
  #pragma omp parallel
  {
    // single is used so that duplicate tasks are not created
    #pragma omp single 
    {
      // Optimized fib program by replacing recursion by loop
      // Space is also optimized in this case, I have used no array to store 
      // fib of other numbers except n as we have to return only fib of n
      // we can use untied clause but that won't make any difference
      // as at a time there is only one task executing due to dependencies 
      // in the fibonacci program
      for(int i = 2; i<=n; i++){
        #pragma omp taskwait
        {
          ans = y + x;
          x = y;
          y = ans;
        }
      }
    }
  }
  return ans;
}


// Class FibBlocking is derived from the base class task in public manner
// which means all the public data members and member function of task class
// can be accessed within FibBlocking class
// This is why we are able to override execute method of task in our class
class FibBlocking: public task{
  public:
    const long n;
    long* const sum;

    // Constructor to initialize data members
    FibBlocking(long num, long* initialSum) : n(num), sum(initialSum) {}

    // In following line we are over-riding a pure virtual method
    task* execute(){
    // if n is small do serial execution
      if(n <= 15) {
            *sum = ser_fib(n);
      } else {
        long x, y;
        // allocate_child is inherited method which is used to allocate space for the task
        FibBlocking& a = *new(task::allocate_child()) FibBlocking(n-1, &x);
        FibBlocking& b = *new(task::allocate_child()) FibBlocking(n-2, &y);
        
        // number 3 in the following function call arguement represents
        // two children and an additional implicit reference that is required by method spawn_and_wait_for_all 
        set_ref_count(3);

        // In the following line we are spawning the second task
        spawn(b);

        // In the following line we are spawning first task and waiting for all child tasks to get over
        spawn_and_wait_for_all(a);
        *sum = x + y;
      }
      return NULL;
    }
  };



long tbb_fib_blocking(int n) {
  // making a long pointer sum which is going to store the value of fibonacci
  long* sum = new long;

  // Making an object of FibBlocking using allocate_root() method which 
  // allocates space for root task
  FibBlocking& a = *new(task::allocate_root())FibBlocking(n, sum);

  // Spawning root task and waiting for it to get complete
  task::spawn_root_and_wait(a);

  // Returning sum after root task has been completed
  return *sum;
}


// A Helper class which is basically derived from task and simply override
// execute function to only store sum of x and y in sum variable
class FibCHelperClass : public task{
public:
  long* const sum;
  long x, y;
  FibCHelperClass(long* s) : sum(s) {}
  task* execute(){
    *sum = x + y;
    return NULL;
  }
};

// FibContinuation class is a derived class from class task
// It basically creates child tasks and spawn them 
// In this class parent does not wait for root to get completed 
// Which is why we are using Helper class so that even if parent returns 
// In future we will get some value from child tasks and add it to the sum variable
class FibContinuation : public task{
public:
  long* sum;
  long n;
  
  // Constructor to initialize data members
  FibContinuation(long num, long* s) : n(num), sum(s) {}

  // execute method is overridden 
  task* execute(){
    // If n is smaller than or equal to 15 just compute fibonacci serially
    if(n <= 15){
      *sum = ser_fib(n);
    }else{
      // Create an object of helper class which basically keep track of x and y values 
      // of all the child tasks even when their parent function returns
      FibCHelperClass& obj = *new(allocate_continuation()) FibCHelperClass(sum);

      // Creating one child task using allocate_child() method and saving its value in obj.x
      // for n - 2
      FibContinuation& a = *new(obj.allocate_child()) FibContinuation(n-2, &obj.x);

      // Creating one child task using allocate_child() method and saving its value in obj.y
      // for n - 1
      FibContinuation& b = *new(obj.allocate_child()) FibContinuation(n-1, &obj.y);

      // Setting reference count 2 for both the child tasks
      obj.set_ref_count(2);

      // Spawning second child task
      spawn(b);

      // This method returns task* which we are returning and it will get executed by the thread
      return &a;
    }
    return NULL;
  }
};


long tbb_fib_cps(int n) {
  // Creating long sum pointer to store fib of N
  long* sum = new long;

  // Creating an object of FibContinuation which basically creates
  // root task using method allocate_root()
  FibContinuation& a = *new(task::allocate_root())FibContinuation(n, sum);

  // Spawning root task and waiting for it to get complete
  task::spawn_root_and_wait(a);

  // Returning the Fib(N) stored in the address pointed by sum pointer
  return *sum;
}

int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  // cout << s_fib << endl;
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2 = omp_fib_v2(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
