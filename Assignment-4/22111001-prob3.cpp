// Compile : g++ -std=c++11 -fopenmp 22111001-prob3.cpp -o pi -ltbb
// Execute : ./pi

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tbb/tbb.h>
#include <omp.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// const int NUM_INTERVALS = std::numeric_limits<int>::max();
const int NUM_INTERVALS = 1000000000;

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h ;
    
  }
  double pi = 4 * sum * dx;
  return pi;
}

double omp_pi() {

  // Since p is initialised outside the parallel block, it will be shared among threads
  double p = 0.0;

  // Defining number of steps to be taken 
  double step = 1.0 / NUM_INTERVALS;

  uint16_t num_of_threads;
  // Starting omp parallel block
  #pragma omp parallel 
  {
    // sum is a private variable of each thread, therefore two different thread 
    // will work upon two different variable
    double sum = 0.0;

    // Only one thread is going to execute the block inside single
    // Single block has implicit barrier which makes sure that
    // No thread will surpass its block without it getting completed
    #pragma omp single 
      num_of_threads = omp_get_num_threads();

    // Each thread will get its id in variable thread_id 
    uint16_t thread_id = omp_get_thread_num();   

    // This loops make sure that work is divided among threads equally
    // Each thread will take iterations multiple to its id
    // For example thread having id 4 will execute iteration number 4, 8, 12 ... so on
    // Each thread will compute their local sum in variable sum
    for(int i = thread_id; i < NUM_INTERVALS; i+=num_of_threads){

      double x = (i + 0.5) * step;
      double h = std::sqrt( 1.0 - x * x);

      // Since sum is private to each thread, therefore false sharing won't happen in this case
      sum += h;
    }

    // Since p is shared variable, therefore we need to provide mutual exclusion to it
    // that is only one thread at a time will be accessing p += sum line
    #pragma omp atomic
      p +=  sum;

  }

  // Returning value of pi
  return 4 * step * p;
}

double tbb_pi() {
  class ComputePi{
    public:

      // Data members of the class 
      double sum;
      double step = 1.0/NUM_INTERVALS;

      // Constructor to initialize sum variable
      ComputePi(double s) : sum(s) {}

      // This constructor will be called when the work will be distributed among different threads
      ComputePi(ComputePi& obj, tbb::split) : sum(0.0) {}

      // This method is to join the result of computation of threads
      void join(ComputePi& obj){
        sum += obj.sum;
      }

      // This operator() method is overloaded and it contains all the logic of pi value computation
      void operator()(tbb::blocked_range<int>& r){

        // This is the localSum for each thread
        double localSum = 0;

        // Following loop iterations will be distributed among threads
        for(int i = r.begin(); i!=r.end(); i++){
          double x = (i + 0.5) * step;
          double h = std::sqrt( 1.0 - x * x);
          localSum += h;
        }

        // Threads adding their local sum in sum variable
        sum += localSum;
      }
  };

  // Making class object
  ComputePi obj(0.0);

  // Calling parallel_reduce to perform reduction operation
  parallel_reduce(tbb::blocked_range<int>(0, NUM_INTERVALS), obj);

  // Return pi value
  return obj.sum * obj.step * 4; 
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "g++ -std=c++11 pi.cpp -o pi; ./pi"
// End:
