#include <iostream>
#include <pthread.h>
#include <random>
#include <vector>
#include <time.h>
#include <iomanip>

using namespace std;


/* 

<---------------- I have written the approach in brief which I have used at the end of this code file within comment -------------->

*/


// Global array in which we are going to count number of inversions
vector<int> arr;

// Mutex variables
pthread_mutex_t mutexIndex, mutexInversionCount;

// numOfInversion is numberOfInversion at any point and ind at any point will give us index of number which will be taken next by a thread
long long numOfInversion = 0;
long long ind = 0;

// Function to generate random numbers
void generateRandomNumbers(int lowerBound, int upperBound){
	int numOfElements = upperBound -  lowerBound; 

	// Resizing global vector(array)
	arr.resize(numOfElements);
	random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_int_distribution<> distr(0, numOfElements-1); // define the range
    for(int i=0; i<numOfElements; ++i){
    	arr[i] = distr(gen); 		// generate numbers
    	// cout << arr[i] << ' '; 	// uncomment to see the array elements separated by space
    }
}

int findInversionCount()
{
	int n = (int)arr.size();
    long long inversionCount = 0;
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (arr[i] > arr[j]) {
                inversionCount++;
            }
        }
    }
    cout << endl << "Number of inversion from single threaded Execution : " << inversionCount << endl;
}

void* countInversion(void* ptr){
	while(1){

		// Locking the mutexIndex variable
		pthread_mutex_lock(&mutexIndex);

		// If all the indexes have already been taken up by threads the current thread can exit
		if(ind >= (int)arr.size()){
			pthread_mutex_unlock(&mutexIndex);
			pthread_exit(NULL);
		}

		// Created a local variable so that when we can use this value even after unlocking mutex or else
		// another thread may change it just after it unlocks the mutexIndex
		int x = ind;

		// Increasing array index(global variable that's why used mutex)
		ind++;

		

		// Unlocking teh mutexIndex
		pthread_mutex_unlock(&mutexIndex);

		// Inversion count logic
		int curNum = arr[x];
		int n = (int)arr.size();
		long long k = 0;
		for(int j = x; j<n; j++){
			// cout << "i : " << i << " " << "j : " << j << " "  << "Cur Num : " << curNum << " " << "Comparing Num " << arr[j] << endl;
			if(curNum > arr[j]) k++;	
		}

		// Locking the mutexInversionCount variable as it is global and every thread can update(increase) it
		pthread_mutex_lock(&mutexInversionCount);

		numOfInversion += k;

		// Unlocking the mutexInversionCount variable
		pthread_mutex_unlock(&mutexInversionCount);
	}
}

int main(int argc, char ** argv){

	// Initializing mutex
	pthread_mutex_init(&mutexIndex, NULL);
	pthread_mutex_init(&mutexInversionCount, NULL);

	// Declaring variables and setting it to command line arguements
	int numOfElements, numOfThreads;
	numOfElements = atoi(argv[1]), numOfThreads = atoi(argv[2]);

	// Calling generateRandomNumbers function which will fill our array with random elements
	generateRandomNumbers(0, numOfElements); 

	// Creating a vector of type pthread_t
	vector<pthread_t> threads(numOfThreads);

	// Declaring two clock_t type variable to calculate the time it takes for normal and multithreaded function
	clock_t start, end;
	
	// Starting clock for multithreading function
	start = clock();
	for(int i = 0; i<numOfThreads; i++){
		pthread_create(&threads[i], NULL,countInversion, NULL);
	}
	for(int i = 0; i<numOfThreads; i++){
		pthread_join( threads[i], NULL);
	}
	cout << endl << "Number of inversion from multi threaded Execution : " << numOfInversion << endl;

	// Ending clock for multi threading function
	end = clock();

	// Calculating time taken by multithreading function 
	double time_taken_mt = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by multi threaded execution is : " << fixed << time_taken_mt << setprecision(5) << " seconds" << endl;

    // Starting clock for normal inversion count function
    start = clock();

    // Calling normal inversion count function
	findInversionCount();
	
	// Ending clock for normal inversion count function
	end = clock();

	// Calculating time taken by normal inversion count function
	double time_taken_st = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by single threaded inversion count method is : " << fixed << time_taken_st << setprecision(5) << " seconds" << endl;
	if((!time_taken_st == 0 || time_taken_mt == 0)){
		if(time_taken_st > time_taken_mt){
			cout << endl << "Multithread execution performs " << setprecision(5) << time_taken_st/time_taken_mt << " better than single threaded execution !!" << endl;	
		}else if(time_taken_st < time_taken_mt){
			cout << endl << "Single threaded execution performs " << setprecision(5) << time_taken_mt/time_taken_st << " better than multi threaded execution !!" << endl;				
		}
		
	}else if(time_taken_st == 0 && time_taken_mt != 0){
		cout << endl << "Multithreaded execution performs poor in this case " << endl;
	}	

    // Destroying Mutex
    pthread_mutex_destroy(&mutexIndex);
    pthread_mutex_destroy(&mutexInversionCount);
	
    return 0;
}

/* 

Approach : The approach which I have followed is to initialize a global index variable with 0, then I will create as many number of threads
given in the arguement then those thread will take that index variable one by one, they will work on that and before working they will increase 
it by 1 so that the next thread always work upon the next index not the same index. I have use mutex lock for index as it is global and thread 
will increase it. Also I have use mutex for numOfInversion as it is also global and a thread will increase it by 0 or more than 0.


*/