#include <iostream>
#include <omp.h>

using namespace std;

int main(){
	int n = 18;
	int noOfChunks = 3;
	#pragma omp parallel
	{
		#pragma omp single
		cout << "Num of threads : " << omp_get_num_threads() << endl;
		#pragma omp for schedule(static, noOfChunks)
		for(int i = 0; i<n; i++){
			#pragma omp critical
	    	cout << i << " executed by " << omp_get_thread_num() << endl;
		}	
	}		
}