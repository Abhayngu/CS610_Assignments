#include <iostream>
#include <pthread.h>
#include <fstream>
#include <vector>
#include <stack>
#include <time.h>
// #include <windows.h>

using namespace std;

int p, b, w;
vector<int> buffer;
fstream fio;
ofstream fo;
stack<int> s;
pthread_mutex_t mutexBuffer, mutexFileInput, mutexFilePrime;
pthread_cond_t condBufferEmpty, condBufferFull;
clock_t cs, ce;

void* produceNumber(void* ptr){
	string line;
	int id = *((int *)ptr);
	while(1){
		    if(fio.eof()){
		    	break;
		    }
		    pthread_mutex_lock(&mutexBuffer);
			while(s.size() == b){
				// cout << "-- producer in busy waiting-- " << id << endl;
				if(fio.eof()){
					// cout << "<-----------------producer dead in busy waiting---------------->"  << id << endl;
					pthread_mutex_unlock(&mutexBuffer);
					pthread_cond_broadcast(&condBufferEmpty);
					return NULL;
				}
				pthread_cond_wait(&condBufferFull, &mutexBuffer);
			}
			
			// cout << "mutex Buffer Locked by producer " << id  << endl;
			pthread_mutex_lock(&mutexFileInput);
			// cout << "mutex File Locked by producer " << id  << endl;
			if(fio.eof()){
				pthread_mutex_unlock(&mutexBuffer);
				pthread_mutex_unlock(&mutexFileInput);
				pthread_cond_broadcast(&condBufferEmpty);
				// cout << "<-----------------producer dead ---------------->"  << id << endl;
				return NULL;
			}
			getline(fio, line);
			if(line  == "" || line == " " || line == "\n") {
				pthread_mutex_unlock(&mutexBuffer);
				pthread_mutex_unlock(&mutexFileInput);
				pthread_cond_broadcast(&condBufferEmpty);
				break;
			}
		    int num = stoi(line);
			s.push(num);
			// cout << "Produced : " << num << endl;
			cout << "Number produced by thread id : "<< id << " --> " << num << endl;
			// Sleep(1000);
			pthread_mutex_unlock(&mutexBuffer);
			pthread_mutex_unlock(&mutexFileInput);
			pthread_cond_broadcast(&condBufferEmpty);
	}

	// cout << "<-----------------producer dead exiting 1---------------->" << id << endl;
	return NULL;	
}

void* checkPrime(void* ptr){
	int id = *((int *)ptr);
	// cout  << "Consumer "  << id << " activated !!" << endl;
	while(1){
		if(fio.eof() && s.empty()){
			// cout << "<--------------- End of file by consumer intial break -------------> " << id << endl;
			break;
		}
		// Sleep(2000);
		// cout << "Going to unlock mutex by consumer " << id << endl;
		pthread_mutex_lock(&mutexBuffer);
		// cout << "mutex Buffer Locked by consumer " << id << endl;
		while(s.empty()){
			// cout << "--Consumer in busy waiting-- " << id << endl;
			if(fio.eof() && s.empty()){
				pthread_mutex_unlock(&mutexBuffer);
				pthread_cond_broadcast(&condBufferFull);
				// cout << "<--------------- End of file by consumer -------------> " << id << endl;
				return NULL;
			}
			pthread_cond_wait(&condBufferEmpty, &mutexBuffer);
		}
		pthread_mutex_lock(&mutexFilePrime);
		// cout << "mutex File Locked by consumer " << id << endl;
		if(fio.eof() && s.empty()){
			// cout << "mutex Buffer unlcked by consumer " << id << endl;
			pthread_mutex_unlock(&mutexBuffer);
			pthread_mutex_unlock(&mutexFilePrime);
			pthread_cond_broadcast(&condBufferFull);
			 // cout << "<--------------- End of file by consumer -------------> " << id << endl;
			 pthread_exit(NULL);
		}
		int num = s.top();
		s.pop();
		bool isPrime = true;
		for(int i = 2; i*i<=num; i++){
			if(num % i == 0){
				isPrime = false;
				break;
			}
		}
		// cout  << "Consumed : " << num << endl;
		cout  << "Number consumed by thread id : " << id << " --> " << num << endl;
		if(isPrime){
			fo << num << endl;
		}
		pthread_mutex_unlock(&mutexBuffer);
		pthread_mutex_unlock(&mutexFilePrime);
		pthread_cond_broadcast(&condBufferFull);
	}	
	// cout << "<--------------- End of file by consumer -------------> " << id << endl;
	return NULL;
}

int main(int argc, char** argv){
	p = atoi(argv[1]);
	b = atoi(argv[2]);
	w = atoi(argv[3]);
	buffer.resize(b);
	cout << " p : " << p << endl << " b : " << b << endl << " w : " << w << endl;
	pthread_mutex_init(&mutexBuffer, NULL);
	pthread_mutex_init(&mutexFileInput, NULL);
	pthread_mutex_init(&mutexFilePrime, NULL);
	pthread_cond_init(&condBufferEmpty, NULL);
	pthread_cond_init(&condBufferFull, NULL);

	pthread_t producerThread, consumerThread;

	cs = clock();

	vector<pthread_t> pT(p);
	fio.open("in.txt");
	for(int i = 0; i<p; i++){
		int* id = new int;
		*id = i+1;
    	pthread_create(&pT[i], NULL, &produceNumber, id);
    }
   

	// Sleep(1000);

    vector<pthread_t> wT(w);
	fo.open("prime.txt");
	for(int i = 0; i<w; i++){
		int* id = new int;
		*id = i+1;
    	pthread_create(&wT[i], NULL, &checkPrime, id);
    }
     for(int i = 0; i<p; i++){
    	pthread_join(pT[i], NULL);
    }
    for(int i = 0; i<w; i++){
    	pthread_join(wT[i], NULL);
    }

	ce = clock();
	cout << endl << "Clock cycles program took ----------> " << ce - cs << endl;

	pthread_mutex_destroy(&mutexBuffer);
	pthread_mutex_destroy(&mutexFileInput);
	pthread_mutex_destroy(&mutexFilePrime);
	pthread_cond_destroy(&condBufferEmpty);
	pthread_cond_destroy(&condBufferFull);
    fio.close();
    fo.close();
	return 0;
}