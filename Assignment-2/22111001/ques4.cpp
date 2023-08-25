#include <iostream>
#include <pthread.h>
#include <unordered_map>
#include <queue>
#include <stdint.h>
#include <cstdlib>
#include <time.h>

using namespace std;
int n;
unordered_map<int, int> kv;
pthread_t manager, worker[4];
pthread_cond_t condBuffer;
pthread_mutex_t mutexJobBuffer, mutexKVStore;

struct Operation{
	int type;
	uint32_t key;
	uint64_t value;
};

struct Job{
	struct Operation* op;
};


queue<struct Operation*> opq;
queue<struct Job*> jobq;


void insert(struct Job* tempJob){
	int key = tempJob->op->key;
	int value = tempJob->op->value;
	kv[key] = value;
}


int update(struct Job* tempJob){
	int key = tempJob->op->key;
	int value = tempJob->op->value;
	if(kv.find(key) == kv.end()){
		return -1;
	}
	kv[key] = value;
	return 1;
}

int del(struct Job* tempJob){
	int key = tempJob->op->key;
	int value = tempJob->op->value;
	if(kv.find(key) == kv.end()){
		return -1;
	}
	kv.erase(key);
	return 1;
}

int find(struct Job* tempJob){
	int key = tempJob->op->key;
	if(kv.find(key) == kv.end()){
		return -1;
	}else{
		return 1;
	}
}

void* workerRoutine(void *){
	while(1){
		if(n<=0 && jobq.empty()){
			break;
		}
		pthread_mutex_lock(&mutexJobBuffer);
		while(jobq.size() == 0){
			if(n<=0 && jobq.empty()){
				pthread_mutex_unlock(&mutexJobBuffer);
				return NULL;
			}
			pthread_cond_wait(&condBuffer, &mutexJobBuffer);
		}
		if(n<=0 && jobq.empty()){
				break;
		}
		struct Job* tempJob = jobq.front();
		jobq.pop();
		// cout << "Popped " << endl;
		pthread_mutex_unlock(&mutexJobBuffer);
		int type = tempJob->op->type;
		pthread_mutex_lock(&mutexKVStore);
		int retVal;
		switch(type){

			case 0 :insert(tempJob); 
					cout << "Insert operation : " << tempJob->op->value << " inserted successfully " << endl;
					// for(auto itr = kv.begin(); itr!=kv.end(); itr++){
					// 	cout << itr->first << " " << itr->second << endl;
					// }
					 break;
					
			case 1 : retVal = update(tempJob);
					if(retVal == 1){
						cout << "Update operation : " << tempJob->op->key << " value updated successfully " << endl;
					 }else{
					 	cout <<  "Update operation : " << tempJob->op->key << " key not found to update!" << endl;
					 }
						break;
			case 2 : retVal = del(tempJob);
					 if(retVal == 1){
						cout <<"Delete operation : " << tempJob->op->key << " value deleted successfully " << endl;
					 }else{
					 	cout << "Delete operation : " << tempJob->op->key << " key not found to delete!" << endl;
					 }
						break;
			case 3 : retVal = find(tempJob);
					 if(retVal == 1){
						cout << "Find operation : " <<tempJob->op->key << " founded successfully " << endl;
					 }else{
					 	cout << "Find operation : " << tempJob->op->key << " key not found!" << endl;
					 }
						break;
			default : cout << "Invalid value in type variable " << type << endl;
		}
		pthread_mutex_unlock(&mutexKVStore);
	}
	// cout << "Worker exits" << endl;
}

int enqueue(struct Operation* opp){
	if(jobq.size() >= 8 && !jobq.empty()){
			n--;
			return -1;
	}
	struct Job* tempJob = (struct Job*)malloc(sizeof(struct Job));
	tempJob->op = opp;
	jobq.push(tempJob);
	n--;
	pthread_cond_broadcast(&condBuffer);
	return jobq.size()-1;
}

// --------------------------------------------- Function below is to change test cases ----------------------------------------

void helperFunction(){
	for(int i = 0; i<n; i++){
		struct Operation* temp = (struct Operation*)malloc(sizeof(struct Operation));

		int type, key, value;

		// I have taken everything in random
		// A for loop is running for n times, change values assigned to type, key and value 
		// Can also remove the srand function which I have put to seed the random
		// Change the following 5 lines to put a custom test case for each operation 
		type = i%4;
		key = ((i*2) % 5) + 1;
		value = (i*5) + 1;
		cout << "type : " << type << " key : " << key << " value : " << value << endl;


		// Dont change the following lines, these are for setting values to operation pointer 
		// and pushing these operations into the operation queue
		temp->type = type;
		temp->key = key;
		temp->value = value;
		opq.push(temp);
	}
}

// --------------------------------------------- Function above is to change test cases -----------------------------------------



int main(int argc, char **argv){
	n = atoi(argv[1]);
	int x = n;
	pthread_mutex_init(&mutexJobBuffer, NULL);
	pthread_mutex_init(&mutexKVStore, NULL);
	pthread_cond_init(&condBuffer, NULL);
	cout << "Operations List : " << endl << endl;
	helperFunction();
	cout << endl;
	cout << "logs : " << endl << endl;
	for (int i = 0; i < 4; ++i)
	{
		pthread_create(worker+i, NULL, &workerRoutine, NULL);
	}
	for(int i = 0; i<x; i++){
		struct Operation* tempOp;
		tempOp = opq.front();
		opq.pop();
		pthread_mutex_lock(&mutexJobBuffer);
		int enqRet = enqueue(tempOp);
		pthread_mutex_unlock(&mutexJobBuffer);
		if(enqRet == -1){
			cout << "Job queue size full, operation discarded! "  << endl; 
		}else{
			// job may be pushed at index 0 again and again as the worker is faster and lock mutex before manager does 
			// because manager has a function over call which is costly
			// We can sleep the worker for few milli seconds using sleep function before it locks mutex 
			// then multiple job will be pushed 
			cout << "Job pushed successfully at index " << jobq.size() <<  endl; 
		}
	}
	// cout << "Manager exits" << endl;
	for (int i = 0; i < 4; ++i)
	{
		pthread_join(worker[i], NULL);	
	}
	pthread_mutex_destroy(&mutexJobBuffer);
	pthread_mutex_destroy(&mutexKVStore);
	pthread_cond_destroy(&condBuffer);
	return 0;
}