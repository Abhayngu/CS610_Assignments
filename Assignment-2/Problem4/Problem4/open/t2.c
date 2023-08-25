#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include "pkeystore.h"

#define NUM_ELEMS 50
#define MAX_KEY (1<<10)
#define MAX_VALUE (1<<20)
#define NUM_OPERS 100
#define INVALID (-2)

struct jobdetails{
    int valid; // = 1 means the job status is not yet checked
    int jobloc;
    struct operation* op; 
};
void calculate(uint64_t value){
    int i = 0;
    while(value>>1){
        i+=1;
	value = value>>1;
    }
    printf("i: %d\n",i);
}
void jobstatus(struct jobdetails jd){
    uint64_t val; 
    if(!check_status(jd.jobloc)){
        //insert failed, inserts should not fail. 
        if((jd.op)->type == 0){
            printf("insert failed for key:%u\n",(jd.op)->key);
	}
        //update failed, if the update is to an existing key, then it is an issue with the implementation
        //update to non-existing key failure is an expected behavior
        else if((jd.op)->type == 1){
	    printf("update failed for key:%u\n",(jd.op)->key);
	}
        //delete failed, if the delete on an existing key, then it is an issue with the implementation
        //delete on non-existing key failure is an expected behavior
        else if((jd.op)->type == 2){
	    printf("delete failed for key:%u\n",(jd.op)->key);
	}
    }
    else{
        //enqueued operation is done, now compute using value
        val = find((jd.op)->key);
	if(val != -1)
	    calculate(val);
    }
    return;
}

int main(int argc, char* argv[]){
    struct operation * ops = (struct operation*)malloc(NUM_ELEMS*sizeof(struct operation));
    int fd = open("operation_2.data",O_RDWR|O_CREAT,S_IRUSR|S_IWUSR);
    int pos = 0;
    struct jobdetails jd[NUM_OPERS];
    uint64_t val;
    int j = 0;
    if( fd< 0){
        perror("file open:");
	exit(-1);
    }
    srand(0);
   /*insert to a already existing key in key-value store is equvalent to update, as key is unique in key-value store*/
    for(int i=0; i<NUM_ELEMS; i++){
        ops[i].type = rand()%3;
	ops[i].key = rand()%MAX_KEY;
	ops[i].value = rand()%MAX_VALUE;
	//write list of operations to file for verification.
	if(write(fd,&ops[i],sizeof(struct operation))<0){
	    perror("write");
	    exit(-1);
	}
    }
    //enqueue each operation in ops list, receive the jobqueue location in jd array.
    srand(9);
    for(int i=0; i<NUM_OPERS; i++){
	jd[i].valid = 0;
	jd[i].jobloc = INVALID;
	//select between enqueue and check status of enqueued operation.
	if(rand()%100 < 30){
            jd[i].valid = 1;
	    jd[i].jobloc = enqueue(&ops[pos]);
	    jd[i].op = &ops[pos];
	    pos = (pos+1)%NUM_ELEMS;
	}
	else{
            j = 0;
	    while(j<i){
                //enqueue failed, we retry it once.
                if(jd[j].jobloc == -1){
		    jd[j].valid = 1;
                    jd[j].jobloc = enqueue(jd[j].op);
		}
		else{
                    //this job status is yet to be checked
		    if(jd[j].valid){
                        jobstatus(jd[j]);
			jd[j].valid = 0;
		    }
		}
		j++;
	    }
	}
    }
    free(ops);
    return 0;
}
