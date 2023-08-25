#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
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
/*Test case for insert, update, update a key
 * here update are applied and find gets correct value*/
int main(int argc, char* argv[]){
    struct operation ops[3];
    int pos = 0;
    struct jobdetails jd[3];
    uint64_t val;
    int j = 0;
    ops[0].type = 0;
    ops[0].key = 508;
    ops[0].value = 467815;
    ops[1].type = 1;
    ops[1].key = 508;
    ops[1].value = 427515;
    ops[2].type = 1;
    ops[2].key = 508;
    ops[2].value = 98874;

    for(int i=0; i<3; i++){
	jd[i].valid = 1;
	jd[i].jobloc = enqueue(&ops[i]);
	jd[i].op = &ops[i];
    }
    for(int i=0; i<3; i++){
	assert(jd[i].jobloc >= 0); //ensuring enqueue did not fail
	jobstatus(jd[i]);
	jd[i].valid = 0;

    }
    return 0;
}
