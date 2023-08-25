#include "pkeystore.h"
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define NUM_OPERS 10000

/*Checking proper job queue handling by issuing
 * large number of operations*/

// this variable is used to wait till worker threads finish work,
// decrement this variable when worker thread do insert/update/delete/find operations
// so that the while(num_enqueue>0) exits when workers finish operations.
int num_enqueue = NUM_OPERS;

int main(int argc, char *argv[]) {
  struct operation ops[NUM_OPERS];
  int pos = 0;
  int jobloc = 0;
  uint64_t val;
  num_enqueue = NUM_OPERS;
  for (int i = 0; i < NUM_OPERS; i++){
    ops[i].type = 0;
    ops[i].key = i*100;
    ops[i].value = i*1000;
  }
  for (int i = 0; i < NUM_OPERS; i++) {
    jobloc = enqueue(&ops[i]);
    //printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-1\n");

  num_enqueue = NUM_OPERS;
  //update even keys and delete odd keys
  for (int i = 0; i < NUM_OPERS; i++){
    if(i%4 == 2){
      ops[i].type = 0;
    }
    else{
      ops[i].type = i%4;
    }
    ops[i].key = i*100;
    ops[i].value = i*1000;
  }

  for (int i = 0; i < NUM_OPERS; i++){
    jobloc = enqueue(&ops[i]);
    //printf("job enqueued at loc:%d\n",jobloc);
  }

  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-2\n");
  
  num_enqueue = NUM_OPERS;
  //update even keys and delete odd keys
  for (int i = 0; i < NUM_OPERS; i++){
    if(i%4 == 2){
      ops[i].type = 0;
    }
    else{
      ops[i].type = i%4;
    }
    ops[i].key = i*100;
    ops[i].value = i*1000;
  }

  for (int i = 0; i < NUM_OPERS; i++){
    jobloc = enqueue(&ops[i]);
    //printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-3\n");
  return EXIT_SUCCESS;
}
