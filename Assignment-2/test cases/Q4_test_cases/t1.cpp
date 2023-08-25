#include "pkeystore.h"
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define NUM_OPERS 10

/*Check correctness of key-value store, issuing different 
 * sets of insert, update, delete and then finally find operation.*/

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
  for (int i = 0; i < NUM_OPERS; i++) {
    ops[i].type = 0;
    ops[i].key = i*100;
    ops[i].value = i*1000;
  }
  for (int i = 0; i < NUM_OPERS; i++) {
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-1\n");

  num_enqueue = NUM_OPERS;
  //update even keys and delete odd keys
  for (int i = 0; i < NUM_OPERS; i++){
    if( i%2 == 0 ){
      ops[i].type = 1;
      ops[i].key = i*100;
      ops[i].value = i*1000;
    }
    else{
      ops[i].type = 2;
      ops[i].key = i*100;
      ops[i].value = i*1000;
    }
  }
  for (int i = 0; i < NUM_OPERS; i++){
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }

  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-2\n");
  
  num_enqueue = 2;
  for (int i = 0; i < 2; i++) {
    ops[i].type = 3;
    ops[i].key = i*100;
    ops[i].value = i*1000;
  }
  for (int i = 0; i < 2; i++){
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-3\n");
  return EXIT_SUCCESS;
}
