#include "pkeystore.h"
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define NUM_ELEMS 50
#define MAX_KEY (1 << 10)
#define MAX_VALUE (1 << 20)
#define NUM_OPERS 100
#define INVALID (-2)

/*Basic testcase to check the functionality with 
 * one set of insert, update, delete, find operation*/

// this variable is used to wait till worker threads finish work,
// decrement this variable when worker thread do insert/update/delete/find operations
// so that the while(num_enqueue>0) at the end exits when workers finish operations.
int num_enqueue = 4; 
int main(int argc, char *argv[]) {
  struct operation ops[4];
  int pos = 0;
  int jobloc = 0;
  uint64_t val;
  for (int i = 0; i < 4; i++) {
    ops[i].type = i;
    ops[i].key = 966;
    ops[i].value = 825449;
    if( i == 1 ) {
      ops[i].value = 9978;
    }
  }
  for (int i = 0; i < 4; i++) {
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done\n");
  return EXIT_SUCCESS;
}
