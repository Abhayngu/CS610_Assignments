#include "pkeystore.hpp"
#include<bits/stdc++.h>
using namespace std;

#define NUM_OPERS 100

/*Test case for insert, update, delete, update a key
 * here last update will fail as key is deleted*/
int num_enqueue = NUM_OPERS; // this variable is used to wait till worker threads finish work
int main(int argc, char *argv[]) {
  struct operation ops[NUM_OPERS];
  int pos = 0;
  int jobloc = 0;
  uint64_t val;
  num_enqueue = NUM_OPERS;
  for (int i = 0; i < NUM_OPERS-1; i++){
    ops[i].type = 0;
    ops[i].key = i;
    ops[i].value = i*100;
    if(i%13 == 0){
        ops[i].type = 2;
        ops[i].key = 5;
        ops[i].value = 500; 
    }
    if(i % 17 == 0){
        ops[i].type = 0;
        ops[i].key = 5;
        ops[i].value = 500;
    }
  }
  ops[NUM_OPERS-1].type = 0;
  ops[NUM_OPERS-1].key = 5;
  ops[NUM_OPERS-1].value = 500;

  for (int i = 0; i < NUM_OPERS; i++) {
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-1\n");
//   printStore();
//   cout << isPresent(5);
  return EXIT_SUCCESS;
}
