#include "pkeystore.hpp"
#include<bits/stdc++.h>
using namespace std;

#define NUM_OPERS 50

/*Test case for insert, update, delete, update a key
 * here last update will fail as key is deleted*/
int num_enqueue = NUM_OPERS; // this variable is used to wait till worker threads finish work
int main(int argc, char *argv[]) {
  struct operation ops[NUM_OPERS];
  int pos = 0;
  int jobloc = 0;
  uint64_t val;
  num_enqueue = NUM_OPERS;
  for(int i=0; i < NUM_OPERS; i++){
    
	ops[i].type = 3-(i%4);
	ops[i].key = i%5;
	ops[i].value = 825449%5;
    }

  for (int i = 0; i < NUM_OPERS; i++) {
    jobloc = enqueue(&ops[i]);
    printf("job enqueued at loc:%d\n",jobloc);
  }
  printf("num_enqueue:%d\n",num_enqueue);
  while(num_enqueue>0);
  printf("done-1\n");
  printStore();
//   cout << isPresent(5);
  return EXIT_SUCCESS;
}
