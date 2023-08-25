/*Job Queue, manager, worker thread creation and responsibilities as mentioned in assignment are implemented here*/
#include "pkeystore.h"
#include <stdio.h>

int enqueue(struct operation *op){
    printf("called enqueue\n");
    return 0;
}

int check_status(int jobloc){
    printf("called check status\n");
    return 1;
}

uint64_t find(uint32_t key){
    printf("called find\n");
    return 1;
}
