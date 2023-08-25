#ifndef __PKEY__FILE_H
#define __PKEY__FILE_H

#include <stdint.h>

struct operation{
    uint8_t type;
    uint32_t key;
    uint64_t value;
};

int enqueue(struct operation *op);
int check_status(int jobloc);
uint64_t find(uint32_t key);

#endif
