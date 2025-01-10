#ifndef MEDIAN_H
#define MEDIAN_H
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <arm_math.h>

typedef struct Median 
{
    volatile float32_t* buffer;
    volatile float32_t** ptBufferSorted;
    uint8_t size;
    volatile uint8_t index;
    volatile uint8_t iterationCount;
    volatile uint8_t initialKnuthGap;
    bool init;
} median;

extern void MedianInit(median* this, volatile float32_t* buffer, volatile float32_t** ptBufferSorted, uint16_t size);
extern float32_t MedianFilter(median* this, volatile float32_t input);
extern uint8_t MedianIterationGet(median* this);

#endif
