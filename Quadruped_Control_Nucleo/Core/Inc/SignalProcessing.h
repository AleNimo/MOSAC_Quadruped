#ifndef SIGNALPROCESSING_H
#define SIGNALPROCESSING_H

#include <stdlib.h>

// The datatype of the variables stored in the filter
#define SP_DATATYPE float

typedef struct spMedianFilter* spMedianFilter;

// Creates a median filter with a data buffer of the requested size.
spMedianFilter spCreateMedianFilter(unsigned int size);
// Destroys the median filter, freeing all resources assigned to it
void spDestroyMedianFilter(spMedianFilter* pFilter);
// Returns the current median of the filter's data
SP_DATATYPE spMedianFilterMedian(spMedianFilter filter);
// Inserts a new value in the filter's buffer. If the buffer is full the oldest preexisting entry is overwritten.
SP_DATATYPE spMedianFilterInsert(spMedianFilter filter, SP_DATATYPE value);




//#define SP_SELF_TEST

struct spFilterElement
{
	unsigned int nIdx;	// Index of the next element
	unsigned int pIdx;	// Index of the previous element
	SP_DATATYPE  raw;	// Exact value (without casting)
};
typedef struct spFilterElement spFilterElement;

struct spMedianFilter
{
	unsigned int state;			// Flag indicating whether the buffer still has free space or if the oldest entry should be overwritten
	unsigned int rIdx;			// The index of the next element to be replaced
	unsigned int mIdx;			// The index of the median element (Keeps track of the middle of the double linked list)
	unsigned int size;			// Maximum number of entries in the data buffer
	spFilterElement pData[];	// Flexible array that contains the data used by the filter
};
// Structures with a flexible array member (introduced in C99) have an undefined length array as the structure's last member.
// The structure's memory must be allocated as: malloc(sizeof(struct) + flexible array element count * sizeof(flexible array element)).
// NOTE: Additional unused memory may be allocated for structure padding.



#endif
