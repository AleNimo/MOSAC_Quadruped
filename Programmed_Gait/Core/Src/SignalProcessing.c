#include "SignalProcessing.h"

spMedianFilter spCreateMedianFilter(unsigned int size)
{
	spMedianFilter pFilter = NULL;
	// Check if the requested size is valid
	if ((size == 0) || (size == ~0)) { return NULL; }
	// Allocate memory for the structure + flexible array member
	pFilter = (spMedianFilter)malloc(sizeof(struct spMedianFilter) + sizeof(spFilterElement) + size * sizeof(spFilterElement));
	if (pFilter == NULL) { return NULL; }
	// Initialize the structure
	pFilter->size = size;		// Buffer size
	pFilter->rIdx = 1;			// Mark the first element as the next to be replaced
	pFilter->mIdx = 0;			// Median index
	pFilter->state = 0;			// Cleared "full buffer" flag
	pFilter->pData[0].raw = 0;	// Initial median
	// Initialize each element as unused (circular reference in both links)
	size++;
	while (size > 0)
	{
		size--;
		pFilter->pData[size].pIdx = 0;
		pFilter->pData[size].nIdx = 0;
	}
	// Return the pointer to the filter
	return pFilter;
}

void spDestroyMedianFilter(spMedianFilter* ppFilter)
{
	// Release the allocated memory and return
	if ((ppFilter != NULL) && (*ppFilter != NULL))
	{
		free(*ppFilter);
		*ppFilter = NULL;
	}
	return;
}

SP_DATATYPE spMedianFilterMedian(spMedianFilter pFilter)
{
	return pFilter->pData[0].raw;
}

SP_DATATYPE spMedianFilterInsert(spMedianFilter pFilter, SP_DATATYPE value)
{
	unsigned int size;
	unsigned int idx = pFilter->mIdx;
	const unsigned int rIdx = pFilter->rIdx;
	spFilterElement* const pData = pFilter->pData;

	// Check if the next element is free (first link leads to itself)
	if (pFilter->state == 0)
	{ // Free element
		// Find the position in which the new element should be inserted
		if (value < pData[pFilter->mIdx].raw)
		{ // The new value is less than the median
			// Move backwards until the list element is less than the new element
			for (idx = pData[idx].pIdx; (idx != 0) && (value < pData[idx].raw); idx = pData[idx].pIdx);
			// Link the new element to its new adjacent elements
			pData[rIdx].pIdx = idx;
			pData[rIdx].nIdx = pData[idx].nIdx;
			pData[pData[rIdx].nIdx].pIdx = rIdx;
			pData[idx].nIdx = rIdx;
			// Decrement the median index if there will be an even number of elements
			if ((!(rIdx & 1)) || (rIdx == 1)) { pFilter->mIdx = pData[pFilter->mIdx].pIdx; }
		}
		else
		{ // The new value is greater than or equal to the median
			// Move forward until the list element is greater than or equal to the new element
			for (idx = pData[idx].nIdx; (idx != 0) && (value >= pData[idx].raw); idx = pData[idx].nIdx);
			// Link the new element to its new adjacent elements
			pData[rIdx].pIdx = pData[idx].pIdx;
			pData[rIdx].nIdx = idx;
			pData[pData[rIdx].pIdx].nIdx = rIdx;
			pData[idx].pIdx = rIdx;
			// Increment the median index if there will be an odd number of elements
			if (rIdx & 1) { pFilter->mIdx = pData[pFilter->mIdx].nIdx; }
		}
		// ┌──────┬──────┬────────┬─────────┬─────────────┬──────────────────────────────┬─────────────────────────┐
		// │ N    │ N'   │ Insert │ Median  │ Median'     │ Target Median                │ Correction              │
		// ├──────┼──────┼────────┼─────────┼─────────────┼──────────────────────────────┼─────────────────────────┤
		// │      │      │ Before │         │ (N+1)/2 + 1 │                              │ mIdx = pData[mIdx].pIdx │
		// │ Odd  │ Even ├────────┤ (N+1)/2 ├─────────────┤ N'/2 = (N+1)/2               ├─────────────────────────┤
		// │      │      │ After  │         │ (N+1)/2     │                              │ None                    │
		// ├──────┼──────┼────────┼─────────┼─────────────┼──────────────────────────────┼─────────────────────────┤
		// │      │      │ Before │         │ N/2 + 1     │                              │ None                    │
		// │ Even │ Odd  ├────────┤ N/2     ├─────────────┤ (N'+1)/2 = (N+2)/2 = N/2 + 1 ├─────────────────────────┤
		// │      │      │ After  │         │ N/2         │                              │ mIdx = pData[mIdx].nIdx │
		// └──────┴──────┴────────┴─────────┴─────────────┴──────────────────────────────┴─────────────────────────┘

		// The number of elements matches the replacement index
		size = rIdx;
	}
	else
	{ // Already in use, overwrite
		// Find the position in which the new element should be inserted
		if (value < pData[pFilter->mIdx].raw)
		{ // The new value is less than the old median
			// Move backwards until the list element is less than the new element
			for (idx = pData[idx].pIdx; (idx != 0) && (value < pData[idx].raw); idx = pData[idx].pIdx);
			// Temporarily link the new preceeding element, and move the median index backward if the replaced element used to be after the median or was the median
			pData[pData[idx].nIdx].pIdx = rIdx;
			if ((pData[rIdx].raw > pData[pFilter->mIdx].raw) || (rIdx == pFilter->mIdx)) { pFilter->mIdx = pData[pFilter->mIdx].pIdx; }
			pData[pData[idx].nIdx].pIdx = idx;
			// Unlink the replaced element from its adjacent elements
			pData[pData[rIdx].pIdx].nIdx = pData[rIdx].nIdx;
			pData[pData[rIdx].nIdx].pIdx = pData[rIdx].pIdx;
			// Link the new element to its new adjacent elements
			pData[rIdx].pIdx = pData[pData[idx].nIdx].pIdx;
			pData[rIdx].nIdx = pData[idx].nIdx;
			pData[pData[rIdx].nIdx].pIdx = rIdx;
			pData[pData[rIdx].pIdx].nIdx = rIdx;
		}
		else
		{ // The new value is greater than or equal to the median
			// Move forward until the list element is greater than or equal to the new element
			for (idx = pData[idx].nIdx; (idx != 0) && (value >= pData[idx].raw); idx = pData[idx].nIdx);
			// Temporarily link the new following element, and move the median index forward if the replaced element used to be before the median (older values equal to the median are placed before it)
			pData[pData[idx].pIdx].nIdx = rIdx;
			if (pData[rIdx].raw <= pData[pFilter->mIdx].raw) { pFilter->mIdx = pData[pFilter->mIdx].nIdx; }
			pData[pData[idx].pIdx].nIdx = idx;
			// Unlink the replaced element from its adjacent elements
			pData[pData[rIdx].pIdx].nIdx = pData[rIdx].nIdx;
			pData[pData[rIdx].nIdx].pIdx = pData[rIdx].pIdx;
			// Link the new element to its new adjacent elements
			pData[rIdx].pIdx = pData[idx].pIdx;
			pData[rIdx].nIdx = pData[pData[idx].pIdx].nIdx;
			pData[pData[rIdx].pIdx].nIdx = rIdx;
			pData[pData[rIdx].nIdx].pIdx = rIdx;
		}
		// The number of elements matches the full size
		size = pFilter->size;
	}

	// Fill the element
	pData[rIdx].raw = value;

	// Advance the replacement index
	if (rIdx == pFilter->size) { pFilter->rIdx = 1; pFilter->state = 1; }
	else { pFilter->rIdx++; }

	// Compute the median
	if (size & 1) // Odd number of elements (single value median)
	{
		pData[0].raw = pData[pFilter->mIdx].raw;
	}
	else // Even number of elements (two value average median)
	{
		pData[0].raw = ((pData[pFilter->mIdx].raw) + (pData[pData[pFilter->mIdx].nIdx].raw)) / 2;
	}

	// Return successfully
	return pData[0].raw;
}

#ifdef SP_SELF_TEST

#include <stdio.h>
#include <math.h>
#include <time.h>

//#define RAND_SEED 0
#define RAND_TEST_BUFFER_SIZE 10000
#define RAND_TEST_INSERTS 100000
#define RAND_TEST_MEAN 0.0				// Normal and uniform distributions
#define RAND_TEST_STD_DEV 1.0			// Normal distribution
#define RAND_TEST_RANGE 2.0				// Uniform distribution
#define TIMED_TEST_ITERATIONS 20		// Number of timed iterations
#define TIMED_TEST_BUFFER_SIZE 10000
#define TIMED_TEST_INSERTS 30000

static inline unsigned int checkMedianList(spMedianFilter pFilter);
unsigned int checkMedianList(spMedianFilter pFilter)
{
	unsigned int i = pFilter->pData[0].nIdx, j = 1, lastIdx = 0, lastEqualIdx = 0, cycled = 0;
	SP_DATATYPE lastVal = pFilter->pData[pFilter->pData[0].nIdx].raw;
	// Move through the list checking its order, backward linkage and median position
	while (i != pFilter->mIdx)
	{
		// List order (sorted by value and then by insertion order) and backward linkage checks
		if ((pFilter->pData[i].raw < lastVal) || (pFilter->pData[i].pIdx != lastIdx))
		{
			return 0;
		}
		if ((pFilter->pData[i].raw == lastVal) && (i < lastIdx))
		{
			if (cycled && (i > lastEqualIdx)) { return 0; }
			cycled = 1;
		}
		else { lastEqualIdx = lastIdx; cycled = 0; }
		// Median position count
		j++;
		// Update the variables for the next iteration
		lastVal = pFilter->pData[i].raw;
		lastIdx = i;
		i = pFilter->pData[i].nIdx;
	}
	while (i != 0)
	{
		// List order and backward linkage checks for the rest of the list
		if ((pFilter->pData[i].raw < lastVal) || (pFilter->pData[i].pIdx != lastIdx))
		{
			return 0;
		}
		if ((pFilter->pData[i].raw == lastVal) && (i < lastIdx))
		{
			if (cycled && (i > lastEqualIdx)) { return 0; }
			cycled = 1;
		}
		else { lastEqualIdx = lastIdx; cycled = 0; }
		// Update the variables for the next iteration
		lastVal = pFilter->pData[i].raw;
		lastIdx = i;
		i = pFilter->pData[i].nIdx;
	}
	if (pFilter->state & 1)
	{
		return j == (pFilter->size + 1) / 2;
	}
	return j == pFilter->rIdx / 2;
}

int main(void)
{
	spMedianFilter filter = NULL;
	unsigned int res = 1, i = 0, j = 0, cntLow = 0, cntHigh = 0;
	float val = 0.0;
	float* pBuffer = NULL;
	struct timespec tStart, tEnd;
	double meanTime = 0;

	printf("----------------------------------------------------\n");
	printf("Creation and Destruction:\n");
	filter = spCreateMedianFilter(1);
	printf("Creation:                       %20s\n", (filter != NULL) ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);
	printf("Destruction:                    %20s\n", (filter == NULL) ? "OK" : "ERROR");
	filter = spCreateMedianFilter(0);
	printf("Creation with invalid size:     %20s\n", (filter == NULL) ? "OK" : "ERROR");

	printf("----------------------------------------------------\n");
	printf("Single Value Buffer\n");
	filter = spCreateMedianFilter(1);
	res = spMedianFilterMedian(filter) == 0.0;
	printf("Empty buffer:                   %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 1.0) == 1.0;
	res &= checkMedianList(filter);
	printf("Single insert:                  %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterMedian(filter) == 1.0;
	res &= checkMedianList(filter);
	printf("Median function:                %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 2.0) == 2.0;
	res &= spMedianFilterMedian(filter) == 2.0;
	res &= checkMedianList(filter);
	printf("Full buffer insert (overwrite): %20s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);
	filter = spCreateMedianFilter(1);
	res = spMedianFilterInsert(filter, -1.0) == -1.0;
	res &= spMedianFilterMedian(filter) == -1.0;
	res &= checkMedianList(filter);
	printf("Negative Value Insert:          %20s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);

	printf("----------------------------------------------------\n");
	printf("Multivalue Buffer (Sequential Insertions)\n");
	filter = spCreateMedianFilter(3);
	res = spMedianFilterMedian(filter) == 0.0;
	printf("Empty buffer:                   %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 1.0) == 1.0;
	res &= spMedianFilterMedian(filter) == 1.0;
	res &= checkMedianList(filter);
	printf("First insert:                   %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 3.0) == 2.0;
	res &= spMedianFilterMedian(filter) == 2.0;
	res &= checkMedianList(filter);
	printf("Second insert:                  %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 5.0) == 3.0;
	res &= spMedianFilterMedian(filter) == 3.0;
	res &= checkMedianList(filter);
	printf("Third insert:                   %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 7.0) == 5.0;
	res &= spMedianFilterMedian(filter) == 5.0;
	res &= checkMedianList(filter);
	printf("Full buffer insert (overwrite): %20s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);
	filter = spCreateMedianFilter(4);
	res = 1;
	for (i = 0; i < 10; i++)
	{
		res &= spMedianFilterInsert(filter, 1.0) == 1.0;
		res &= spMedianFilterMedian(filter) == 1.0;
		res &= checkMedianList(filter);
	}
	res &= spMedianFilterInsert(filter, 0.0) == 1.0;
	res &= spMedianFilterMedian(filter) == 1.0;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 2.0) == 1.0;
	res &= spMedianFilterMedian(filter) == 1.0;
	res &= checkMedianList(filter);
	printf("Repeated Insert (same value):   %20s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);

	printf("----------------------------------------------------\n");
	printf("Multivalue Buffer (Non-sequential Insertions)\n");
	filter = spCreateMedianFilter(4);
	res = spMedianFilterInsert(filter, 1.0) == 1.0;
	res &= spMedianFilterMedian(filter) == 1.0;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.5) == (0.5 + 1.0) / 2;
	res &= spMedianFilterMedian(filter) == (0.5 + 1.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.0) == 0.5;
	res &= spMedianFilterMedian(filter) == 0.5;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, -1.0) == (0.0 + 0.5) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.5) / 2;
	res &= checkMedianList(filter);
	printf("Insert Before Median:           %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, 0.125) == (0.0 + 0.125) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.125) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.0) == (0.0 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, -0.125) == (-0.125 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (-0.125 + 0.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, -1.0) == (-0.125 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (-0.125 + 0.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, -2.0) == (-1.0 + -0.125) / 2;
	res &= spMedianFilterMedian(filter) == (-1.0 + -0.125) / 2;
	res &= checkMedianList(filter);
	printf("Insert Before Median (overwrite):%19s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);
	filter = spCreateMedianFilter(4);
	res = spMedianFilterInsert(filter, -1.0) == -1.0;
	res &= spMedianFilterMedian(filter) == -1.0;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, -0.5) == (-1.0 + -0.5) / 2;
	res &= spMedianFilterMedian(filter) == (-1.0 + -0.5) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.0) == -0.5;
	res &= spMedianFilterMedian(filter) == -0.5;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 1.0) == (-0.5 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (-0.5 + 0.0) / 2;
	res &= checkMedianList(filter);
	printf("Insert After Median:            %20s\n", res ? "OK" : "ERROR");
	res = spMedianFilterInsert(filter, -0.125) == (-0.125 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (-0.125 + 0.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.0) == (0.0 + 0.0) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.0) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 0.125) == (0.0 + 0.125) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.125) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 1.0) == (0.0 + 0.125) / 2;
	res &= spMedianFilterMedian(filter) == (0.0 + 0.125) / 2;
	res &= checkMedianList(filter);
	res &= spMedianFilterInsert(filter, 2.0) == (0.125 + 1.0) / 2;
	res &= spMedianFilterMedian(filter) == (0.125 + 1.0) / 2;
	res &= checkMedianList(filter);
	printf("Insert After Median (overwrite):%20s\n", res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);

	printf("----------------------------------------------------\n");
#ifdef RAND_SEED
	i = RAND_SEED;
#else
	i = ((unsigned int)time(NULL)) % 1000;
#endif
	srand(i);
	printf("Random Insertions (seed = %d)\n", i);
	pBuffer = (float*)malloc(sizeof(float) * RAND_TEST_BUFFER_SIZE);
	filter = spCreateMedianFilter(RAND_TEST_BUFFER_SIZE);
	res = 1;
	for (i = 0; res && (i < RAND_TEST_INSERTS); i++)
	{
		// Generate and store the random value
		val = (float)((((double)rand()) / RAND_MAX * RAND_TEST_RANGE) - (RAND_TEST_RANGE / 2) + RAND_TEST_MEAN);
		pBuffer[i % RAND_TEST_BUFFER_SIZE] = val;
		// Insert and verify the median function
		val = spMedianFilterInsert(filter, val);
		res &= spMedianFilterMedian(filter) == val;
		res &= checkMedianList(filter);
		// Check if the computed median matches the true median
		cntLow = 0; cntHigh = 0;
		j = min(i + 1, RAND_TEST_BUFFER_SIZE);
		while (j > 0) { j--; cntLow += pBuffer[j] < val; cntHigh += pBuffer[j] > val; }
		res &= (cntLow <= min(i + 1, RAND_TEST_BUFFER_SIZE) / 2) && (cntHigh <= min(i + 1, RAND_TEST_BUFFER_SIZE) / 2);
	}
	printf("Uniform Distribution (%6d/%6d)  %14s\n", i, RAND_TEST_INSERTS, res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);

	filter = spCreateMedianFilter(RAND_TEST_BUFFER_SIZE);
	res = 1;
	for (i = 0; res && (i < RAND_TEST_INSERTS); i++)
	{
		// Generate and store the random value (Box-Muller transform)
		val = (float)(RAND_TEST_MEAN + RAND_TEST_STD_DEV * (sqrt(-2.0 * log(((double)rand()) / RAND_MAX)) * cos(2 * 3.14159265358979323846 * (((double)rand()) / RAND_MAX))));
		pBuffer[i % RAND_TEST_BUFFER_SIZE] = val;
		// Insert and verify the median function
		val = spMedianFilterInsert(filter, val);
		res &= spMedianFilterMedian(filter) == val;
		res &= checkMedianList(filter);
		// Check if the computed median matches the true median
		cntLow = 0; cntHigh = 0;
		j = min(i + 1, RAND_TEST_BUFFER_SIZE);
		while (j > 0) { j--; cntLow += pBuffer[j] < val; cntHigh += pBuffer[j] > val; }
		res &= (cntLow <= min(i + 1, RAND_TEST_BUFFER_SIZE) / 2) && (cntHigh <= min(i + 1, RAND_TEST_BUFFER_SIZE) / 2);
	}
	printf("Normal Distribution (%6d/%6d)   %14s\n", i, RAND_TEST_INSERTS, res ? "OK" : "ERROR");
	spDestroyMedianFilter(&filter);
	free(pBuffer);

	printf("----------------------------------------------------\n");
	pBuffer = (float*)malloc(sizeof(float) * (TIMED_TEST_INSERTS+1));
	printf("%d Timed Random Insertions (filter size = %d)\n", TIMED_TEST_INSERTS, TIMED_TEST_BUFFER_SIZE);
	res = 1; meanTime = 0;
	for (i = TIMED_TEST_ITERATIONS; i > 0; i--)
	{
		filter = spCreateMedianFilter(TIMED_TEST_BUFFER_SIZE);
		// Generate and store the random values
		for (j = TIMED_TEST_INSERTS; j > 0; j--)
		{ pBuffer[j] = (float)((((double)rand()) / RAND_MAX * RAND_TEST_RANGE) - (RAND_TEST_RANGE / 2) + RAND_TEST_MEAN); }
		// Timed Insertions
		timespec_get(&tStart, TIME_UTC);
		for (j = TIMED_TEST_INSERTS; j > 0; j--)
		{ spMedianFilterInsert(filter, pBuffer[j]); }
		timespec_get(&tEnd, TIME_UTC);
		// Add the result to the mean time computation
		meanTime += (tEnd.tv_sec - tStart.tv_sec + (tEnd.tv_nsec - tStart.tv_nsec) * (double)1E-9) / TIMED_TEST_ITERATIONS;
		spDestroyMedianFilter(&filter);
	}
	printf("Uniform Distribution: Total %.2es / Insert %.2es\n", meanTime, meanTime/TIMED_TEST_INSERTS);
	res = 1; meanTime = 0;
	for (i = TIMED_TEST_ITERATIONS; i > 0; i--)
	{
		filter = spCreateMedianFilter(TIMED_TEST_BUFFER_SIZE);
		// Generate and store the random value (Box-Muller transform)
		for (j = TIMED_TEST_INSERTS; j > 0; j--)
		{ pBuffer[j] = (float)(RAND_TEST_MEAN + RAND_TEST_STD_DEV * (sqrt(-2.0 * log(((double)rand())/RAND_MAX)) * cos(2 * 3.14159265358979323846 * (((double)rand())/RAND_MAX)))); }
		// Timed Insertions
		timespec_get(&tStart, TIME_UTC);
		for (j = TIMED_TEST_INSERTS; j > 0; j--)
		{ spMedianFilterInsert(filter, pBuffer[j]); }
		timespec_get(&tEnd, TIME_UTC);
		// Add the result to the mean time computation
		meanTime += (tEnd.tv_sec - tStart.tv_sec + (tEnd.tv_nsec - tStart.tv_nsec) * (double)1E-9) / TIMED_TEST_ITERATIONS;
		spDestroyMedianFilter(&filter);
	}
	printf("Normal Distribution: Total %.2es / Insert %.2es\n", meanTime, meanTime/TIMED_TEST_INSERTS);
	free(pBuffer);

	printf("----------------------------------------------------\n");
	return 0;
}
#endif //!SP_SELF_TEST
