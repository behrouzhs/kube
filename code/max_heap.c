#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "max_heap.h"


MaxHeap* maxheap_create(int capacity)
{
	// we don't use index 0 for easier indexing
	int i;
	++capacity;
	MaxHeap *heap = (MaxHeap*)malloc(sizeof(MaxHeap));
	if (heap == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	heap->elements = (MaxHeapItem**)malloc(sizeof(MaxHeapItem*) * capacity);
	if (heap->elements == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < capacity; ++i)
	{
		heap->elements[i] = (MaxHeapItem*)malloc(sizeof(MaxHeapItem));
		if (heap->elements[i] == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	}

	heap->elements[0]->dist = DBL_MAX;
	heap->no_items = 0;
	return heap;
}


MaxHeap** maxheap_create_multi(int no_heap, int capacity)
{
	int i;
	MaxHeap **heap = (MaxHeap**)malloc(sizeof(MaxHeap*) * no_heap);
	if (heap == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < no_heap; ++i)
		heap[i] = maxheap_create(capacity);

	return heap;
}


void maxheap_reset(MaxHeap *heap)
{
	heap->no_items = 0;
}


void maxheap_destroy(MaxHeap *heap)
{
	free(heap->elements); heap->elements = NULL;
	free(heap); heap = NULL;
}


void maxheap_destroy_multi(MaxHeap **heap, int no_heap)
{
	int i;
	for (i = 0; i < no_heap; ++i)
	{
		free(heap[i]->elements); heap[i]->elements = NULL;
		free(heap[i]); heap[i] = NULL;
	}
	free(heap); heap = NULL;
}


void maxheap_push(MaxHeap *heap, double dist, int index)
{
	++(heap->no_items);
	int now = heap->no_items;
	int parent = now / 2;
	while (heap->elements[parent]->dist < dist)
	{
		*heap->elements[now] = *heap->elements[parent];
		now = parent;
		parent = now / 2;
	}
	heap->elements[now]->dist = dist;
	heap->elements[now]->index = index;
}


void maxheap_pop(MaxHeap *heap, double *dist, int *index)
{
	double lastElement;
	int child, now, lastIndex;
	*dist = heap->elements[1]->dist;
	*index = heap->elements[1]->index;
	lastElement = heap->elements[heap->no_items]->dist;
	lastIndex = heap->elements[heap->no_items]->index;
	--(heap->no_items);

	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (lastElement < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = lastElement;
	heap->elements[now]->index = lastIndex;
}


void maxheap_pop_discard(MaxHeap *heap)
{
	double lastElement;
	int child, now, lastIndex;
	lastElement = heap->elements[heap->no_items]->dist;
	lastIndex = heap->elements[heap->no_items]->index;
	--(heap->no_items);

	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (lastElement < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = lastElement;
	heap->elements[now]->index = lastIndex;
}

void maxheap_pop_push(MaxHeap *heap, double dist, int index)
{
	int child, now;
	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (dist < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = dist;
	heap->elements[now]->index = index;
}
