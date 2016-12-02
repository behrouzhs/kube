#pragma once

#ifndef MAX_HEAP_H
#define MAX_HEAP_H

typedef struct _MaxHeapItem
{
	double dist;
	int index;
}MaxHeapItem;


typedef struct _MaxHeap
{
	int no_items;
	MaxHeapItem **elements;
}MaxHeap;

MaxHeap *maxheap_create(int capacity);
MaxHeap **maxheap_create_multi(int no_heap, int capacity);
void maxheap_reset(MaxHeap *heap);
void maxheap_destroy(MaxHeap *heap);
void maxheap_destroy_multi(MaxHeap **heap, int no_heap);
void maxheap_push(MaxHeap *heap, double dist, int index);
void maxheap_pop(MaxHeap *heap, double* dist, int* index);
void maxheap_pop_discard(MaxHeap *heap);
void maxheap_pop_push(MaxHeap *heap, double dist, int index);

#endif