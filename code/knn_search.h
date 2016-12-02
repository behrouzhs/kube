#pragma once

#ifndef KNN_SEARCH_H
#define KNN_SEARCH_H

#include "max_heap.h"

// sequential functions
void fill_knn_from_heap_euclidean(MaxHeap **heap, int no_point, int knn, int *out_index, double *out_dist);
void fill_knn_from_heap_cosine(MaxHeap **heap, int no_point, int knn, int *out_index, double *out_dist);
void calculate_sqlengths(double *X, int no_point, int no_dim, double *out_len);
void knn_search(double *X, int no_point, int no_dim, int knn, int *out_index, double *out_dist);
void knn_search_sparse_euclidean(int *col, double *val, int no_point, int *start_indices, int knn, int *out_index, double *out_dist);
void knn_search_sparse_cosine(int *col, double *val, int no_point, int *start_indices, int knn, int *out_index, double *out_dist);

// parallel functions
void fill_knn_from_heap_euclidean_omp(MaxHeap **heap, int no_point, int knn, int no_thread, int *out_index, double *out_dist);
void fill_knn_from_heap_cosine_omp(MaxHeap **heap, int no_point, int knn, int no_thread, int *out_index, double *out_dist);
void calculate_sqlengths_omp(double *X, int no_point, int no_dim, int no_thread, double *out_len);
void knn_search_omp(double *X, int no_point, int no_dim, int knn, int no_thread, int *out_index, double *out_dist);
void knn_search_sparse_euclidean_omp(int *col, double *val, int no_point, int *start_indices, int knn, int no_thread, int *out_index, double *out_dist);
void knn_search_sparse_cosine_omp(int *col, double *val, int no_point, int *start_indices, int knn, int no_thread, int *out_index, double *out_dist);

typedef struct _ThreadQueue
{
	int size;
	double *dist;
	int *heap_index;
	int *nn_index;
	int cnt;
}ThreadQueue;

ThreadQueue* thread_queue_create(int capacity);
ThreadQueue** thread_queue_create_multi(int no_queue, int capacity, int no_thread);
void thread_queue_destroy_multi(ThreadQueue **queue, int no_queue, int no_thread);
void thread_queue_insert(ThreadQueue *queue, int heap_idx, double distance, int nn_idx);
void thread_queue_insert_fixed_sized(ThreadQueue *queue, int heap_idx, double distance, int nn_idx);

#endif
