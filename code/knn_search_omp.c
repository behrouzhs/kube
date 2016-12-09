#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "max_heap.h"
#include "knn_search.h"
#include "tools_portable.h"


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
// faster way of calculating euclidean distances: |x-y|^2 = x^2 + y^2 - 2xy
void knn_search_omp(double *X, int no_point, int no_dim, int knn, int no_thread, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	ThreadQueue **queue = thread_queue_create_multi(no_thread, no_point, no_thread);
	ThreadQueue **queue_self = thread_queue_create_multi(no_thread, no_point, no_thread);
	LOCK_MUTEX *lock = lock_create_multi(no_thread);
	
	double d_temp, dist;
	int i, j, d, n_idx1, n_idx2, n_temp;

	double *len = (double*)malloc(sizeof(double) * no_point);
	if (len == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	calculate_sqlengths_omp(X, no_point, no_dim, no_thread, len);

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
		#pragma omp master
		no_thread = omp_get_num_threads();
	}
#pragma omp parallel default(none) private(i, j, d, n_idx1, n_idx2, d_temp, n_temp, dist) shared(X, no_point, no_dim, knn, len, heap, no_thread, lock, queue, queue_self)
	{
		int tid = omp_get_thread_num();
		for (i = tid; i < no_point; i += no_thread)
		{
			n_idx1 = i * no_dim;
			for (j = i + 1; j < no_point; ++j)
			{
				d_temp = 0;
				n_idx2 = j * no_dim;
				for (d = 0; d < no_dim; ++d)
					d_temp += X[n_idx1 + d] * X[n_idx2 + d];
				dist = len[i] + len[j] - (2 * d_temp);

				if (heap[i]->no_items < knn)
					maxheap_push(heap[i], dist, j);
				else if (dist < heap[i]->elements[1]->dist)
					maxheap_pop_push(heap[i], dist, j);
				
				n_temp = j % no_thread;
				if (n_temp == tid)
				{
					if (heap[j]->no_items < knn)
						maxheap_push(heap[j], dist, i);
					else if (dist < heap[j]->elements[1]->dist)
						maxheap_pop_push(heap[j], dist, i);
				}
				else
				{
					if (lock_try_acquire(&lock[n_temp]) == 0)
					{
						thread_queue_insert(queue[n_temp], j, dist, i);
						lock_release(&lock[n_temp]);
					}
					else
						thread_queue_insert_fixed_sized(queue_self[tid], j, dist, i);
				}
			}

			while (queue_self[tid]->cnt > 0)
			{
				--(queue_self[tid]->cnt);
				n_temp = (queue_self[tid]->heap_index[queue_self[tid]->cnt]) % no_thread;
				lock_acquire(&lock[n_temp]);
				thread_queue_insert(queue[n_temp], queue_self[tid]->heap_index[queue_self[tid]->cnt], queue_self[tid]->dist[queue_self[tid]->cnt], queue_self[tid]->nn_index[queue_self[tid]->cnt]);
				lock_release(&lock[n_temp]);
			}

			lock_acquire(&lock[tid]);
			while (queue[tid]->cnt > 0)
			{
				--(queue[tid]->cnt);
				n_temp = queue[tid]->heap_index[queue[tid]->cnt];
				if (heap[n_temp]->no_items < knn)
					maxheap_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
				else if (queue[tid]->dist[queue[tid]->cnt] < heap[n_temp]->elements[1]->dist)
					maxheap_pop_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
			}
			lock_release(&lock[tid]);
		}
	}

	fill_knn_from_heap_euclidean_omp(heap, no_point, knn, no_thread, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
	lock_destroy_multi(lock, no_thread);
	thread_queue_destroy_multi(queue, no_thread, no_thread);
	thread_queue_destroy_multi(queue_self, no_thread, no_thread);
	free(len); len = NULL;
}


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
void knn_search_sparse_euclidean_omp(int *col, double *val, int no_point, int *start_indices, int knn, int no_thread, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	ThreadQueue **queue = thread_queue_create_multi(no_thread, no_point, no_thread);
	ThreadQueue **queue_self = thread_queue_create_multi(no_thread, no_point, no_thread);
	LOCK_MUTEX *lock = lock_create_multi(no_thread);

	double dist, d_temp;
	int i, j, ix1, ix2, n_max1, n_max2, n_temp;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
		#pragma omp master
		no_thread = omp_get_num_threads();
	}
#pragma omp parallel default(none) private(i, j, ix1, ix2, n_max1, n_max2, n_temp, d_temp, dist) shared(col, val, no_point, knn, start_indices, no_thread, heap, lock, queue, queue_self)
	{
		int tid = omp_get_thread_num();
		for (i = tid; i < no_point; i += no_thread)
		{
			n_max1 = start_indices[i + 1];
			for (j = i + 1; j < no_point; ++j)
			{
				dist = 0;
				ix1 = start_indices[i];
				ix2 = start_indices[j];
				n_max2 = start_indices[j + 1];
				while (ix1 < n_max1 && ix2 < n_max2)
				{
					if (col[ix1] < col[ix2])
					{
						dist += val[ix1] * val[ix1];
						++ix1;
					}
					else if (col[ix2] < col[ix1])
					{
						dist += val[ix2] * val[ix2];
						++ix2;
					}
					else
					{
						d_temp = val[ix1] - val[ix2];
						dist += d_temp * d_temp;
						++ix1;
						++ix2;
					}
				}
				for (; ix1 < n_max1; ++ix1)
					dist += val[ix1] * val[ix1];
				for (; ix2 < n_max2; ++ix2)
					dist += val[ix2] * val[ix2];

				if (heap[i]->no_items < knn)
					maxheap_push(heap[i], dist, j);
				else if (dist < heap[i]->elements[1]->dist)
					maxheap_pop_push(heap[i], dist, j);

				n_temp = j % no_thread;
				if (n_temp == tid)
				{
					if (heap[j]->no_items < knn)
						maxheap_push(heap[j], dist, i);
					else if (dist < heap[j]->elements[1]->dist)
						maxheap_pop_push(heap[j], dist, i);
				}
				else
				{
					if (lock_try_acquire(&lock[n_temp]) == 0)
					{
						thread_queue_insert(queue[n_temp], j, dist, i);
						lock_release(&lock[n_temp]);
					}
					else
						thread_queue_insert_fixed_sized(queue_self[tid], j, dist, i);
				}
			}

			while (queue_self[tid]->cnt > 0)
			{
				--(queue_self[tid]->cnt);
				n_temp = (queue_self[tid]->heap_index[queue_self[tid]->cnt]) % no_thread;
				lock_acquire(&lock[n_temp]);
				thread_queue_insert(queue[n_temp], queue_self[tid]->heap_index[queue_self[tid]->cnt], queue_self[tid]->dist[queue_self[tid]->cnt], queue_self[tid]->nn_index[queue_self[tid]->cnt]);
				lock_release(&lock[n_temp]);
			}

			lock_acquire(&lock[tid]);
			while (queue[tid]->cnt > 0)
			{
				--(queue[tid]->cnt);
				n_temp = queue[tid]->heap_index[queue[tid]->cnt];
				if (heap[n_temp]->no_items < knn)
					maxheap_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
				else if (queue[tid]->dist[queue[tid]->cnt] < heap[n_temp]->elements[1]->dist)
					maxheap_pop_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
			}
			lock_release(&lock[tid]);
		}
	}

	fill_knn_from_heap_euclidean_omp(heap, no_point, knn, no_thread, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
	lock_destroy_multi(lock, no_thread);
	thread_queue_destroy_multi(queue, no_thread, no_thread);
	thread_queue_destroy_multi(queue_self, no_thread, no_thread);
}


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
void knn_search_sparse_cosine_omp(int *col, double *val, int no_point, int *start_indices, int knn, int no_thread, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	ThreadQueue **queue = thread_queue_create_multi(no_thread, no_point, no_thread);
	ThreadQueue **queue_self = thread_queue_create_multi(no_thread, no_point, no_thread);
	LOCK_MUTEX *lock = lock_create_multi(no_thread);

	double dist;
	int i, j, ix1, ix2, n_max1, n_max2, n_temp;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
		#pragma omp master
		no_thread = omp_get_num_threads();
	}
#pragma omp parallel default(none) private(i, j, ix1, ix2, n_max1, n_max2, n_temp, dist) shared(col, val, no_point, knn, start_indices, heap, no_thread, lock, queue, queue_self)
	{
		int tid = omp_get_thread_num();
		for (i = tid; i < no_point; i += no_thread)
		{
			n_max1 = start_indices[i + 1];
			for (j = i + 1; j < no_point; ++j)
			{
				dist = 0;
				ix1 = start_indices[i];
				ix2 = start_indices[j];
				n_max2 = start_indices[j + 1];
				while (ix1 < n_max1 && ix2 < n_max2)
				{
					if (col[ix1] < col[ix2])
						++ix1;
					else if (col[ix2] < col[ix1])
						++ix2;
					else
					{
						dist += val[ix1] * val[ix2];
						++ix1;
						++ix2;
					}
				}
				dist = 1 - dist;

				if (heap[i]->no_items < knn)
					maxheap_push(heap[i], dist, j);
				else if (dist < heap[i]->elements[1]->dist)
					maxheap_pop_push(heap[i], dist, j);

				n_temp = j % no_thread;
				if (n_temp == tid)
				{
					if (heap[j]->no_items < knn)
						maxheap_push(heap[j], dist, i);
					else if (dist < heap[j]->elements[1]->dist)
						maxheap_pop_push(heap[j], dist, i);
				}
				else
				{
					if (lock_try_acquire(&lock[n_temp]) == 0)
					{
						thread_queue_insert(queue[n_temp], j, dist, i);
						lock_release(&lock[n_temp]);
					}
					else
						thread_queue_insert_fixed_sized(queue_self[tid], j, dist, i);
				}
			}

			while (queue_self[tid]->cnt > 0)
			{
				--(queue_self[tid]->cnt);
				n_temp = (queue_self[tid]->heap_index[queue_self[tid]->cnt]) % no_thread;
				lock_acquire(&lock[n_temp]);
				thread_queue_insert(queue[n_temp], queue_self[tid]->heap_index[queue_self[tid]->cnt], queue_self[tid]->dist[queue_self[tid]->cnt], queue_self[tid]->nn_index[queue_self[tid]->cnt]);
				lock_release(&lock[n_temp]);
			}

			lock_acquire(&lock[tid]);
			while (queue[tid]->cnt > 0)
			{
				--(queue[tid]->cnt);
				n_temp = queue[tid]->heap_index[queue[tid]->cnt];
				if (heap[n_temp]->no_items < knn)
					maxheap_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
				else if (queue[tid]->dist[queue[tid]->cnt] < heap[n_temp]->elements[1]->dist)
					maxheap_pop_push(heap[n_temp], queue[tid]->dist[queue[tid]->cnt], queue[tid]->nn_index[queue[tid]->cnt]);
			}
			lock_release(&lock[tid]);
		}
	}

	fill_knn_from_heap_cosine_omp(heap, no_point, knn, no_thread, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
	lock_destroy_multi(lock, no_thread);
	thread_queue_destroy_multi(queue, no_thread, no_thread);
	thread_queue_destroy_multi(queue_self, no_thread, no_thread);
}


void fill_knn_from_heap_euclidean_omp(MaxHeap **heap, int no_point, int knn, int no_thread, int *out_index, double *out_dist)
{
	int i, j, n_temp;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, j, n_temp) shared(no_point, knn, heap, out_index, out_dist)
	{
		#pragma omp for
		for (i = 0; i < no_point; ++i)
		{
			for (j = 0; j < knn; ++j)
			{
				n_temp = (i * knn) + knn - j - 1;
				maxheap_pop(heap[i], &out_dist[n_temp], &out_index[n_temp]);
				if (out_dist[n_temp] < 0)
					out_dist[n_temp] = 0;
				else
					out_dist[n_temp] = sqrt(out_dist[n_temp]);
			}
		}
	}
}


void fill_knn_from_heap_cosine_omp(MaxHeap **heap, int no_point, int knn, int no_thread, int *out_index, double *out_dist)
{
	int i, j, n_temp;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, j, n_temp) shared(no_point, knn, heap, out_index, out_dist)
	{
		#pragma omp for
		for (i = 0; i < no_point; ++i)
		{
			for (j = 0; j < knn; ++j)
			{
				n_temp = (i * knn) + knn - j - 1;
				maxheap_pop(heap[i], &out_dist[n_temp], &out_index[n_temp]);
				//out_dist[n_temp] = 1 - out_dist[n_temp];
			}
		}
	}
}


void calculate_sqlengths_omp(double *X, int no_point, int no_dim, int no_thread, double *out_len)
{
	int i, j;
	int n_ind, idx;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, j, n_ind, idx) shared(X, no_point, no_dim, out_len)
	{
		#pragma omp for
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			out_len[i] = 0;
			for (j = 0; j < no_dim; ++j)
			{
				idx = n_ind + j;
				out_len[i] += X[idx] * X[idx];
			}
		}
	}
}


ThreadQueue* thread_queue_create(int capacity)
{
	ThreadQueue *queue = (ThreadQueue*)malloc(sizeof(ThreadQueue));
	if (queue == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	queue->dist = (double*)malloc(sizeof(double) * capacity);
	queue->heap_index = (int*)malloc(sizeof(int) * capacity);
	queue->nn_index = (int*)malloc(sizeof(int) * capacity);
	if (queue->dist == NULL || queue->heap_index == NULL || queue->nn_index == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	queue->size = capacity;
	queue->cnt = 0;
	return queue;
}


ThreadQueue** thread_queue_create_multi(int no_queue, int capacity, int no_thread)
{
	int i;
	ThreadQueue **queue = (ThreadQueue**)malloc(sizeof(ThreadQueue*) * no_queue);
	if (queue == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i) shared(no_queue, capacity, queue)
	{
		#pragma omp for
		for (i = 0; i < no_queue; ++i)
			queue[i] = thread_queue_create(capacity);
	}

	return queue;
}


void thread_queue_destroy_multi(ThreadQueue **queue, int no_queue, int no_thread)
{
	int i;
	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i) shared(no_queue, queue)
	{
		#pragma omp for
		for (i = 0; i < no_queue; ++i)
		{
			free(queue[i]->dist); queue[i]->dist = NULL;
			free(queue[i]->heap_index); queue[i]->heap_index = NULL;
			free(queue[i]->nn_index); queue[i]->nn_index = NULL;
			free(queue[i]); queue[i] = NULL;
		}
	}
	free(queue); queue = NULL;
}


void thread_queue_insert(ThreadQueue *queue, int heap_idx, double distance, int nn_idx)
{
	if (queue->cnt >= queue->size)
	{
		queue->size *= 2;
		queue->dist = realloc(queue->dist, queue->size * sizeof(double));
		queue->heap_index = realloc(queue->heap_index, queue->size * sizeof(int));
		queue->nn_index = realloc(queue->nn_index, queue->size * sizeof(int));
		if (queue->dist == NULL || queue->heap_index == NULL || queue->nn_index == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	}

	queue->dist[queue->cnt] = distance;
	queue->heap_index[queue->cnt] = heap_idx;
	queue->nn_index[queue->cnt] = nn_idx;
	++(queue->cnt);
}


void thread_queue_insert_fixed_sized(ThreadQueue *queue, int heap_idx, double distance, int nn_idx)
{
	queue->dist[queue->cnt] = distance;
	queue->heap_index[queue->cnt] = heap_idx;
	queue->nn_index[queue->cnt] = nn_idx;
	++(queue->cnt);
}
