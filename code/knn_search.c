#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "max_heap.h"
#include "knn_search.h"


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
// faster way of calculating euclidean distances: |x-y|^2 = x^2 + y^2 - 2xy
void knn_search(double *X, int no_point, int no_dim, int knn, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	double d_temp, dist;
	int i, j, d, n_idx1, n_idx2;

	double *len = (double*)malloc(sizeof(double) * no_point);
	if (len == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	calculate_sqlengths(X, no_point, no_dim, len);

	for (i = 0; i < no_point; ++i)
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

			if (heap[j]->no_items < knn)
				maxheap_push(heap[j], dist, i);
			else if (dist < heap[j]->elements[1]->dist)
				maxheap_pop_push(heap[j], dist, i);
		}
	}

	fill_knn_from_heap_euclidean(heap, no_point, knn, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
	free(len); len = NULL;
}


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
void knn_search_sparse_cosine(int *col, double *val, int no_point, int *start_indices, int knn, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	double dist;
	int i, j, ix1, ix2, n_max1, n_max2;

	for (i = 0; i < no_point; ++i)
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

			if (heap[j]->no_items < knn)
				maxheap_push(heap[j], dist, i);
			else if (dist < heap[j]->elements[1]->dist)
				maxheap_pop_push(heap[j], dist, i);
		}
	}

	fill_knn_from_heap_cosine(heap, no_point, knn, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
}


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
void knn_search_sparse_euclidean(int *col, double *val, int no_point, int *start_indices, int knn, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	double dist, d_temp;
	int i, j, ix1, ix2, n_max1, n_max2;

	for (i = 0; i < no_point; ++i)
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

			if (heap[j]->no_items < knn)
				maxheap_push(heap[j], dist, i);
			else if (dist < heap[j]->elements[1]->dist)
				maxheap_pop_push(heap[j], dist, i);
		}
	}

	fill_knn_from_heap_euclidean(heap, no_point, knn, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
}


void fill_knn_from_heap_euclidean(MaxHeap **heap, int no_point, int knn, int *out_index, double *out_dist)
{
	int i, j, n_temp;
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


void fill_knn_from_heap_cosine(MaxHeap **heap, int no_point, int knn, int *out_index, double *out_dist)
{
	int i, j, n_temp;
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


void calculate_sqlengths(double *X, int no_point, int no_dim, double *out_len)
{
	int i, j;
	int idx = 0;
	for (i = 0; i < no_point; ++i)
	{
		out_len[i] = 0;
		for (j = 0; j < no_dim; ++j)
		{
			out_len[i] += X[idx] * X[idx];
			++idx;
		}
	}
}


// calculating full n^2 distances and maintaining 1 heap operating row by row
/*
void knn_search_full(double *X, int no_point, int no_dim, int knn, int *out_index, double *out_dist)
{
	MaxHeap *heap = maxheap_create(knn);
	double diff, dist;
	int n_temp = 0;

	for (int i = 0; i < no_point; ++i)
	{
		maxheap_reset(heap);
		for (int j = 0; j < no_point; ++j)
		{
			dist = 0;
			for (int d = 0; d < no_dim; ++d)
			{
				diff = X[(i * no_dim) + d] - X[(j * no_dim) + d];
				dist += diff * diff;
			}

			// add or replace the new item in the heap if necessary
			if (heap->no_items < knn)
				maxheap_push(heap, dist, j);
			else if (dist < heap->elements[1]->dist)
				maxheap_pop_push(heap, dist, j);
		}

		// pop the k nearest neighbors from the heap
		for (int j = 0; j < knn; ++j)
		{
			n_temp = (i * knn) + knn - j - 1;
			maxheap_pop(heap, &out_dist[n_temp], &out_index[n_temp]);
			out_dist[n_temp] = sqrt(out_dist[n_temp]);
		}
	}
	maxheap_destroy(heap);
}
*/


// calculating only the upper triangular distances n(n-1)/2 and maintaining n heaps
/*
void knn_search_tri(double *X, int no_point, int no_dim, int knn, int *out_index, double *out_dist)
{
	MaxHeap **heap = maxheap_create_multi(no_point, knn);
	double diff, dist;
	int n_temp = 0;

	for (int i = 0; i < no_point; ++i)
	{
		if (heap[i]->no_items < knn)
			maxheap_push(heap[i], (double)0.0, i);
		else
			maxheap_pop_push(heap[i], (double)0.0, i);

		for (int j = i + 1; j < no_point; ++j)
		{
			dist = 0;
			for (int d = 0; d < no_dim; ++d)
			{
				diff = X[(i * no_dim) + d] - X[(j * no_dim) + d];
				dist += diff * diff;
			}

			// add or replace the new item in the heap if necessary
			if (heap[i]->no_items < knn)
				maxheap_push(heap[i], dist, j);
			else if (dist < heap[i]->elements[1]->dist)
				maxheap_pop_push(heap[i], dist, j);

			if (heap[j]->no_items < knn)
				maxheap_push(heap[j], dist, i);
			else if (dist < heap[j]->elements[1]->dist)
				maxheap_pop_push(heap[j], dist, i);
		}
	}

	fill_knn_from_heap_euclidean(heap, no_point, knn, out_index, out_dist);
	maxheap_destroy_multi(heap, no_point);
}
*/
