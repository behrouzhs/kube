#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "tools_portable.h"
#include "knn_search.h"
#include "ube.h"
#include "file_operations.h"


void ube(double *X, int *no_point, int no_dim_in, int no_dim, int knn, int min_knn, int is_euclidean, int no_thread, double *Y)
{
	double start_time;

	double *affinity = (double*)malloc(sizeof(double) * (*no_point) * knn);
	int *nn_idx = (int*)malloc(sizeof(int) * (*no_point) * knn);
	if (affinity == NULL || nn_idx == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if (no_dim_in == 3)
	{
		int *row = (int*)malloc(sizeof(int) * (*no_point));
		int *col = (int*)malloc(sizeof(int) * (*no_point));
		double *val = (double*)malloc(sizeof(double) * (*no_point));
		if (row == NULL || col == NULL || val == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

		int *start_indices;
		int n_row = (*no_point);

		start_time = time_gettime();
		if (no_thread > 1)
		{
			sparse_build_omp(X, n_row, no_thread, row, col, val, no_point, &start_indices);
			sparse_normalize_vectors_omp(val, (*no_point), start_indices, no_thread);
			if (is_euclidean > 0)
				knn_search_sparse_euclidean_omp(col, val, (*no_point), start_indices, knn, no_thread, nn_idx, affinity);
			else
				knn_search_sparse_cosine_omp(col, val, (*no_point), start_indices, knn, no_thread, nn_idx, affinity);
		}
		else
		{
			sparse_build(X, n_row, row, col, val, no_point, &start_indices);
			sparse_normalize_vectors(val, (*no_point), start_indices);
			if (is_euclidean > 0)
				knn_search_sparse_euclidean(col, val, (*no_point), start_indices, knn, nn_idx, affinity);
			else
				knn_search_sparse_cosine(col, val, (*no_point), start_indices, knn, nn_idx, affinity);
		}
		printf("knn search time: %lf seconds\r\n", time_duration(start_time));

		free(row); row = NULL;
		free(col); col = NULL;
		free(val); val = NULL;
		free(start_indices); start_indices = NULL;
	}
	else
	{
		start_time = time_gettime();
		if (no_thread > 1)
			knn_search_omp(X, (*no_point), no_dim_in, knn, no_thread, nn_idx, affinity);
		else
			knn_search(X, (*no_point), no_dim_in, knn, nn_idx, affinity);
		printf("knn search time: %lf seconds\r\n", time_duration(start_time));
	}

	start_time = time_gettime();
	if (no_thread > 1)
	{
		if (is_euclidean > 0)
			dist_to_affinity_elbow_underdevel_omp(affinity, (*no_point), knn, min_knn, no_thread);
		else
			dist_to_affinity_elbow_cosine_omp(affinity, (*no_point), knn, min_knn, no_thread);
		ube_optimize_kernel_omp(nn_idx, affinity, (*no_point), no_dim, knn, no_thread, Y);
	}
	else
	{
		if (is_euclidean > 0)
			dist_to_affinity_elbow_underdevel(affinity, (*no_point), knn, min_knn);
		else
			dist_to_affinity_elbow_cosine(affinity, (*no_point), knn, min_knn);
		ube_optimize_underdevel(nn_idx, affinity, (*no_point), no_dim, knn, Y);
	}
	printf("optimization time: %lf seconds\r\n", time_duration(start_time));

	free(affinity); affinity = NULL;
	free(nn_idx); nn_idx = NULL;
}


void ube_optimize_icmla2016(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, double *Y)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_ink;

	double len, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)MAX_ITER);
	double *similarity = (double*)malloc(sizeof(double) * no_point);
	double *att_f = (double*)malloc(sizeof(double) * no_dim);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim);
	double *gradient = (double*)malloc(sizeof(double) * no_dim);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random(no_point, no_dim, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			n_ink = i * knn;
			// compute attractive force
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				att_f[d] = 0;
				for (k = 0; k < knn; ++k)
					att_f[d] += affinity[n_ink + k] * Y[(nn_idx[n_ink + k] * no_dim) + d];
				len += att_f[d] * att_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				att_f[d] /= len;

			// compute repulsive force
			for (j = 0; j < no_point; ++j)
			{
				n_jnd = j * no_dim;
				similarity[j] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[j] += Y[n_jnd + d] * Y[n_ind + d];
			}
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_jnd = d;
				rep_f[d] = 0;
				for (j = 0; j < no_point; ++j, n_jnd += no_dim)
					rep_f[d] += similarity[j] * Y[n_jnd];
				len += rep_f[d] * rep_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				rep_f[d] /= len;

			// compute final gradient
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				gradient[d] = att_f[d] - rep_f[d];
				len += gradient[d] * gradient[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				gradient[d] /= len;

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				Y[n_temp] = Y[n_temp] + (alpha * gradient[d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				Y[n_ind + d] /= len;
		}

		// update learning rate
		alpha_base -= alpha_step;
		alpha = (alpha_base * 13.0 / ALPHA_INIT) - 6.0;
		alpha = 1.0 / (1.0 + exp(-alpha));
		alpha = (alpha * 0.9 * ALPHA_INIT) + (0.1 * ALPHA_INIT);
	}

	free(similarity); similarity = NULL;
	free(att_f); att_f = NULL;
	free(rep_f); rep_f = NULL;
	free(gradient); gradient = NULL;
}


void ube_optimize_underdevel(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, double *Y)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_ink;

	double len, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)MAX_ITER);
	double *similarity = (double*)malloc(sizeof(double) * no_point);
	double *att_f = (double*)malloc(sizeof(double) * no_dim);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim);
	double *gradient = (double*)malloc(sizeof(double) * no_dim);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random(no_point, no_dim, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			n_ink = i * knn;
			// compute attractive force
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				att_f[d] = 0;
				for (k = 0; k < knn; ++k)
					att_f[d] += affinity[n_ink + k] * Y[(nn_idx[n_ink + k] * no_dim) + d];
				len += att_f[d] * att_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				att_f[d] /= len;

			// compute repulsive force
			for (j = 0; j < no_point; ++j)
			{
				n_jnd = j * no_dim;
				similarity[j] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[j] += Y[n_jnd + d] * Y[n_ind + d];
				similarity[j] = (similarity[j] + 1.0) / 2.0; // negf 01
			}
			similarity[i] = 0;
			for (k = 0; k < knn; ++k)
			{
				if (affinity[n_ink + k] > 0)
					//similarity[nn_idx[n_ink + k]] -= affinity[n_ink + k]; // negf subaff
					similarity[nn_idx[n_ink + k]] = 0; // negf onc
				else
					break;
			}
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_jnd = d;
				rep_f[d] = 0;
				for (j = 0; j < no_point; ++j, n_jnd += no_dim)
					rep_f[d] += similarity[j] * Y[n_jnd];
				len += rep_f[d] * rep_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				rep_f[d] /= len;

			// compute final gradient
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				gradient[d] = att_f[d] - rep_f[d];
				len += gradient[d] * gradient[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				gradient[d] /= len;

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				Y[n_temp] = Y[n_temp] + (alpha * gradient[d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				Y[n_ind + d] /= len;
		}

		// update learning rate
		alpha_base -= alpha_step;
		alpha = (alpha_base * 13.0 / ALPHA_INIT) - 6.0;
		alpha = 1.0 / (1.0 + exp(-alpha));
		alpha = (alpha * 0.9 * ALPHA_INIT) + (0.1 * ALPHA_INIT);
	}

	free(similarity); similarity = NULL;
	free(att_f); att_f = NULL;
	free(rep_f); rep_f = NULL;
	free(gradient); gradient = NULL;
}


void ube_optimize_hybrid_underdevel(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, double *Y)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_ink;

	double len, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)MAX_ITER);
	double *similarity = (double*)malloc(sizeof(double) * no_point);
	double *att_f = (double*)malloc(sizeof(double) * no_dim);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim);
	double *gradient = (double*)malloc(sizeof(double) * no_dim);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random(no_point, no_dim, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			n_ink = i * knn;
			// compute attractive force
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				att_f[d] = 0;
				for (k = 0; k < knn; ++k)
					att_f[d] += affinity[n_ink + k] * Y[(nn_idx[n_ink + k] * no_dim) + d];
				len += att_f[d] * att_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				att_f[d] /= len;

			// compute repulsive force
			for (j = 0; j < no_point; ++j)
			{
				n_jnd = j * no_dim;
				similarity[j] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[j] += Y[n_jnd + d] * Y[n_ind + d];
				similarity[j] = (similarity[j] + 1.0) / 2.0; // negf 01
			}
			similarity[i] = 0;
			for (k = 0; k < knn; ++k)
			{
				if (affinity[n_ink + k] > 0)
					similarity[nn_idx[n_ink + k]] -= affinity[n_ink + k]; // negf subaff
																		  //similarity[nn_idx[n_ink + k]] = 0; // negf onc
				else
					break;
			}
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_jnd = d;
				rep_f[d] = 0;
				for (j = 0; j < no_point; ++j, n_jnd += no_dim)
					rep_f[d] += similarity[j] * Y[n_jnd];
				len += rep_f[d] * rep_f[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				rep_f[d] /= len;

			// compute final gradient
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				gradient[d] = att_f[d] - rep_f[d];
				len += gradient[d] * gradient[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				gradient[d] /= len;

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				Y[n_temp] = Y[n_temp] + (alpha * gradient[d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				Y[n_ind + d] /= len;
		}

		// update learning rate
		alpha_base -= alpha_step;
		alpha = (alpha_base * 13.0 / ALPHA_INIT) - 6.0;
		alpha = 1.0 / (1.0 + exp(-alpha));
		alpha = (alpha * 0.9 * ALPHA_INIT) + (0.1 * ALPHA_INIT);
	}

	free(similarity); similarity = NULL;
	free(att_f); att_f = NULL;
	free(rep_f); rep_f = NULL;
	free(gradient); gradient = NULL;
}


void ube_optimize_norm_underdevel(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, double *Y)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_ink;

	double len, sum_sim, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)MAX_ITER);
	double *similarity = (double*)malloc(sizeof(double) * no_point);
	double *gradient = (double*)malloc(sizeof(double) * no_dim);
	if (similarity == NULL || gradient == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random(no_point, no_dim, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			n_ink = i * knn;
			// compute gradient
			sum_sim = 0;
			for (j = 0; j < no_point; ++j)
			{
				n_jnd = j * no_dim;
				similarity[j] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[j] += Y[n_jnd + d] * Y[n_ind + d];
				similarity[j] = (similarity[j] + 1.0) / 2.0;
				sum_sim += similarity[j];
			}
			sum_sim -= similarity[i];
			similarity[i] = 0;
			for (j = 0; j < no_point; ++j)
				similarity[j] /= sum_sim;
			for (k = 0; k < knn; ++k)
				similarity[nn_idx[n_ink + k]] -= affinity[n_ink + k];
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_jnd = d;
				gradient[d] = 0;
				for (j = 0; j < no_point; ++j, n_jnd += no_dim)
					gradient[d] += similarity[j] * Y[n_jnd];
				len += gradient[d] * gradient[d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				gradient[d] /= len;

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				Y[n_temp] = Y[n_temp] + (alpha * gradient[d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				Y[n_ind + d] /= len;
		}

		// update learning rate
		alpha_base -= alpha_step;
		alpha = (alpha_base * 13.0 / ALPHA_INIT) - 6.0;
		alpha = 1.0 / (1.0 + exp(-alpha));
		alpha = (alpha * 0.9 * ALPHA_INIT) + (0.1 * ALPHA_INIT);
	}

	free(similarity); similarity = NULL;
	free(gradient); gradient = NULL;
}


void initialize_random(int no_point, int no_dim, double *out_Y)
{
	int i, d, n_temp, n_ind;
	double len;
	for (i = 0; i < no_point; ++i)
	{
		n_ind = i * no_dim;
		len = 0;
		for (d = 0; d < no_dim; ++d)
		{
			n_temp = n_ind + d;
			out_Y[n_temp] = ((double)rand() / (double)RAND_MAX) - 0.5;
			len += out_Y[n_temp] * out_Y[n_temp];
		}
		len = sqrt(len);
		for (d = 0; d < no_dim; ++d)
			out_Y[n_ind + d] /= len;
	}
}


void dist_to_affinity_icmla2016(double *D_W, int no_point, int knn)
{
	int i, k, n_temp;
	double sigma2;

	for (i = 0; i < no_point; ++i)
	{
		sigma2 = D_W[(i * knn) + knn - 1] / (double)SIGMA_COVERAGE;
		sigma2 *= sigma2;
		if (sigma2 == 0)
			sigma2 = 1;

		for (k = 0; k < knn; ++k)
		{
			n_temp = (i * knn) + k;
			D_W[n_temp] = exp(-0.5 * D_W[n_temp] * D_W[n_temp] / sigma2);
		}
	}
}


void dist_to_affinity_underdevel(double *D_W, int no_point, int knn, int min_knn)
{
	int i, k, n_temp, n_ink;
	double sigma2, d_temp, min_d, max_d;

	for (i = 0; i < no_point; ++i)
	{
		n_ink = i * knn;
		min_d = D_W[n_ink];
		max_d = D_W[n_ink + knn - 1];
		sigma2 = max_d / (double)SIGMA_COVERAGE;
		//sigma2 = (max_d - min_d) / (double)SIGMA_COVERAGE;
		sigma2 *= (1 - ((max_d - min_d) / max_d));
		//sigma2 *= (1 - ((max_d - ((D_W[n_ink] + D_W[n_ink + 1]) / 2.0)) / max_d));
		sigma2 = fmax(sigma2, D_W[n_ink + min_knn - 1] / (double)SIGMA_COVERAGE);
		//sigma2 = fmax(sigma2, (D_W[n_ink + min_knn - 1] - min_d) / (double)SIGMA_COVERAGE);
		//sigma2 = (D_W[n_ink] + D_W[n_ink + 1] + D_W[n_ink + 2]) / 3.0;
		//sigma2 = min(sigma2, (D_W[n_ink + knn - 1] - min_d) / (double)SIGMA_COVERAGE);
		sigma2 *= sigma2;
		if (sigma2 == 0)
			sigma2 = 1;

		for (k = 0; k < knn; ++k)
		{
			n_temp = n_ink + k;
			d_temp = D_W[n_temp];
			//d_temp = D_W[n_temp] - min_d;
			D_W[n_temp] = exp(-0.5 * d_temp * d_temp / sigma2);
			if (D_W[n_temp] < 0.0111)
				D_W[n_temp] = 0;
		}
	}
}


void dist_to_affinity_elbow_underdevel(double *D_W, int no_point, int knn, int min_knn)
{
	int i, k, n_temp, n_ink, elbow_cut = 0;
	double sigma2, d_temp, min_d, max_d, elbow_mxd = 0.0;

	for (i = 0; i < no_point; ++i)
	{
		n_ink = i * knn;
		min_d = D_W[n_ink];
		max_d = D_W[n_ink + knn - 1];

		elbow(&D_W[n_ink], knn, &elbow_cut, &elbow_mxd);
		if (elbow_mxd > (double)(ELBOW_THRESH * (double)knn))
			sigma2 = D_W[n_ink + elbow_cut - 1] / (double)SIGMA_COVERAGE;
			//sigma2 = (D_W[n_ink + elbow_cut - 1] - min_d) / (double)SIGMA_COVERAGE;
		else
		{
			sigma2 = max_d / (double)SIGMA_COVERAGE;
			//sigma2 = (max_d - min_d) / (double)SIGMA_COVERAGE;
			sigma2 *= (1 - ((max_d - min_d) / max_d));
		}
		sigma2 = fmax(sigma2, D_W[n_ink + min_knn - 1] / (double)SIGMA_COVERAGE);
		//sigma2 = fmax(sigma2, (D_W[n_ink + min_knn - 1] - min_d) / (double)SIGMA_COVERAGE);
		
		sigma2 *= sigma2;
		if (sigma2 == 0)
			sigma2 = 1;

		for (k = 0; k < knn; ++k)
		{
			n_temp = n_ink + k;
			d_temp = D_W[n_temp];
			//d_temp = D_W[n_temp] - min_d;
			D_W[n_temp] = exp(-0.5 * d_temp * d_temp / sigma2);
			if (D_W[n_temp] < 0.0110)
				D_W[n_temp] = 0;
		}
	}
}


void dist_to_affinity_elbow_cosine(double *D_W, int no_point, int knn, int min_knn)
{
	int i, k, n_temp, n_ink, elbow_cut = 0;
	double elbow_mxd = 0.0;

	for (i = 0; i < no_point; ++i)
	{
		n_ink = i * knn;
		for (k = 0; k < knn; ++k)
		{
			n_temp = n_ink + k;
			D_W[n_temp] = 1 - D_W[n_temp];
		}

		elbow(&D_W[n_ink], knn, &elbow_cut, &elbow_mxd);
		if (elbow_mxd > (double)(ELBOW_THRESH * (double)knn))
			for (k = ((int)fmax(elbow_cut, min_knn) + 1); k < knn; ++k)
				D_W[n_ink + k] = 0;
		else
			for (k = ((int)fmax(min_knn, (double)knn * (1.0 - ((D_W[n_ink] - D_W[n_ink + knn - 1]) / D_W[n_ink]))) + 1); k < knn; ++k)
				D_W[n_ink + k] = 0;
	}
}


void dist_to_affinity_norm_underdevel(double *D_W, int no_point, int knn)
{
	int i, k, n_temp, n_ink;
	double sigma2, d_temp, sum_w, min_d;

	for (i = 0; i < no_point; ++i)
	{
		n_ink = i * knn;
		min_d = D_W[n_ink];
		sigma2 = (D_W[n_ink + knn - 1] - min_d) / (double)SIGMA_COVERAGE;
		//sigma2 = (D_W[n_ink] + D_W[n_ink + 1] + D_W[n_ink + 2]) / 3.0;
		sigma2 *= sigma2;
		if (sigma2 == 0)
			sigma2 = 1;

		sum_w = 0;
		for (k = 0; k < knn; ++k)
		{
			n_temp = n_ink + k;
			d_temp = D_W[n_temp] - min_d;
			D_W[n_temp] = exp(-0.5 * d_temp * d_temp / sigma2);
			sum_w += D_W[n_temp];
			/*if (D_W[n_temp] < 0.01)
				D_W[n_temp] = 0;*/
		}
		for (k = 0; k < knn; ++k)
			D_W[n_ink + k] /= sum_w;
	}
}


void sparse_build(double *data, int n_row, int *out_row, int *out_col, double *out_val, int *no_point, int **start_indices)
{
	int i;

	(*no_point) = -1;
	for (i = 0; i < n_row; ++i)
	{
		out_row[i] = (int)data[3 * i];
		out_col[i] = (int)data[(3 * i) + 1];
		out_val[i] = data[(3 * i) + 2];

		if (out_row[i] >(*no_point))
			(*no_point) = out_row[i];
	}
	++(*no_point);

	int *counts_nz = (int*)malloc(sizeof(int) * (*no_point));
	if (counts_nz == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < (*no_point); ++i)
		counts_nz[i] = 0;
	for (i = 0; i < n_row; ++i)
		++counts_nz[out_row[i]];

	(*start_indices) = (int*)malloc(sizeof(int) * ((*no_point) + 1));
	if ((*start_indices) == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	(*start_indices)[0] = 0;
	for (i = 1; i <= (*no_point); ++i)
		(*start_indices)[i] = (*start_indices)[i - 1] + counts_nz[i - 1];

	free(counts_nz); counts_nz = NULL;
}


void sparse_normalize_vectors(double *val, int no_point, int *start_indices)
{
	int i, j;
	double len;

	for (i = 0; i < no_point; ++i)
	{
		len = 0;
		for (j = start_indices[i]; j < start_indices[i + 1]; ++j)
			len += val[j] * val[j];
		len = sqrt(len);

		for (j = start_indices[i]; j < start_indices[i + 1]; ++j)
			val[j] /= len;
	}
}


void elbow(double *curve, int n, int *cut, double *max_dist)
{
	int i;
	double xlvn, ylvn, len, xp, yp, scalar_prod;
	double *x = (double*)malloc(sizeof(double) * n);
	double *y = (double*)malloc(sizeof(double) * n);
	if (x == NULL || y == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	double mx = curve[0];
	for (i = 1; i < n; ++i)
		if (curve[i] > mx)
			mx = curve[i];
	for (i = 0; i < n; i++)
	{
		y[i] = (curve[i] - curve[0]) * n / mx;
		x[i] = i;
	}
	len = sqrt((x[n - 1] * x[n - 1]) + (y[n - 1] * y[n - 1]));
	xlvn = x[n - 1] / len;
	ylvn = y[n - 1] / len;

	*max_dist = -1.0;
	for (i = 0; i < n; i++)
	{
		scalar_prod = (x[i] * xlvn) + (y[i] * ylvn);
		xp = (scalar_prod * xlvn) - x[i];
		yp = (scalar_prod * ylvn) - y[i];
		len = sqrt((xp * xp) + (yp * yp));

		if (len > (*max_dist))
		{
			*max_dist = len;
			*cut = i;
		}
	}

	free(x); x = NULL;
	free(y); y = NULL;
}
