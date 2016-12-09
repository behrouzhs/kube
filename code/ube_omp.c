#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "ube.h"
#include "algebra_tools.h"
#include "file_operations.h"


void ube_optimize_omp(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, int kernel_poly_degree, int is_newton, int no_thread, double *Y)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_ink, n_tnp, n_tnd;

	double len, d_temp, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)MAX_ITER);

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
#pragma omp master
		no_thread = omp_get_num_threads();
	}
	double *similarity = (double*)malloc(sizeof(double) * no_point * no_thread);
	double *att_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *gradient = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *newton = (double*)malloc(sizeof(double) * no_dim * no_thread);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL || newton == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random_omp(no_point, no_dim, no_thread, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
#pragma omp parallel default(none) private(i, j, k, d, n_ind, n_jnd, n_ink, n_tnd, n_tnp, len, n_temp, d_temp) shared(Y, nn_idx, affinity, no_point, no_dim, knn, alpha, similarity, att_f, rep_f, gradient, newton, is_newton, kernel_poly_degree)
	{
		int tid = omp_get_thread_num();
		n_tnd = tid * no_dim;
		n_tnp = tid * no_point;

		#pragma omp for
		for (i = 0; i < no_point; ++i)
		{
			n_ind = i * no_dim;
			n_ink = i * knn;
			// compute attractive force
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_tnd + d;
				att_f[n_temp] = 0;
				for (k = 0; k < knn; ++k)
					att_f[n_temp] += affinity[n_ink + k] * Y[(nn_idx[n_ink + k] * no_dim) + d];
				len += att_f[n_temp] * att_f[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				att_f[n_tnd + d] /= len;

			// compute repulsive force
			for (j = 0; j < no_point; ++j)
			{
				n_temp = n_tnp + j;
				n_jnd = j * no_dim;
				similarity[n_temp] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[n_temp] += Y[n_jnd + d] * Y[n_ind + d];
				similarity[n_temp] = (similarity[n_temp] + 1.0) / 2.0;

				// poynomial kernel
				d_temp = similarity[n_temp];
				for (d = 0; d < (kernel_poly_degree - 1); d++)
					similarity[n_temp] *= d_temp;
			}
			similarity[n_tnp + i] = 0;

			for (k = 0; k < knn; ++k)
			{
				if (affinity[n_ink + k] > 0)
				{
					/*similarity[n_tnp + nn_idx[n_ink + k]] -= affinity[n_ink + k];
					if (similarity[n_tnp + nn_idx[n_ink + k]] < 0)
						similarity[n_tnp + nn_idx[n_ink + k]] = 0;*/
					similarity[n_tnp + nn_idx[n_ink + k]] = 0;
				}
				else
					break;
			}
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_tnd + d;
				n_jnd = d;
				rep_f[n_temp] = 0;
				for (j = 0; j < no_point; ++j, n_jnd += no_dim)
					rep_f[n_temp] += similarity[n_tnp + j] * Y[n_jnd];
				len += rep_f[n_temp] * rep_f[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				rep_f[n_tnd + d] /= len;

			// compute final gradient
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				gradient[n_tnd + d] = att_f[n_tnd + d] - rep_f[n_tnd + d];
				len += gradient[n_tnd + d] * gradient[n_tnd + d];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				gradient[n_tnd + d] /= len;

			if (is_newton)
				inverse_mult(Y, no_point, no_dim, &gradient[n_tnd], &newton[n_tnd]);

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				if (is_newton)
					Y[n_temp] = Y[n_temp] + (alpha * newton[n_tnd + d]);
				else
					Y[n_temp] = Y[n_temp] + (alpha * gradient[n_tnd + d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			for (d = 0; d < no_dim; ++d)
				Y[n_ind + d] /= len;
		}
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
	free(newton); newton = NULL;
}


void initialize_random_omp(int no_point, int no_dim, int no_thread, double *out_Y)
{
	int i, d, n_temp, n_ind;
	double len;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, d, n_ind, n_temp, len) shared(no_point, no_dim, out_Y)
	{
		#pragma omp for
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
}


void dist_to_affinity_elbow_omp(double *D_W, int no_point, int knn, int min_knn, int no_thread)
{
	int i, k, n_temp, n_ink, elbow_cut = 0;
	double sigma2, d_temp, min_d, max_d, elbow_mxd = 0.0;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, k, n_temp, n_ink, sigma2, d_temp, min_d, max_d, elbow_cut, elbow_mxd) shared(no_point, knn, min_knn, D_W)
	{
		#pragma omp for
		for (i = 0; i < no_point; ++i)
		{
			n_ink = i * knn;
			min_d = D_W[n_ink];
			max_d = D_W[n_ink + knn - 1];

			elbow(&D_W[n_ink], knn, &elbow_cut, &elbow_mxd);
			if (elbow_mxd >(double)(ELBOW_THRESH * (double)knn))
				//sigma2 = D_W[n_ink + elbow_cut - 1] / (double)SIGMA_COVERAGE;
				sigma2 = (D_W[n_ink + elbow_cut - 1] - min_d) / (double)SIGMA_COVERAGE;
			else
			{
				//sigma2 = max_d / (double)SIGMA_COVERAGE;
				sigma2 = (max_d - min_d) / (double)SIGMA_COVERAGE;
				sigma2 *= (1 - ((max_d - min_d) / max_d));
			}
			//sigma2 = fmax(sigma2, D_W[n_ink + min_knn - 1] / (double)SIGMA_COVERAGE);
			sigma2 = fmax(sigma2, (D_W[n_ink + min_knn - 1] - min_d) / (double)SIGMA_COVERAGE);

			sigma2 *= sigma2;
			if (sigma2 == 0)
				sigma2 = 1;

			for (k = 0; k < knn; ++k)
			{
				n_temp = n_ink + k;
				//d_temp = D_W[n_temp];
				d_temp = D_W[n_temp] - min_d;
				D_W[n_temp] = exp(-0.5 * d_temp * d_temp / sigma2);
				if (D_W[n_temp] < 0.0110)
					D_W[n_temp] = 0;
			}
		}
	}
}


void sparse_build_omp(double *data, int n_row, int no_thread, int *out_row, int *out_col, double *out_val, int *no_point, int **start_indices)
{
	int i;
	(*no_point) = -1;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i) shared(data, n_row, no_point, out_row, out_col, out_val)
	{
		#pragma omp for
		for (i = 0; i < n_row; ++i)
		{
			out_row[i] = (int)data[3 * i];
			out_col[i] = (int)data[(3 * i) + 1];
			out_val[i] = data[(3 * i) + 2];

			#pragma omp critical
			{
				if (out_row[i] > (*no_point))
					(*no_point) = out_row[i];
			}
		}
	}
	++(*no_point);

	int *counts_nz = (int*)malloc(sizeof(int) * (*no_point));
	if (counts_nz == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

#pragma omp parallel default(none) private(i) shared(n_row, no_point, out_row, counts_nz)
	{
		#pragma omp for
		for (i = 0; i < (*no_point); ++i)
			counts_nz[i] = 0;

		#pragma omp for
		for (i = 0; i < n_row; ++i)
		{
			#pragma omp atomic
			++counts_nz[out_row[i]];
		}
	}

	(*start_indices) = (int*)malloc(sizeof(int) * ((*no_point) + 1));
	if ((*start_indices) == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	(*start_indices)[0] = 0;
	for (i = 1; i <= (*no_point); ++i)
		(*start_indices)[i] = (*start_indices)[i - 1] + counts_nz[i - 1];

	free(counts_nz); counts_nz = NULL;
}


void sparse_normalize_vectors_omp(double *val, int no_point, int *start_indices, int no_thread)
{
	int i, j;
	double len;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) private(i, j, len) shared(val, no_point, start_indices)
	{
		#pragma omp for
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
}
