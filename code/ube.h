#pragma once

#ifndef UBE_H
#define UBE_H

#define ALPHA_INIT 0.1
#define MAX_ITER 500
#define SIGMA_COVERAGE 3.0
#define ELBOW_THRESH 0.1

void ube(double *X, int *no_point, int no_dim_in, int no_dim, int knn, int min_knn, int kernel_poly_degree, int is_newton, int is_euclidean, int no_thread, double *Y);
void elbow(double *curve, int n, int *cut, double *max_dist);

// sequential functions
void sparse_build(double *data, int n_row, int *out_row, int *out_col, double *out_val, int *no_point, int **start_indices);
void sparse_normalize_vectors(double *val, int no_point, int *start_indices);
void initialize_random(int no_point, int no_dim, double *out_Y);
void dist_to_affinity_elbow(double *D_W, int no_point, int knn, int min_knn);
void ube_optimize(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, double *Y);

// parallel functions
void sparse_build_omp(double *data, int n_row, int no_thread, int *out_row, int *out_col, double *out_val, int *no_point, int **start_indices);
void sparse_normalize_vectors_omp(double *val, int no_point, int *start_indices, int no_thread);
void initialize_random_omp(int no_point, int no_dim, int no_thread, double *out_Y);
void dist_to_affinity_elbow_omp(double *D_W, int no_point, int knn, int min_knn, int no_thread);
void ube_optimize_omp(int *nn_idx, double *affinity, int no_point, int no_dim, int knn, int kernel_poly_degree, int is_newton, int no_thread, double *Y);

#endif
