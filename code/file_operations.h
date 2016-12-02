#pragma once

#ifndef FILE_OPERATIONS
#define FILE_OPERATIONS

int load_matrix_double(char *path, double **data, int *n_row, int *n_col);
void save_matrix_double(char *path, double *data, int n_row, int n_col);
void save_matrix_int(char *path, int *data, int n_row, int n_col);

#endif