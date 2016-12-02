#include <stdio.h>
#include <stdlib.h>


int load_matrix_double(char *path, double **data, int *n_row, int *n_col)
{
	FILE *h = fopen(path, "r+b");
	if (h == NULL) { printf("Error: could not open data file \"%s\".\r\n", path); return 0; }
	fread(n_row, sizeof(int), 1, h);
	fread(n_col, sizeof(int), 1, h);
	(*data) = (double*)malloc((*n_col) * (*n_row) * sizeof(double));
	if ((*data) == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	fread(*data, sizeof(double), (*n_row) * (*n_col), h);
	fclose(h);
	return 1;
}


void save_matrix_double(char *path, double *data, int n_row, int n_col)
{
	FILE *h = fopen(path, "w+b");
	if (h == NULL) { printf("Error: could not open data file \"%s\".\r\n", path); return; }
	fwrite(&n_row, sizeof(int), 1, h);
	fwrite(&n_col, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n_row * n_col, h);
	fclose(h);
}


void save_matrix_int(char *path, int *data, int n_row, int n_col)
{
	FILE *h = fopen(path, "w+b");
	if (h == NULL) { printf("Error: could not open data file \"%s\".\r\n", path); return; }
	fwrite(&n_row, sizeof(int), 1, h);
	fwrite(&n_col, sizeof(int), 1, h);
	fwrite(data, sizeof(int), n_row * n_col, h);
	fclose(h);
}
