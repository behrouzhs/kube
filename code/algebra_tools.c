#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "algebra_tools.h"


/*
Recursive definition of determinate using expansion by minors.
*/
double determinant(double **a, int n)
{
	int i, j, j1, j2;
	double det = 0;
	double **m = NULL;

	if (n == 1) { /* Shouldn't get used */
		det = a[0][0];
	}
	else if (n == 2) {
		det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
	}
	else {
		det = 0;
		for (j1 = 0;j1<n;j1++) {
			m = malloc((n - 1)*sizeof(double *));
			for (i = 0;i<n - 1;i++)
				m[i] = malloc((n - 1)*sizeof(double));
			for (i = 1;i<n;i++) {
				j2 = 0;
				for (j = 0;j<n;j++) {
					if (j == j1)
						continue;
					m[i - 1][j2] = a[i][j];
					j2++;
				}
			}
			det += pow(-1.0, j1 + 2.0) * a[0][j1] * determinant(m, n - 1);
			for (i = 0;i<n - 1;i++)
				free(m[i]);
			free(m);
		}
	}
	return(det);
}


/*
Find the cofactor matrix of a square matrix
*/
void cofactor(double **a, int n, double **b)
{
	int i, j, ii, jj, i1, j1;
	double det;
	double **c;

	c = (double**)malloc((n - 1)*sizeof(double *));
	for (i = 0;i<n - 1;i++)
		c[i] = (double*)malloc((n - 1)*sizeof(double));

	for (j = 0; j < n; j++) {
		for (i = 0; i < n; i++) {

			/* Form the adjoint a_ij */
			i1 = 0;
			for (ii = 0; ii < n; ii++) {
				if (ii == i)
					continue;
				j1 = 0;
				for (jj = 0; jj < n; jj++) {
					if (jj == j)
						continue;
					c[i1][j1] = a[ii][jj];
					j1++;
				}
				i1++;
			}

			/* Calculate the determinate */
			det = determinant(c, n - 1);

			/* Fill in the elements of the cofactor */
			b[i][j] = pow(-1.0, i + j + 2.0) * det;
		}
	}
	for (i = 0;i<n - 1;i++)
		free(c[i]);
	free(c);
}


/*
Transpose of a square matrix, do it in place
*/
void transpose(double **a, int n)
{
	int i, j;
	double tmp;

	for (i = 1; i < n; i++) {
		for (j = 0; j < i; j++) {
			tmp = a[i][j];
			a[i][j] = a[j][i];
			a[j][i] = tmp;
		}
	}
}


void covariance(double *Y, int no_point, int no_dim, double **cov)
{
	int i, j, k, n_temp;
	for (i = 0; i < no_dim; ++i)
	{
		for (j = i; j < no_dim; ++j)
		{
			cov[i][j] = 0.0;
			for (k = 0; k < no_point; ++k)
			{
				n_temp = k * no_dim;
				cov[i][j] += Y[n_temp + i] * Y[n_temp + j];
			}
			cov[j][i] = cov[i][j];
		}
	}
}


void inverse_mult(double *Y, int no_point, int no_dim, double *grad, double *newton)
{
	int i, j;
	double len = 0;
	double **cov, **inv_cov;

	cov = (double**)malloc(no_dim * sizeof(double *));
	inv_cov = (double**)malloc(no_dim * sizeof(double *));
	if (cov == NULL || inv_cov == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < no_dim; ++i)
	{
		cov[i] = (double*)malloc(no_dim * sizeof(double));
		inv_cov[i] = (double*)malloc(no_dim * sizeof(double));
		if (cov[i] == NULL || inv_cov[i] == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	}

	covariance(Y, no_point, no_dim, cov);
	cofactor(cov, no_dim, inv_cov);
	for (i = 0; i < no_dim; ++i)
	{
		newton[i] = 0.0;
		for (j = 0; j < no_dim; ++j)
			newton[i] += inv_cov[i][j] * grad[j];
		len += newton[i] * newton[i];
	}
	len = sqrt(len);
	for (i = 0; i < no_dim; ++i)
		newton[i] /= len;

	for (i = 0; i < no_dim; ++i)
	{
		free(cov[i]); cov[i] = NULL;
		free(inv_cov[i]); inv_cov[i] = NULL;
	}
	free(cov); cov = NULL;
	free(inv_cov); inv_cov = NULL;
}