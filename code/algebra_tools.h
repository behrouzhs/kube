#pragma once

#ifndef ALGEBRA_TOOLS_H
#define ALGEBRA_TOOLS_H

double determinant(double **a, int n);
void cofactor(double **a, int n, double **b);
void transpose(double **a, int n);
void covariance(double *Y, int no_point, int no_dim, double **cov);
void inverse_mult(double *Y, int no_point, int no_dim, double *grad, double *newton);

#endif