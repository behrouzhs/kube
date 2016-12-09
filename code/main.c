#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "file_operations.h"
#include "tools_portable.h"
#include "ube.h"
#ifdef _WIN32
	#include "getopt.h"
	#define ARG_PARSER_ARGUMENT optarg_a
	#define ARG_PARSER_FUNC getopt_a
#else
	#include <getopt.h>
	#define ARG_PARSER_ARGUMENT optarg
	#define ARG_PARSER_FUNC getopt
#endif


#define DEFAULT_NO_THREAD 1
#define DEFAULT_DIMENSIONALITY 3
#define DEFAULT_KNN 90
#define DEFAULT_MIN_KNN 15
#define DEFAULT_KERNEL_POLY_DEGREE 1
#define DEFAULT_IS_EUCLIDEAN 1
#define DEFAULT_IS_NEWTON 0


void create_filename(char *input_path, char **output_path);


int main(int argc, char **argv)
{
	int knn = DEFAULT_KNN, min_knn = DEFAULT_MIN_KNN, no_dim_target = DEFAULT_DIMENSIONALITY, kernel_poly_degree = DEFAULT_KERNEL_POLY_DEGREE, is_newton = DEFAULT_IS_NEWTON, is_euclidean = DEFAULT_IS_EUCLIDEAN, no_thread = DEFAULT_NO_THREAD;
	char *input_path = (char*)"data.dat";
	char *output_path = NULL;
	int arg_option;

	while ((arg_option = ARG_PARSER_FUNC(argc, argv, "cCnNd:D:k:K:m:M:p:P:i:I:o:O:t::T::")) != EOF)
	{
		switch (arg_option)
		{
		case 'c':
		case 'C':
			is_euclidean = 0;
			break;
		case 'n':
		case 'N':
			is_newton = 1;
			break;
		case 'd':
		case 'D':
			no_dim_target = atoi(ARG_PARSER_ARGUMENT);
			break;
		case 'k':
		case 'K':
			knn = atoi(ARG_PARSER_ARGUMENT);
			break;
		case 'm':
		case 'M':
			min_knn = atoi(ARG_PARSER_ARGUMENT);
			break;
		case 'p':
		case 'P':
			kernel_poly_degree = atoi(ARG_PARSER_ARGUMENT);
			break;
		case 'i':
		case 'I':
			input_path = ARG_PARSER_ARGUMENT;
			break;
		case 'o':
		case 'O':
			output_path = ARG_PARSER_ARGUMENT;
			break;
		case 't':
		case 'T':
			if (ARG_PARSER_ARGUMENT == NULL || ARG_PARSER_ARGUMENT[0] == '\0')
				no_thread = omp_get_num_procs() / 2;
			else
				no_thread = atoi(ARG_PARSER_ARGUMENT);
			break;
		default:
			printf("Invalid usage - trying to run with default arguments.\r\n");
			break;
		}
	}

	if (no_dim_target > 5)
		is_newton = 0;

	if (output_path == NULL)
		create_filename(input_path, &output_path);

	double* data;
	int no_point, no_dim;
	if (load_matrix_double(input_path, &data, &no_point, &no_dim) == 0)
		return EXIT_FAILURE;

	double* Y = (double*)malloc(sizeof(double) * no_point * no_dim_target);
	if (Y == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	double start_time = time_gettime();
	ube(data, &no_point, no_dim, no_dim_target, knn, min_knn, kernel_poly_degree, is_newton, is_euclidean, no_thread, Y);
	printf("total time: %lf seconds\r\n", time_duration(start_time));

	save_matrix_double(output_path, Y, no_point, no_dim_target);

	free(data); data = NULL;
	free(Y); Y = NULL;
	return EXIT_SUCCESS;
}


void create_filename(char *input_path, char **output_path)
{
	int len = (int)strlen(input_path);
	int idx_dot, i;
	for (idx_dot = len - 1; idx_dot >= 0 && input_path[idx_dot] != '.'; --idx_dot);
	if (idx_dot < 0)
		idx_dot = len;
	(*output_path) = (char*)malloc(sizeof(char) * (len + 5));
	if ((*output_path) == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < idx_dot; ++i)
		(*output_path)[i] = input_path[i];
	(*output_path)[i++] = '_';
	(*output_path)[i++] = 'u';
	(*output_path)[i++] = 'b';
	(*output_path)[i++] = 'e';
	for (; idx_dot < len; ++idx_dot, ++i)
		(*output_path)[i] = input_path[idx_dot];
	(*output_path)[i] = '\0';
}
