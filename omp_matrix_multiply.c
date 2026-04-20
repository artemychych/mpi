#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double** allocate_matrix(int N) {
    double** mat = (double**)malloc(N * sizeof(double*));
    double* data = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        mat[i] = data + i * N;
    }
    return mat;
}

void free_matrix(double** mat) {
    free(mat[0]);
    free(mat);
}

void init_matrix_vector(double** A, double* B, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
        B[i] = (double)rand() / RAND_MAX;
    }
}

void matvec_serial(double** A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * B[j];
        }
        C[i] = sum;
    }
}

void matvec_parallel(double** A, double* B, double* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * B[j];
        }
        C[i] = sum;
    }
}

int compare_vectors(double* C1, double* C2, int N, double eps) {
    for (int i = 0; i < N; i++) {
        if (fabs(C1[i] - C2[i]) > eps) {
            return 0;
        }
    }
    return 1;
}

int main() {
    const int N = 10000;

    double** A = allocate_matrix(N);
    double* B = (double*)malloc(N * sizeof(double));
    double* C_serial = (double*)malloc(N * sizeof(double));
    double* C_parallel = (double*)malloc(N * sizeof(double));

    init_matrix_vector(A, B, N);

    double start_serial = omp_get_wtime();
    matvec_serial(A, B, C_serial, N);
    double time_serial = omp_get_wtime() - start_serial;

    double start_parallel = omp_get_wtime();
    matvec_parallel(A, B, C_parallel, N);
    double time_parallel = omp_get_wtime() - start_parallel;

    int correct = compare_vectors(C_serial, C_parallel, N, 1e-10);
    printf("Result %s\n", correct ? "correct" : "not correct");
    printf("Threads:      %d\n", omp_get_max_threads());
    printf("Time serial: %.4f sec.\n", time_serial);
    printf("Time parallel:     %.4f sec.\n", time_parallel);
    printf("Boost:             %.2fx\n", time_serial / time_parallel);

    free_matrix(A);
    free(B);
    free(C_serial);
    free(C_parallel);

    return 0;
}