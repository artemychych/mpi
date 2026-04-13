#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>

void init_matrix(double *mat, int n, int value) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * n + j] = value;
        }
    }
}

double check_result(double *C, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += C[i * n + j];
        }
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    double *A = NULL, *B = NULL, *C = NULL;
    double *A_local = NULL, *C_local = NULL;
    int local_rows;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    N = atoi(argv[1]);
    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: matrix size must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    local_rows = N / size;

    B = (double*)malloc(N * N * sizeof(double));
    A_local = (double*)malloc(local_rows * N * sizeof(double));
    C_local = (double*)malloc(local_rows * N * sizeof(double));
    if (B == NULL || A_local == NULL || C_local == NULL) {
        fprintf(stderr, "Process %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        if (A == NULL || C == NULL) {
            fprintf(stderr, "Process 0: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        init_matrix(A, N, 1);
        init_matrix(B, N, 1);

    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, local_rows * N, MPI_DOUBLE, A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    #pragma omp parallel for
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = sum;
        }
    }

    MPI_Gather(C_local, local_rows * N, MPI_DOUBLE, C, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) end_time = MPI_Wtime();

    if (rank == 0) {
        int num_threads = omp_get_max_threads();
        double elapsed = end_time - start_time;
        printf("p=%d, t=%d, N=%d, time=%.6f sec\n", size, num_threads, N, elapsed);

        double sum = check_result(C, N);
        printf("Sum of C elements = %.0f (expected = %d)\n", sum, N * N * N);
        free(A);
        free(C);
    }

    free(B);
    free(A_local);
    free(C_local);

    MPI_Finalize();
    return 0;
}