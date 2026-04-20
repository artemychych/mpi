#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

double* allocate_matrix(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

void fill_matrix(double *mat, int rows, int cols, int start_val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = (double)(i * cols + j + start_val);
        }
    }
}

void matmul_seq(double *A, double *B, double *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 1000;  // строки A
    int K = 800;   // столбцы A / строки B
    int N = 600;   // столбцы B

    double *A = NULL;
    double *B = allocate_matrix(K, N);
    double *C_seq = NULL;

    double start_time, end_time;

    if (rank == 0) {
        A = allocate_matrix(M, K);
        fill_matrix(A, M, K, 0);
        fill_matrix(B, K, N, 0);
        printf("Matrixes: A(%d x %d), B(%d x %d)\n", M, K, K, N);
    }

    MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *sendcounts = NULL;
    int *displs = NULL;
    int my_rows;
    double *local_A = NULL;

    int base_rows = M / size;
    int remainder = M % size;

    if (remainder == 0) {
        my_rows = base_rows;
        local_A = allocate_matrix(my_rows, K);

        MPI_Scatter(A, my_rows * K, MPI_DOUBLE,
                    local_A, my_rows * K, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = rows_for_i * K;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        my_rows = base_rows + (rank < remainder ? 1 : 0);
        local_A = allocate_matrix(my_rows, K);

        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                     local_A, my_rows * K, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    double *local_C = allocate_matrix(my_rows, N);
    start_time = MPI_Wtime();

    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += local_A[i * K + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    end_time = MPI_Wtime();
    double local_comp_time = end_time - start_time;
    double max_comp_time;
    MPI_Reduce(&local_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *C_par = NULL;
    if (rank == 0) {
        C_par = allocate_matrix(M, N);
    }

    if (remainder == 0) {
        MPI_Gather(local_C, my_rows * N, MPI_DOUBLE,
                   C_par, my_rows * N, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    } else {
        int *recvcounts = (int*)malloc(size * sizeof(int));
        int *rdispls = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            recvcounts[i] = rows_for_i * N;
            rdispls[i] = offset;
            offset += recvcounts[i];
        }

        MPI_Gatherv(local_C, my_rows * N, MPI_DOUBLE,
                    C_par, recvcounts, rdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        free(recvcounts);
        free(rdispls);
    }

    if (rank == 0) {
        C_seq = allocate_matrix(M, N);
        double t_start = MPI_Wtime();
        matmul_seq(A, B, C_seq, M, K, N);
        double t_seq = MPI_Wtime() - t_start;

        int errors = 0;
        double eps = 1e-6;
        for (int i = 0; i < M * N; i++) {
            if (fabs(C_par[i] - C_seq[i]) > eps) {
                errors++;
                if (errors < 5)
                    printf("Error in element %d: %f != %f\n", i, C_par[i], C_seq[i]);
            }
        }

        if (errors == 0)
            printf("Equal!\n");
        else
            printf("%d errors.\n", errors);

        printf("Time sequence: %f sec.\n", t_seq);
        printf("Time parallel (max): %f sec.\n", max_comp_time);
        printf("Speed: %.2fx\n", t_seq / max_comp_time);

        free(A);
        free(C_par);
        free(C_seq);
    }

    free(local_A);
    free(local_C);
    free(B);
    if (remainder != 0) {
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}