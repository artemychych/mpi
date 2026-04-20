#include <stdio.h>
#include "mpi.h"
#include <omp.h>

#define GLOBAL_N 100000000

int main(int argc, char *argv[]) {
    int rank, size;
    double total_sum = 0.0;
    double start_time = 0.0, end_time = 0.0;
    int N_per_process;
    double h;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N_per_process = GLOBAL_N / size;
    if (N_per_process * size != GLOBAL_N) {
        if (rank == 0) {
            fprintf(stderr, "Error: GLOBAL_N should by n_per_proc.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    h = 1.0 / GLOBAL_N;
    double start_x = rank * (1.0 / size);
    double local_sum = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    #pragma omp parallel reduction(+:local_sum)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int N_per_thread = N_per_process / num_threads;

        if (N_per_thread * num_threads != N_per_process) {
            #pragma omp single
            if (rank == 0) {
                fprintf(stderr, "Error: n_per_process cant substract num_threads\n");
            }
        }

        double thread_start = start_x + thread_id * N_per_thread * h;
        double local_thread_sum = 0.0;

        for (int i = 0; i < N_per_thread; i++) {
            double x = thread_start + (i + 0.5) * h;
            local_thread_sum += (4.0 / (1.0 + x * x)) * h;
        }
        local_sum += local_thread_sum;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) end_time = MPI_Wtime();

    if (rank == 0) {
        total_sum = local_sum;
        double recv_sum;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }
        int num_threads = omp_get_max_threads();
        double elapsed = end_time - start_time;
        printf("p=%d, t=%d, time=%.6f s, pi=%.15f\n", size, num_threads, elapsed, total_sum);
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}