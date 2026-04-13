#include <stdio.h>
#include "mpi.h"

#define LOCAL_N 1000000

int main(int argc, char *argv[]) {
    int rank, size;
    double local_sum = 0.0;
    double total_sum = 0.0;
    double h, start, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    h = 1.0 / (size * LOCAL_N);
    start = rank * (1.0 / size);

    for (int i = 0; i < LOCAL_N; i++) {
        x = start + (i + 0.5) * h;
        local_sum += (4.0 / (1.0 + x * x)) * h;
    }

    if (rank == 0) {
        total_sum = local_sum;
        double recv_sum;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }
        printf("pi = %.15f\n", total_sum);
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}