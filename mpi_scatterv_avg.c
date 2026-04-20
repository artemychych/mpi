#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendcounts = NULL;
    int *displs = NULL;
    int total = 0;
    double *full_array = NULL;

    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        total = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i + 1) * 100;
            total += sendcounts[i];
        }

        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }

        full_array = (double*)malloc(total * sizeof(double));
        for (int i = 0; i < total; i++) {
            full_array[i] = (double)i;
        }
    }

    int my_count;
    MPI_Scatter(sendcounts, 1, MPI_INT, &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double *my_data = (double*)malloc(my_count * sizeof(double));

    MPI_Scatterv(full_array, sendcounts, displs, MPI_DOUBLE,
                 my_data, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (int i = 0; i < my_count; i++) {
        local_sum += my_data[i];
    }

    double local_avg = local_sum / my_count;

    printf("Process %d: get %d elements, avg = %f\n",
           rank, my_count, local_avg);

    double *all_avgs = NULL;

    if (rank == 0) {
        all_avgs = (double*)malloc(size * sizeof(double));
    }
    MPI_Gather(&local_avg, 1, MPI_DOUBLE,
               all_avgs, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double weighted_sum = 0.0;
        for (int i = 0; i < size; i++) {
            weighted_sum += all_avgs[i] * sendcounts[i];
        }
        double overall_avg = weighted_sum / total;
        printf("\nAll avg vzv = %f\n", overall_avg);

        double seq_sum = 0.0;
        for (int i = 0; i < total; i++) {
            seq_sum += full_array[i];
        }
        double seq_avg = seq_sum / total;
        printf("serial avg = %f\n", seq_avg);
        printf("offset = %e\n", overall_avg - seq_avg);

        free(sendcounts);
        free(displs);
        free(full_array);
        free(all_avgs);
    }

    free(my_data);
    MPI_Finalize();
    return 0;
}