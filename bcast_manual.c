#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 5000000

int main(int argc, char *argv[]) {
    int rank, size;
    int *array = NULL;
    double start, end, bcast_time, manual_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Process %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = i;
        }
    }

    // ручная рассылка
    if (rank != 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) array[i] = -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start = MPI_Wtime();

    if (rank == 0) {
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(array, ARRAY_SIZE, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(array, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        manual_time = end - start;
    }

    // bcast
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start = MPI_Wtime();

    MPI_Bcast(array, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        bcast_time = end - start;
    }

    

    if (rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Array size: %d integers (%.2f MB)\n", ARRAY_SIZE,
               (double)(ARRAY_SIZE * sizeof(int)) / (1024 * 1024));
        printf("MPI_Bcast time:    %.6f seconds\n", bcast_time);
        printf("Manual Send/Recv time: %.6f seconds\n", manual_time);
        printf("Speedup (Bcast vs manual): %.2f\n", manual_time / bcast_time);
    }

    free(array);
    MPI_Finalize();
    return 0;
}