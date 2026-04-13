#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    int message;
    int next, prev;
    int tag = 100;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    next = (rank + 1) % size;
    prev = (rank + size - 1) % size;

    printf("Rank: %d, Size: %d\n", rank, size);

    if (rank == 0) {
        srand(time(NULL));
        message = rand() % 100;
        printf("Process 0: Initial random number = %d\n", message);
    }

    if (rank == 0) {
        MPI_Send(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
        MPI_Recv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0: Received final number = %d\n", message);
    } else {
        MPI_Recv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: Received number = %d\n", next, message);
        message += 1;
        MPI_Send(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
