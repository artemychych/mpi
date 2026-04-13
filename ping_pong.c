#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define ITERATIONS 50
#define WARMUP      5

const int sizes[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048
};
const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

double measure_one_mode(int rank, int mode, int msg_size, int iter);

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Program requires exactly 2 processes.\n");
            fprintf(stderr, "Usage: mpiexec -n 2 ./ping_pong\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Ping-Pong benchmark: MPI_Send, MPI_Ssend, MPI_Bsend, MPI_Rsend\n");
        printf("Bytes\tSend (ms)\tSsend (ms)\tBsend (ms)\tRsend (ms)\n");
        printf("--------------------------------------------------------------------\n");
    }

    for (int i = 0; i < num_sizes; i++) {
        int msg_size = sizes[i];
        double t_send, t_ssend, t_bsend, t_rsend;

        t_send   = measure_one_mode(rank, 0, msg_size, ITERATIONS);
        t_ssend  = measure_one_mode(rank, 1, msg_size, ITERATIONS);
        t_bsend  = measure_one_mode(rank, 2, msg_size, ITERATIONS);
        t_rsend  = measure_one_mode(rank, 3, msg_size, ITERATIONS);

        if (rank == 0) {
            printf("%-12d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
                   msg_size, t_send, t_ssend, t_bsend, t_rsend);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}

double measure_one_mode(int rank, int mode, int msg_size, int iter) {
    char *send_buf = (char*)malloc(msg_size);
    char *recv_buf = (char*)malloc(msg_size);
    if (!send_buf || !recv_buf) {
        fprintf(stderr, "Memory allocation failed for size %d\n", msg_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    memset(send_buf, 'A', msg_size);
    memset(recv_buf, 0, msg_size);

    char *bsend_buffer = NULL;
    int bsend_size = 0;
    if (mode == 2) {
        int total_sends = ITERATIONS + WARMUP;
        bsend_size = total_sends * (msg_size + MPI_BSEND_OVERHEAD) + 1024 * 1024;
        bsend_buffer = (char*)malloc(bsend_size);
        if (!bsend_buffer) {
            fprintf(stderr, "BSend buffer allocation failed for size %d\n", msg_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Buffer_attach(bsend_buffer, bsend_size);
    }

    for (int w = 0; w < WARMUP; w++) {
        if (rank == 0) {
            switch (mode) {
                case 0: MPI_Send(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 1: MPI_Ssend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 2: MPI_Bsend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 3: MPI_Rsend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
            }
            MPI_Recv(recv_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 1) {
            MPI_Recv(recv_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            switch (mode) {
                case 0: MPI_Send(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 1: MPI_Ssend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 2: MPI_Bsend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 3: MPI_Rsend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < iter; i++) {
        if (rank == 0) {
            switch (mode) {
                case 0: MPI_Send(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 1: MPI_Ssend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 2: MPI_Bsend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
                case 3: MPI_Rsend(send_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); break;
            }
            MPI_Recv(recv_buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == 1) {
            MPI_Recv(recv_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            switch (mode) {
                case 0: MPI_Send(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 1: MPI_Ssend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 2: MPI_Bsend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
                case 3: MPI_Rsend(send_buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD); break;
            }
        }
    }

    double end = MPI_Wtime();
    double elapsed = (end - start) / iter;

    MPI_Barrier(MPI_COMM_WORLD);
    if (mode == 2) {
        void *tmp;
        int tmp_size;
        MPI_Buffer_detach(&tmp, &tmp_size);
        free(bsend_buffer);
    }
    free(send_buf);
    free(recv_buf);

    MPI_Barrier(MPI_COMM_WORLD);
    return elapsed * 1e6;
}