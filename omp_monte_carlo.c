#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static unsigned int lcg_rand(unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

int main() {
    const long long total_points = 100000000LL;
    long long inside_circle = 0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:inside_circle)
    for (long long i = 0; i < total_points; i++) {
        unsigned int seed = omp_get_thread_num() * 123456789 + i * 987654321;

        double x = (double)lcg_rand(&seed) / 0x7fffffff * 2.0 - 1.0;
        double y = (double)lcg_rand(&seed) / 0x7fffffff * 2.0 - 1.0;

        if (x * x + y * y <= 1.0)
            inside_circle++;
    }

    double pi_estimate = 4.0 * inside_circle / total_points;
    double time = omp_get_wtime() - start;

    printf("PI      = %.10f\n", pi_estimate);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("Time    = %.4f seconds\n", time);

    return 0;
}