#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

typedef struct {
    unsigned int x, y, z;
} dim3;

void matrixMulCPU(float *C, float *A, float *B, int wA, int wB, int hA, int wC, int hC)
{
    for (int i = 0; i < hC; i++) {
        for (int j = 0; j < wC; j++) {
            float sum = 0;
            for (int k = 0; k < wA; k++) {
                sum += A[i * wA + k] * B[k * wB + j];
            }
            C[i * wC + j] = sum;
        }
    }
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

int matrixMultiply(int argc, char **argv, int block_size, dim3 dimsA, dim3 dimsB)
{
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    unsigned int size_C = dimsA.y * dimsB.x;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = (float *) malloc(mem_size_C);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }

    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    printf("Allocating device memory on host...\n");

    float *d_A = (float *)malloc(mem_size_A);
    if (d_A == NULL) {
        fprintf(stderr, "Failed to allocate device matrix A!\n");
        exit(EXIT_FAILURE);
    }

    float *d_B = (float *)malloc(mem_size_B);
    if (d_B == NULL) {
        fprintf(stderr, "Failed to allocate device matrix B!\n");
        exit(EXIT_FAILURE);
    }

    float *d_C = (float *)malloc(mem_size_C);
    if (d_C == NULL) {
        fprintf(stderr, "Failed to allocate device matrix C!\n");
        exit(EXIT_FAILURE);
    }

    printf("Copying input data\n");
    memcpy(d_A, h_A, mem_size_A);
    memcpy(d_B, h_B, mem_size_B);

    dim3 threads;
    threads.x = block_size;
    threads.y = block_size;
    dim3 grid;
    grid.x = dimsB.x / threads.x;
    grid.y = dimsA.y / threads.y;

    printf("Computing result using CPU...\n");

    printf("done\n");

    clock_t start = clock();

    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        matrixMulCPU(d_C, d_A, d_B, dimsA.x, dimsB.x, dimsA.y, dimsB.x, dimsA.y);
    }

    clock_t stop = clock();

    printf("Copying output data\n");
    memcpy(h_C, d_C, mem_size_C);

    double msecPerMatrixMul = ((double)(stop - start) / CLOCKS_PER_SEC) * 1000.0 / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    printf("Checking computed result for correctness: ");
    bool correct = true;

    double eps = 1.e-6;
    for (int i = 0; i < size_C; i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    printf("shutting down...\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(d_A);
    free(d_B);
    free(d_C);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}

int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CPU] - Starting...\n");

    int block_size = 32;
    dim3 dimsA, dimsB;

    dimsA.x = 5 * 2 * block_size;
    dimsA.y = 5 * 2 * block_size;
    dimsB.x = 5 * 4 * block_size;
    dimsB.y = 5 * 2 * block_size;

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    return matrix_result;
}