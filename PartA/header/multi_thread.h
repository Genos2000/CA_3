#include <assert.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <linux/unistd.h>
#include <nmmintrin.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// Create other necessary functions here
int Np, *matAp, *matBp, *A, *B, *outputp, t_count, tile_size;
pthread_mutex_t lock;

void *parallel_Bpre(void *arg) {
    int N = Np, Nby2 = Np / 2, index, indexN;
    int i1 = ((int *)arg)[0];
    int startB = i1 * tile_size;
    int endB = startB + tile_size;
    for (int i = startB; i < endB; i += 2) {
        for (int j = 0; j < N; ++j) {
            B[i * Nby2 + j] = matBp[j * N + i] + matBp[j * N + i + 1];
        }
    }
    free(arg);
    return NULL;
}

void *parallel_Apre(void *arg) {
    int N = Np, Nby2 = Np / 2, index, indexN;
    int i1 = ((int *)arg)[0];
    int startA = i1 * tile_size;
    int endA = startA + tile_size;
    for (int i = startA; i < endA; i += 2) {
        index = i * N;
        indexN = index + N;
        for (int j = 0; j < N; j += 8) {
            if (indexN + j > N * N) {
                cout << "out of bounds";
            }
            __m256i a0 = _mm256_loadu_si256((__m256i *)&matAp[index + j]);
            __m256i b0 = _mm256_loadu_si256((__m256i *)&matAp[indexN + j]);

            __m256i m = _mm256_add_epi32(a0, b0);

            _mm256_storeu_si256((__m256i *)&A[i / 2 * N + j], m);
        }
    }

    free(arg);
    return NULL;
}

void *parallel_vectMul(void *arg) {

    int o;
    alignas(32) int temp[8];

    int Nby2 = Np / 2;
    int i1 = ((int *)arg)[0], j1 = ((int *)arg)[1];
    for (int i = i1 * tile_size / 2; i < i1 * tile_size / 2 + tile_size / 2;
         i++) {
        for (int j = j1 * tile_size / 2; j < j1 * tile_size / 2 + tile_size / 2;
             j++) {
            __m256i o0 = _mm256_setzero_si256();

            for (int k = 0; k < Np; k += 8) {

                __m256i a0 = _mm256_loadu_si256((__m256i *)&A[i * Np + k]);
                __m256i b0 = _mm256_loadu_si256((__m256i *)&B[j * Np + k]);
                __m256i c0 = _mm256_mullo_epi32(a0, b0);
                o0 = _mm256_add_epi32(o0, c0);
            }
            o = 0;

            _mm256_storeu_si256((__m256i *)&temp, o0);

            for (int z = 0; z < 8; z++) {
                o += temp[z];
            }
            outputp[i * Nby2 + j] = o;
        }
    }

    return NULL;
}
void *parallel_RefMul(void *arg) {

    int o;
    alignas(32) int temp[8];

    int Nby2 = Np / 2;
    int i1 = ((int *)arg)[0], j1 = ((int *)arg)[1];
    for (int i = i1 * tile_size / 2; i < i1 * tile_size / 2 + tile_size / 2;
         i++) {
        for (int j = j1 * tile_size / 2; j < j1 * tile_size / 2 + tile_size / 2;
             j++) {
            __m256i o0 = _mm256_setzero_si256();

            for (int k = 0; k < Np; k += 8) {

                __m256i a0 = _mm256_loadu_si256((__m256i *)&A[i * Np + k]);
                __m256i b0 = _mm256_loadu_si256((__m256i *)&B[j * Np + k]);
                __m256i c0 = _mm256_mullo_epi32(a0, b0);
                o0 = _mm256_add_epi32(o0, c0);
            }
            o = 0;

            _mm256_storeu_si256((__m256i *)&temp, o0);

            for (int z = 0; z < 8; z++) {
                o += temp[z];
            }
            outputp[i * Nby2 + j] = o;
        }
    }

    return NULL;
}

void multi_ref(int N, int *matA, int *matB, int *output, int *A1, int *B1) {
    A = A1;
    // int *B1 = new int[N * N / 2];
    B = B1;
    Np = N;
    matAp = matA;
    matBp = matB;
    outputp = output;

    int Nby2 = N / 2;

    int i, j, k;
    t_count = 8;
    tile_size = N / t_count;

    pthread_t threads[t_count][t_count];
    for (i = 0; i < t_count; i++) {
        for (j = 0; j < t_count; j++) {
            int *s;
            s = (int *)malloc(sizeof(int) * 2);
            s[0] = i;
            s[1] = j;

            pthread_create(&threads[i][j], NULL, &parallel_RefMul, s);
        }
    }

    for (i = 0; i < t_count; i++) {
        for (j = 0; j < t_count; j++) {
            pthread_join(threads[i][j], NULL);
        }
    }
    return;
}

void vectMul_multi(int N, int *matA, int *matB, int *output, int *A1, int *B1) {
    // int *A1 = new int[N / 2 * N];
    A = A1;
    // int *B1 = new int[N * N / 2];
    B = B1;
    Np = N;
    matAp = matA;
    matBp = matB;
    outputp = output;

    int Nby2 = N / 2;

    int i, j, k;
    t_count = 4;
    tile_size = N / t_count;

    pthread_t threads[t_count][t_count];
    pthread_t threadsA[t_count];
    pthread_t threadsB[t_count];

    for (j = 0; j < t_count; j++) {
        int *s;
        s = (int *)malloc(sizeof(int) * 1);
        s[0] = j;
        pthread_create(&threadsB[j], NULL, &parallel_Bpre, s);
    }

    for (j = 0; j < t_count; j++) {
        pthread_join(threadsB[j], NULL);
    }

    for (j = 0; j < t_count; j++) {
        int *s;
        s = (int *)malloc(sizeof(int) * 1);
        s[0] = j;
        pthread_create(&threadsA[j], NULL, &parallel_Apre, s);
    }

    for (j = 0; j < t_count; j++) {
        pthread_join(threadsA[j], NULL);
    }

    for (i = 0; i < t_count; i++) {
        for (j = 0; j < t_count; j++) {
            int *s;
            s = (int *)malloc(sizeof(int) * 2);
            s[0] = i;
            s[1] = j;

            pthread_create(&threads[i][j], NULL, &parallel_vectMul, s);
        }
    }

    for (i = 0; i < t_count; i++) {
        for (j = 0; j < t_count; j++) {
            pthread_join(threads[i][j], NULL);
        }
    }
    return;
}

void *parallel_ref(void *arg) {
    int i = ((int *)arg)[0], j = ((int *)arg)[1];

    int sum, indexC, Nby2 = Np >> 1;
    int rowA, iter, colB, rowAby2, rowAN, rowA1N;

    for (rowA = 0; rowA < tile_size; rowA += 2) {
        rowAby2 = ((i + rowA) >> 1) * Nby2;
        rowAN = (i + rowA) * Np;
        rowA1N = rowAN + Np;
        for (iter = 0; iter < Np; ++iter) {
            for (colB = 0; colB < tile_size; colB += 2) {
                sum = 0;
                indexC = rowAby2 + ((colB + j) >> 1);
                sum += matAp[rowAN + iter] * matBp[iter * Np + colB + j];
                sum += matAp[rowA1N + iter] * matBp[iter * Np + colB + j];
                sum += matAp[rowAN + iter] * matBp[iter * Np + (colB + 1 + j)];
                sum += matAp[rowA1N + iter] * matBp[iter * Np + (colB + 1 + j)];
                outputp[indexC] += sum;
            }
        }
    }

    free(arg);
    return NULL;
}

// Fill in this function
void multiThread(int N, int *matA, int *matB, int *output) {
    int *A1 = new int[N / 2 * N];
    // A = A1;
    int *B1 = new int[N * N / 2];
    vectMul_multi(N, matA, matB, output, A1, B1);

    // // int *A1 = new int[N / 2 * N];
    // A = A1;
    // // int *B1 = new int[N * N / 2];
    // B = B1;
    // Np = N;
    // matAp = matA;
    // matBp = matB;
    // outputp = output;

    // int Nby2 = N / 2;

    // int i, j, k;
    // t_count = 8;
    // tile_size = N / t_count;

    // pthread_t threads[t_count][t_count];
    // pthread_t threadsA[t_count];
    // pthread_t threadsB[t_count];

    // for (j = 0; j < t_count; j++) {
    //     int *s;
    //     s = (int *)malloc(sizeof(int) * 1);
    //     s[0] = j;
    //     pthread_create(&threadsB[j], NULL, &parallel_Bpre, s);
    // }

    // for (j = 0; j < t_count; j++) {
    //     pthread_join(threadsB[j], NULL);
    // }

    // for (j = 0; j < t_count; j++) {
    //     int *s;
    //     s = (int *)malloc(sizeof(int) * 1);
    //     s[0] = j;
    //     pthread_create(&threadsA[j], NULL, &parallel_Apre, s);
    // }

    // for (j = 0; j < t_count; j++) {
    //     pthread_join(threadsA[j], NULL);
    // }

    // for (i = 0; i < t_count; i++) {
    //     for (j = 0; j < t_count; j++) {
    //         int *s;
    //         s = (int *)malloc(sizeof(int) * 2);
    //         s[0] = i;
    //         s[1] = j;

    //         pthread_create(&threads[i][j], NULL, &parallel_vectMul, s);
    //     }
    // }

    // for (i = 0; i < t_count; i++) {
    //     for (j = 0; j < t_count; j++) {
    //         pthread_join(threads[i][j], NULL);
    //     }
    // }
    int Nby2 = N / 2;
    if (N <= 8) {
        cerr << "matA multi: " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < N; j++) {
                cerr << A[i * N + j] << "\t";
            }
            cerr << endl;
        }

        cerr << "matB multi: " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < N; j++) {
                cerr << B[i * N + j] << "\t";
            }
            cerr << endl;
        }

        cerr << "output multi: " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < Nby2; j++) {
                cerr << output[i * Nby2 + j] << "\t";
            }
            cerr << endl;
        }
    }
}
