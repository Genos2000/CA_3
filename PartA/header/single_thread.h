// Optimize this function
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <linux/unistd.h>
#include <nmmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void vectmul1(int N, int *matA, int *matB, int *output) {
    // arithimetic operation reduction

    assert(N >= 4 and N == (N & ~(N - 1)));
    for (int rowA = 0; rowA < N; rowA += 2) {
        for (int colB = 0; colB < N; colB += 2) {
            int sum = 0;
            for (int iter = 0; iter < N; iter++) {
                sum += (matA[rowA * N + iter] + matA[(rowA + 1) * N + iter]) *
                       (matB[iter * N + colB] + matB[iter * N + (colB + 1)]);
            }

            // compute output indices
            int rowC = rowA >> 1;
            int colC = colB >> 1;
            int indexC = rowC * (N >> 1) + colC;
            output[indexC] = sum;
        }
    }
}
void vectmul2(int N, int *matA, int *matB, int *output) {
    // arithimetic operation reduction

    assert(N >= 4 and N == (N & ~(N - 1)));
    int temp;
    int *A = new int[N * N / 2];
    int *B = new int[N * N / 2];
    for (int i = 0; i < N; i += 2) {
        for (int j = 0; j < N; ++j) {
            B[i / 2 * N + j] = matB[j * N + i] + matB[j * N + i + 1];
        }
    }
    for (int i = 0; i < N; i += 2) {
        for (int j = 0; j < N; ++j) {
            A[i / 2 * N + j] = matA[i * N + j] + matA[(i + 1) * N + j];
        }
    }

    for (int rowA = 0; rowA < N / 2; rowA++) {
        for (int rowB = 0; rowB < N / 2; rowB++) {
            int sum = 0;
            for (int iter = 0; iter < N; iter++) {
                sum += (A[rowA * N + iter]) * (B[rowB * N + iter]);
            }
            output[rowA * N / 2 + rowB] = sum;
        }
    }
}

void vectmul(int N, int Nby2, int *matA, int *matB, int *output) {
    int o;
    int temp[8];
    int *A = new int[Nby2 * N];
    int *B = new int[N * Nby2];

    for (int i = 0; i < N; i += 2) {

        for (int j = 0; j < N; j += 8) {
            __m256i a0 = _mm256_loadu_si256((__m256i *)&matA[i * N + j]);
            __m256i b0 = _mm256_loadu_si256((__m256i *)&matA[(i + 1) * N + j]);
            __m256i m = _mm256_add_epi32(a0, b0);
            _mm256_storeu_si256((__m256i *)&A[i * N / 2 + j], m);
        }
    }

    for (int i = 0; i < N; i += 2) {
        for (int j = 0; j < N; ++j) {
            B[i / 2 * N + j] = matB[j * N + i] + matB[j * N + i + 1];
        }
    }

    for (int i = 0; i < Nby2; i++) {
        for (int j = 0; j < Nby2; j++) {
            __m256i o0 = _mm256_setzero_si256();
            for (int k = 0; k < N; k += 8) {
                __m256i a0 = _mm256_loadu_si256((__m256i *)&A[i * N + k]);
                __m256i b0 = _mm256_loadu_si256((__m256i *)&B[j * N + k]);
                __m256i c0 = _mm256_mullo_epi32(a0, b0);
                o0 = _mm256_add_epi32(o0, c0);
            }
            o = 0;
            _mm256_storeu_si256((__m256i *)&temp, o0);

            for (int z = 0; z < 8; z++) {
                o += temp[z];
            }
            output[i * Nby2 + j] = o;
        }
    }
    // int Nby2 = N / 2;
    if (N <= 16) {
        cerr << "matA : " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < N; j++) {
                cerr << A[i * N + j] << "\t";
            }
            cerr << endl;
        }

        cerr << "matB : " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < N; j++) {
                cerr << B[i * N + j] << "\t";
            }
            cerr << endl;
        }

        cerr << "output : " << endl;
        for (int i = 0; i < Nby2; i++) {
            for (int j = 0; j < Nby2; j++) {
                cerr << output[i * Nby2 + j] << "\t";
            }
            cerr << endl;
        }
    }
}

void tiledmul256(int N, int Nby2, int *matA, int *matB, int *output, int b) {

    int o;
    int temp[16];
    int *A = new int[Nby2 * N];
    int *B = new int[N * Nby2];
    int i, j, k, index, indexN;

    for (i = 0; i < Nby2; i++) {
        index = 2 * i * N;
        indexN = index + N;
        for (j = 0; j < N; j += 8) {
            __m256i a0 = _mm256_loadu_si256((__m256i *)&matA[index + j]);
            __m256i b0 = _mm256_loadu_si256((__m256i *)&matA[indexN + j]);
            __m256i m = _mm256_add_epi32(a0, b0);

            _mm256_storeu_si256((__m256i *)&A[i * N + j], m);
        }
    }

    for (i = 0; i < N; i += 2) {
        for (j = 0; j < N; ++j) {
            B[i / 2 * N + j] = matB[j * N + i] + matB[j * N + i + 1];
        }
    }

    for (int i = 0; i < Nby2; i = i + b) {
        for (int j = 0; j < Nby2; j = j + b) {
            for (int k = 0; k < N; k = k + b) {
                for (int ii = i; ii < i + b; ii++) {
                    for (int jj = j; jj < j + b; jj++) {
                        __m256i o0 = _mm256_setzero_si256();
                        for (int kk = k; kk < k + b; kk += 8) {
                            __m256i a0 =
                                _mm256_loadu_si256((__m256i *)&A[ii * N + kk]);
                            __m256i b0 =
                                _mm256_loadu_si256((__m256i *)&B[jj * N + kk]);
                            __m256i c0 = _mm256_mullo_epi32(a0, b0);
                            o0 = _mm256_add_epi32(o0, c0);
                            // A[ii][jj] += B[ii][kk] * C[kk][jj];
                        }
                        o = 0;
                        _mm256_storeu_si256((__m256i *)&temp, o0);

                        for (int z = 0; z < 8; z++) {
                            o += temp[z];
                        }
                        output[ii * Nby2 + jj] += o;
                    }
                }
            }
        }
    }
}
void singleThread(int N, int *matA, int *matB, int *output) {

    assert(N >= 4 and N == (N & ~(N - 1)));
    vectmul(N, N / 2, matA, matB, output);
    // cout << "Block size: " << b << endl;
    // if (b < 8) {
    //     // reInitPerf();
    //     vectmul1(N, matA, matB, output);
    //     // cout << "\n stats for opti 1 \n";
    //     // printStats();
    //     // reInitPerf();
    //     vectmul2(N, matA, matB, output);
    //     // cout << "\n stats for opti 2 \n";
    //     // printStats();
    //     // reInitPerf();
    //     vectmul(N, N / 2, matA, matB, output);
    //     // cout << "\n stats for opti 3 \n";
    //     // printStats();
    // } else {
    //     // reInitPerf();
    //     tiledmul256(N, N / 2, matA, matB, output, b);
    //     // printStats();
    // }
    // int Nby2 = N / 2;
    // if (N <= 8) {
    //     cout << "A: " << endl;
    //     for (int i = 0; i < Nby2; i++) {
    //         for (int j = 0; j < N; j++) {
    //             cerr << A[i * N + j] << "\t";
    //         }
    //         cerr << endl;
    //     }
    //     cerr << "matB: " << endl;
    //     for (int i = 0; i < Nby2; i++) {
    //         for (int j = 0; j < N; j++) {
    //             cerr << B[i * N + j] << "\t";
    //         }
    //         cerr << endl;
    //     }

    //     cerr << "output: " << endl;
    //     for (int i = 0; i < Nby2; i++) {
    //         for (int j = 0; j < Nby2; j++) {
    //             cerr << output[i * Nby2 + j] << "\t";
    //         }
    //         cerr << endl;
    //     }
    // }
}