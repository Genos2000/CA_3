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
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end)                                            \
    std::chrono::duration_cast<gran>(end - start).count()

// perf counter syscall
int perf_event_open(struct perf_event_attr *hw, pid_t pid, int cpu, int grp,
                    unsigned long flags) {
    return syscall(__NR_perf_event_open, hw, pid, cpu, grp, flags);
}
#define is_aligned(POINTER, BYTE_COUNT)                                        \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

struct perf_event_attr attr[10];
uint64_t val1[10], val2[10];
int fd[10], rc;
auto start = TIME_NOW, endT = TIME_NOW;

void printStats() {

    endT = TIME_NOW;

    // Read the counter
    asm volatile("nop;"); // pseudo-barrier
    rc = read(fd[0], &val2[0], sizeof(val2[0]));
    assert(rc);
    rc = read(fd[1], &val2[1], sizeof(val2[1]));
    assert(rc);
    rc = read(fd[2], &val2[2], sizeof(val2[2]));
    assert(rc);
    rc = read(fd[3], &val2[3], sizeof(val2[3]));
    assert(rc);
    rc = read(fd[4], &val2[4], sizeof(val2[4]));
    assert(rc);
    rc = read(fd[5], &val2[5], sizeof(val2[5]));
    assert(rc);
    rc = read(fd[6], &val2[6], sizeof(val2[6]));
    assert(rc);
    rc = read(fd[7], &val2[7], sizeof(val2[7]));
    assert(rc);
    rc = read(fd[8], &val2[8], sizeof(val2[8]));
    assert(rc);

    asm volatile("nop;"); // pseudo-barrier

    // Close the counter
    close(fd[0]);
    close(fd[1]);
    close(fd[2]);
    close(fd[3]);
    close(fd[4]);
    close(fd[5]);
    close(fd[6]);
    close(fd[7]);
    close(fd[8]);

    printf("CPU Cycles:           %lu \n", val2[0] - val1[0]);
    printf("Instructions:         %lu \n", val2[3] - val1[3]);
    printf("IPC:                  %lf\n",
           ((double)val2[3] - val1[3]) / (val2[0] - val1[0]));
    printf("Branch misses:        %lu \n", val2[1] - val1[1]);
    printf("Branch instructions:  %lu \n", val2[2] - val1[2]);
    printf("Branch mispred. rate: %lf%%\n",
           100.0 * ((double)val2[1] - val1[1]) / (val2[2] - val1[2]));
    printf("L1-D cache misses :  %lu \n", val2[4] - val1[4]);
    printf("L1-D cache accesses :  %lu \n", val2[5] - val1[5]);
    printf("L1-D cache miss rate :  %f%% \n",
           100 * (double)(val2[4] - val1[4]) / (val2[5] - val1[5]));
    printf("LL cache misses :  %lu \n", val2[6] - val1[6]);
    printf("LL cache accesses :  %lu \n", val2[7] - val1[7]);
    printf("LL cache miss rate :  %f%% \n",
           100 * (double)(val2[6] - val1[6]) / (val2[7] - val1[7]));
    printf("Page Faults:         %lu \n", val2[8] - val1[8]);
    cout << "exec time "
         << (double)TIME_DIFF(std::chrono::microseconds, start, endT) / 1000.0
         << " ms" << endl;
}
void reInitPerf() {
    start = TIME_NOW;
    attr[0].type = PERF_TYPE_HARDWARE;
    attr[0].config = PERF_COUNT_HW_CPU_CYCLES; /* generic PMU event*/
    attr[0].disabled = 0;
    fd[0] = perf_event_open(&attr[0], getpid(), -1, -1, 0);
    if (fd[0] < 0) {
        perror("Opening performance counter");
    }

    attr[1].type = PERF_TYPE_HARDWARE;
    attr[1].config = PERF_COUNT_HW_BRANCH_MISSES; /* generic PMU event*/
    attr[1].disabled = 0;
    fd[1] = perf_event_open(&attr[1], getpid(), -1, -1, 0);
    if (fd[1] < 0) {
        perror("Opening performance counter");
    }

    attr[2].type = PERF_TYPE_HARDWARE;
    attr[2].config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS; /* generic PMU event*/
    attr[2].disabled = 0;
    fd[2] = perf_event_open(&attr[2], getpid(), -1, -1, 0);
    if (fd[2] < 0) {
        perror("Opening performance counter");
    }

    attr[3].type = PERF_TYPE_HARDWARE;
    attr[3].config = PERF_COUNT_HW_INSTRUCTIONS; /* generic PMU event*/
    attr[3].disabled = 0;
    fd[3] = perf_event_open(&attr[3], getpid(), -1, -1, 0);
    if (fd[3] < 0) {
        perror("Opening performance counter");
    }

    attr[4].type = PERF_TYPE_HW_CACHE;
    // attr[4].config = PERF_COUNT_HW_CACHE_L1D; /* generic PMU event*/
    attr[4].config = (PERF_COUNT_HW_CACHE_L1D) |
                     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    attr[4].disabled = 0;
    fd[4] = perf_event_open(&attr[4], getpid(), -1, -1, 0);
    if (fd[4] < 0) {
        perror("Opening performance counter");
    }

    attr[5].type = PERF_TYPE_HW_CACHE;
    // attr[5].config = PERF_COUNT_HW_CACHE_L1D; /* generic PMU event*/
    attr[5].config = (PERF_COUNT_HW_CACHE_L1D) |
                     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                     (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    attr[5].disabled = 0;
    fd[5] = perf_event_open(&attr[5], getpid(), -1, -1, 0);
    if (fd[5] < 0) {
        perror("Opening performance counter");
    }

    attr[6].type = PERF_TYPE_HW_CACHE;
    attr[6].config = (PERF_COUNT_HW_CACHE_LL) |
                     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    attr[6].disabled = 0;
    fd[6] = perf_event_open(&attr[6], getpid(), -1, -1, 0);
    if (fd[6] < 0) {
        perror("Opening performance counter");
    }

    attr[7].type = PERF_TYPE_HW_CACHE;
    attr[7].config = (PERF_COUNT_HW_CACHE_LL) |
                     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                     (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    attr[7].disabled = 0;
    fd[7] = perf_event_open(&attr[7], getpid(), -1, -1, 0);
    if (fd[7] < 0) {
        perror("Opening performance counter");
    }

    attr[8].type = PERF_TYPE_SOFTWARE;
    attr[8].config = PERF_COUNT_SW_PAGE_FAULTS; /* generic PMU event*/
    attr[8].disabled = 0;
    fd[8] = perf_event_open(&attr[8], getpid(), -1, -1, 0);
    if (fd[8] < 0) {
        perror("Opening performance counter");
    }

    // Tell Linux to start counting events
    asm volatile("nop;"); // pseudo-barrier
    rc = read(fd[0], &val1[0], sizeof(val1[0]));
    assert(rc);
    rc = read(fd[1], &val1[1], sizeof(val1[1]));
    assert(rc);
    rc = read(fd[2], &val1[2], sizeof(val1[2]));
    assert(rc);
    rc = read(fd[3], &val1[3], sizeof(val1[3]));
    assert(rc);
    rc = read(fd[4], &val1[4], sizeof(val1[4]));
    assert(rc);
    rc = read(fd[5], &val1[5], sizeof(val1[5]));
    assert(rc);
    rc = read(fd[6], &val1[6], sizeof(val1[6]));
    assert(rc);
    rc = read(fd[7], &val1[7], sizeof(val1[7]));
    assert(rc);
    rc = read(fd[8], &val1[8], sizeof(val1[8]));
    assert(rc);

    asm volatile("nop;"); // pseudo-barrier
}