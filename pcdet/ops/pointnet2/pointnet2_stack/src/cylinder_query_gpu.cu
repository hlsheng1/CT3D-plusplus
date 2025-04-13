#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
// #include <cstdlib>
// #include <ctime>
using namespace std;

#include "cylinder_query_gpu.h"
#include "cuda_utils.h"


__global__ void cylinder_query_kernel_stack(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the cylinder query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;

    // int *A;
    // A = (int*)malloc(n*sizeof(int));
    // int tmp;
    // int tc;
    // for (int tc=0; tc<n; tc++)
    // {
    //     A[tc] = tc;
    // }
    // tmp = rand();
    // // srand((int)time(NULL));
    // // for (int tc=0; tc<n; tc++)
    // // {
    // //     tmp=rand();
    // //     // tmp = 0;
    // //     int val=A[tc];
    // //     A[tc]=A[tmp];
    // //     A[tmp]=val;
    // // }

    for (int kraw = 0; kraw < n; ++kraw) {
        // ktmp = rand()%10;
        // k = (kraw * ktmp) % n
        k = kraw;
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;
}


void cylinder_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the cylinder query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    cylinder_query_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
