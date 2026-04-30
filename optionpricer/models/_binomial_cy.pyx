# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

cpdef void _build_tree_cython_vectorized(
    double[:] S_arr, 
    double[:] K_arr,
    double[:] result_arr,
    double u,
    double d,
    double df, 
    double p, 
    int N, 
    bint is_call, 
    bint american
):
    cdef int i, j, k
    cdef int num_options = S_arr.shape[0]
    cdef double intrinsic, S_ij
    cdef double* option = <double*>malloc((N + 1) * sizeof(double))
    
    for k in range(num_options):
        for j in range(N + 1):
            S_ij = S_arr[k] * (u ** j) * (d ** (N - j))
            if is_call:
                option[j] = S_ij - K_arr[k] if S_ij > K_arr[k] else 0.0
            else:
                option[j] = K_arr[k] - S_ij if K_arr[k] > S_ij else 0.0

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
                if american:
                    S_ij = S_arr[k] * (u ** j) * (d ** (i - j))
                    if is_call:
                        intrinsic = S_ij - K_arr[k] if S_ij > K_arr[k] else 0.0
                    else:
                        intrinsic = K_arr[k] - S_ij if K_arr[k] > S_ij else 0.0
                        
                    if intrinsic > option[j]:
                        option[j] = intrinsic
                        
        result_arr[k] = option[0]
        
    free(option)
