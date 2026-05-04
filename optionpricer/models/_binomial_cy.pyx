# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
import numpy as np
cimport numpy as np

cdef void _solve_single_binomial(
    int k, double S, double K,
    double u, double d, double df, double p,
    int N, bint is_call, bint american,
    double* option, double[:] result_arr
) nogil:
    cdef int i, j
    cdef double intrinsic, S_ij
    
    for j in range(N + 1):
        S_ij = S * (u ** j) * (d ** (N - j))
        if is_call:
            option[j] = S_ij - K if S_ij > K else 0.0
        else:
            option[j] = K - S_ij if K > S_ij else 0.0

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[j] = df * (p * option[j + 1] + (1.0 - p) * option[j])
            if american:
                S_ij = S * (u ** j) * (d ** (i - j))
                if is_call:
                    intrinsic = S_ij - K if S_ij > K else 0.0
                else:
                    intrinsic = K - S_ij if K > S_ij else 0.0
                    
                if intrinsic > option[j]:
                    option[j] = intrinsic
                    
    result_arr[k] = option[0]

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
    cdef int k
    cdef int num_options = S_arr.shape[0]
    cdef double* all_options = <double*>malloc(num_options * (N + 1) * sizeof(double))
    
    for k in prange(num_options, nogil=True):
        _solve_single_binomial(
            k, S_arr[k], K_arr[k],
            u, d, df, p, N, is_call, american,
            all_options + k * (N + 1), result_arr
        )
        
    free(all_options)
