# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cpdef double _build_tree_cython(
    double[:] option, 
    double[:] S_T, 
    double[:] u_pows, 
    double K, 
    double df, 
    double p, 
    int N, 
    bint is_call, 
    bint american
):
    cdef int i, j
    cdef double scalar, S_ij, intrinsic

    for i in range(N - 1, -1, -1):
        scalar = u_pows[N - i]
        for j in range(i + 1):
            option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
            
            if american:
                S_ij = S_T[j] * scalar
                if is_call:
                    intrinsic = S_ij - K if S_ij > K else 0.0
                else:
                    intrinsic = K - S_ij if K > S_ij else 0.0
                    
                if intrinsic > option[j]:
                    option[j] = intrinsic
                    
    return option[0]
