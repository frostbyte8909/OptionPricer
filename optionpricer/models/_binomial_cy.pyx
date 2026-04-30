# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free

cpdef double _build_tree_cython(
    double S, 
    double K, 
    double u,
    double d,
    double df, 
    double p, 
    int N, 
    bint is_call, 
    bint american
):
    cdef int i, j
    cdef double intrinsic, S_ij, result
    cdef double* option = <double*>malloc((N + 1) * sizeof(double))
    
    for j in range(N + 1):
        S_ij = S * (u ** (N - j)) * (d ** j)
        if is_call:
            option[j] = S_ij - K if S_ij > K else 0.0
        else:
            option[j] = K - S_ij if K > S_ij else 0.0

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
            
            if american:
                S_ij = S * (u ** (i - j)) * (d ** j)
                if is_call:
                    intrinsic = S_ij - K if S_ij > K else 0.0
                else:
                    intrinsic = K - S_ij if K > S_ij else 0.0
                    
                if intrinsic > option[j]:
                    option[j] = intrinsic
                    
    result = option[0]
    free(option)
    return result
