# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, exp, fmax
from cython.parallel cimport prange
import numpy as np
cimport numpy as np

cdef void _solve_single_fdm(
    int k,
    double S_0, double K, double T, double r, double sigma, double q,
    double* alpha, double* beta, double* gamma,
    double* V_old, double* V_new, double* RHS,
    int M, int N, bint is_call, bint american,
    double omega, double tol, int max_iter,
    double[:] result_arr
) nogil:
    cdef int i, j, iter_count
    cdef double dt, dS, S_max, S_j, tau
    cdef double error, diff, y, intrinsic

    S_max = S_0 * exp((r - q) * T + 4.0 * sigma * (T**0.5))
    dS = S_max / M
    dt = T / N
    
    for j in range(1, M):
        alpha[j] = 0.25 * dt * ((sigma * sigma * j * j) - (r - q) * j)
        beta[j]  = -0.5 * dt * ((sigma * sigma * j * j) + r)
        gamma[j] = 0.25 * dt * ((sigma * sigma * j * j) + (r - q) * j)

    for j in range(M + 1):
        S_j = j * dS
        if is_call:
            V_old[j] = S_j - K if S_j > K else 0.0
        else:
            V_old[j] = K - S_j if K > S_j else 0.0
        V_new[j] = V_old[j]

    for i in range(N - 1, -1, -1):
        tau = (N - i) * dt
        if is_call:
            V_new[0] = 0.0
            V_new[M] = fmax(S_max * exp(-q * tau) - K * exp(-r * tau), 0.0)
        else:
            V_new[0] = K * exp(-r * tau)
            V_new[M] = 0.0
        
        for j in range(1, M):
            RHS[j] = alpha[j] * V_old[j-1] + (1.0 + beta[j]) * V_old[j] + gamma[j] * V_old[j+1]
        
        iter_count = 0
        error = tol + 1.0
        
        while error > tol and iter_count < max_iter:
            error = 0.0
            for j in range(1, M):
                S_j = j * dS
                y = (RHS[j] + alpha[j] * V_new[j-1] + gamma[j] * V_new[j+1]) / (1.0 - beta[j])
                y = V_new[j] + omega * (y - V_new[j])
                
                if american:
                    if is_call:
                        intrinsic = S_j - K if S_j > K else 0.0
                    else:
                        intrinsic = K - S_j if K > S_j else 0.0
                    y = fmax(y, intrinsic)
                
                diff = fabs(V_new[j] - y)
                if diff > error: 
                    error = diff
                
                V_new[j] = y
            
            iter_count = iter_count + 1
        
        for j in range(M + 1):
            V_old[j] = V_new[j]
    
    j = int(S_0 / dS)
    if j >= M:
        result_arr[k] = V_old[M]
    else:
        result_arr[k] = V_old[j] + (V_old[j+1] - V_old[j]) * (S_0 - j * dS) / dS

cpdef void _crank_nicolson_psor_vectorized(
    double[:] S_arr, double[:] K_arr, double[:] T_arr,
    double[:] r_arr, double[:] sigma_arr, double[:] q_arr,
    double[:] result_arr, int M, int N, bint is_call, bint american,
    double omega, double tol, int max_iter
):
    cdef int k
    cdef int num_options = S_arr.shape[0]
    
    cdef double* all_alpha = <double*>malloc(num_options * (M + 1) * sizeof(double))
    cdef double* all_beta  = <double*>malloc(num_options * (M + 1) * sizeof(double))
    cdef double* all_gamma = <double*>malloc(num_options * (M + 1) * sizeof(double))
    cdef double* all_V_old = <double*>malloc(num_options * (M + 1) * sizeof(double))
    cdef double* all_V_new = <double*>malloc(num_options * (M + 1) * sizeof(double))
    cdef double* all_RHS   = <double*>malloc(num_options * (M + 1) * sizeof(double))
    
    for k in prange(num_options, nogil=True):
        _solve_single_fdm(
            k, S_arr[k], K_arr[k], T_arr[k], r_arr[k], sigma_arr[k], q_arr[k],
            all_alpha + k * (M + 1),
            all_beta + k * (M + 1),
            all_gamma + k * (M + 1),
            all_V_old + k * (M + 1),
            all_V_new + k * (M + 1),
            all_RHS + k * (M + 1),
            M, N, is_call, american, omega, tol, max_iter, result_arr
        )
        
    free(all_alpha)
    free(all_beta)
    free(all_gamma)
    free(all_V_old)
    free(all_V_new)
    free(all_RHS)
