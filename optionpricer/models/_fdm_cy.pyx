# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, exp, fmax
import numpy as np
cimport numpy as np

cpdef void _crank_nicolson_psor_vectorized(
    double[:] S_arr, 
    double[:] K_arr,
    double[:] T_arr,
    double[:] r_arr,
    double[:] sigma_arr,
    double[:] q_arr,
    double[:] result_arr,
    int M, 
    int N, 
    bint is_call, 
    bint american,
    double omega,
    double tol,
    int max_iter
):
    cdef int i, j, k, iter_count
    cdef int num_options = S_arr.shape[0]
    
    cdef double dt, dS, S_max, S_j, tau
    cdef double error, diff, y, intrinsic
    
    cdef double* alpha = <double*>malloc((M + 1) * sizeof(double))
    cdef double* beta  = <double*>malloc((M + 1) * sizeof(double))
    cdef double* gamma = <double*>malloc((M + 1) * sizeof(double))
    
    cdef double* V_old = <double*>malloc((M + 1) * sizeof(double))
    cdef double* V_new = <double*>malloc((M + 1) * sizeof(double))
    cdef double* RHS   = <double*>malloc((M + 1) * sizeof(double))
    
    cdef double K, T, r, sigma, q, S_0
    
    for k in range(num_options):
        K = K_arr[k]
        T = T_arr[k]
        r = r_arr[k]
        sigma = sigma_arr[k]
        q = q_arr[k]
        S_0 = S_arr[k]
        
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
                
                iter_count += 1
            
            for j in range(M + 1):
                V_old[j] = V_new[j]
        
        j = int(S_0 / dS)
        if j >= M:
            result_arr[k] = V_old[M]
        else:
            result_arr[k] = V_old[j] + (V_old[j+1] - V_old[j]) * (S_0 - j * dS) / dS
        
    free(alpha)
    free(beta)
    free(gamma)
    free(V_old)
    free(V_new)
    free(RHS)
