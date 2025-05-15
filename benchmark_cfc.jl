using BenchmarkTools;
using Finch;


## Imagine: Three {0-3}-D matrices
## Operations: + and *? Maybe more? (Ex: min)

function matmul(B, C)
    return B*C
end

function matmul_einsum(B, C)
    return @einsum A[i, j] += B[i, k] * C[k, j]
end

function matadd(B, C)
    return B + C
end

function matadd_einsum(B, C)
    return @einsum A[i, j] += B[i, j] + C[i, j]
end

function matvectmul(B, x)
    return B * x 
end

function matvectmul_einsum(B, x)
    return @einsum y[i] += B[i, k] * x[k]
end

function tensormatmul(B, C)
    return B * C 
end

"""
KERNEL 1
N = 500
!Reshape : 0.972373 s
 Reshape : 1.939    s
"""

# relative to this guy

function tensormatmul_einsum(B, C)
    return @einsum A[i, j, k] += B[i, j, l] * C[l, k]
end

function timed_tensormatmul_einsum(B, C)
    time_A = @elapsed @einsum A[i, j, k] += B[i, j, l] * C[l, k]
    println(time_A)
    return A
end

# 4 kernels
# compare this guy

function tensormatmul_einsum_reshape(B, C)
    B_p = reshape(B, (size(B)[1]*size(B)[2], size(B)[3]))
    @einsum A_p[ij, k] += B_p[ij, l] * C[l, k]
    A = reshape(A_p, (size(B)[1], size(B)[2], size(C)[2]))
    return A
end

function timed_tensormatmul_einsum_reshape(B, C)
    time_B_p = @elapsed B_p = reshape(B, (size(B)[1]*size(B)[2], size(B)[3]))
    time_A_p = @elapsed @einsum A_p[ij, k] += B_p[ij, l] * C[l, k]
    time_A = @elapsed A = reshape(A_p, (size(B)[1], size(B)[2], size(C)[2]))
    println(time_B_p, " ", time_A_p, " ", time_A)
    return A
end

"""
KERNEL 2
N = 100
!Reshape : 133.984  ms
 Reshape : 221.636  ms
"""

function tensortensormul_einsum(B, C)
    return @einsum A[i, j, l, m] += B[i, j, k] * C[k, l, m]
end

function tensortensormul_einsum_reshape(B, C)
    B_p = reshape(B, (size(B)[1]*size(B)[2], size(B)[3]))
    C_p = reshape(C, (size(C)[1], size(C)[2]*size(C)[3]))
    @einsum A_p[ij, lm] += B_p[ij, k] * C_p[k, lm]
    A = reshape(A_p, (size(B)[1], size(B)[2], size(C)[2], size(C)[3]))
    return A
end

"""
KERNEL 3
N = 500
!Reshape : 1.186    s
 Reshape : 0.705953 s
"""

function tensortensorsharedmul_einsum(B, C)
    return @einsum A[i, m] += B[i, j, k] * C[j, k, m]
end

function tensortensorsharedmul_einsum_reshape(B, C)
    B_p = reshape(B, (size(B)[1], size(B)[2]*size(B)[3]))
    C_p = reshape(C, (size(C)[1]*size(C)[2], size(C)[3]))
    @einsum A[i, m] += B_p[i, jk] * C_p[jk, m]
    return A
end

"""
KERNEL 4
N = 50
!Reshape : 161.840 ms
 Reshape : 297.496 ms
"""

function tensortensorsharedmul4d_einsum(B, C)
    return @einsum A[i, j, m, n] += B[i, j, k, l] * C[k, l, m, n]
end

function tensortensorsharedmul4d_einsum_reshape(B, C)
    B_p = reshape(B, (size(B)[1]*size(B)[2], size(B)[3]*size(B)[4]))
    C_p = reshape(C, (size(C)[1]*size(C)[2], size(C)[3]*size(C)[4]))
    @einsum A_p[ij, mn] += B_p[ij, kl] * C_p[kl, mn]
    A = reshape(A_p, (size(B)[1], size(B)[2], size(C)[3], size(C)[4]))
    return A
end


sparsity = 0.01
row = 1_000
col = 1_000
tube = 100
# TODO: different size + sparsity!
A_tensor = fsprand(row, col, tube, sparsity)
B = fsprand(row, tube, col, sparsity)
C = fsprand(col, row, sparsity)
x = fsprand(row, sparsity)

tensormatmul_einsum(B, C)
tensormatmul_einsum_reshape(B, C)

display(sparsity)
display(@benchmark tensormatmul_einsum(B, C))
display(@benchmark tensormatmul_einsum_reshape(B, C))
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))

# sparsity = 0.5
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
# sparsity = 0.75
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
# sparsity = 1.0
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
