/**
 * @file slavsy.c
 * @brief SLAVSY performs one of the matrix-vector operations
 *        x := A*x or x := A'*x, where A is a factor from the
 *        block U*D*U' or L*D*L' factorization computed by SSYTRF.
 */

#include <math.h>
#include "verify.h"
#include "semicolon_lapack_single.h"
#include <cblas.h>

/* xerbla declared in verify.h */

/**
 * SLAVSY performs one of the matrix-vector operations
 *    x := A*x  or  x := A'*x,
 * where x is an N element vector and A is one of the factors
 * from the block U*D*U' or L*D*L' factorization computed by SSYTRF.
 *
 * If TRANS = 'N', multiplies by U  or U * D  (or L  or L * D)
 * If TRANS = 'T', multiplies by U' or D * U' (or L' or D * L')
 * If TRANS = 'C', multiplies by U' or D * U' (or L' or D * L')
 *
 * @param[in]     uplo   'U': Upper triangular, 'L': Lower triangular
 * @param[in]     trans  'N': No transpose, 'T'/'C': Transpose
 * @param[in]     diag   'U': Unit diagonal (A = U or L only),
 *                        'N': Non-unit (A = U*D or L*D)
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides (vectors x).
 * @param[in]     A      The factored matrix from SSYTRF.
 *                        Double precision array, dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv   Pivot indices from SSYTRF. Integer array, dimension (n).
 *                        0-based indexing.
 * @param[in,out] B      On entry, contains NRHS vectors of length N.
 *                        On exit, overwritten with A * B.
 *                        Double precision array, dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info   = 0: successful exit
 */
void slavsy(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const f32* const restrict A,
    const int lda,
    const int* const restrict ipiv,
    f32* const restrict B,
    const int ldb,
    int* info)
{
    const f32 ONE = 1.0f;

    int nounit;
    int j, k, kp;
    f32 d11, d12, d21, d22, t1, t2;

    /* Test the input parameters. */
    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'T' || trans[0] == 't') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!(diag[0] == 'U' || diag[0] == 'u') && !(diag[0] == 'N' || diag[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("SLAVSY", -(*info));
        return;
    }

    /* Quick return if possible. */
    if (n == 0) return;

    nounit = (diag[0] == 'N' || diag[0] == 'n');

    /*------------------------------------------
     * Compute  B := A * B  (No transpose)
     *------------------------------------------*/
    if (trans[0] == 'N' || trans[0] == 'n') {

        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Compute B := U*B (or U*D*B)
             * where U = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
             * Loop forward applying the transformations. */
            k = 0;
            while (k < n) {
                if (ipiv[k] >= 0) {
                    /* 1x1 pivot block.
                     * Multiply by the diagonal element if forming U * D. */
                    if (nounit) {
                        cblas_sscal(nrhs, A[k + k * lda], &B[k], ldb);
                    }

                    /* Multiply by P(K) * inv(U(K)) if K > 0. */
                    if (k > 0) {
                        /* Apply the transformation. */
                        cblas_sger(CblasColMajor, k, nrhs, ONE,
                                   &A[0 + k * lda], 1, &B[k], ldb, &B[0], ldb);

                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }
                    }
                    k++;
                } else {
                    /* 2x2 pivot block.
                     * Multiply by the diagonal block if forming U * D. */
                    if (nounit) {
                        d11 = A[k + k * lda];
                        d22 = A[(k + 1) + (k + 1) * lda];
                        d12 = A[k + (k + 1) * lda];
                        d21 = d12;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    /* Multiply by P(K) * inv(U(K)) if K > 0. */
                    if (k > 0) {
                        /* Apply the transformations. */
                        cblas_sger(CblasColMajor, k, nrhs, ONE,
                                   &A[0 + k * lda], 1, &B[k], ldb, &B[0], ldb);
                        cblas_sger(CblasColMajor, k, nrhs, ONE,
                                   &A[0 + (k + 1) * lda], 1, &B[k + 1], ldb,
                                   &B[0], ldb);

                        /* Interchange if P(K) != I.
                         * For 2x2 block: kp = abs(ipiv[k]) (Fortran 1-based)
                         * 0-based: kp = -(ipiv[k]+1) */
                        kp = -(ipiv[k] + 1);
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }
                    }
                    k += 2;
                }
            }

        } else {
            /* Compute B := L*B (or L*D*B)
             * where L = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
             * Loop backward applying the transformations to B. */
            k = n - 1;
            while (k >= 0) {
                if (ipiv[k] >= 0) {
                    /* 1x1 pivot block.
                     * Multiply by the diagonal element if forming L * D. */
                    if (nounit) {
                        cblas_sscal(nrhs, A[k + k * lda], &B[k], ldb);
                    }

                    /* Multiply by P(K) * inv(L(K)) if K < N-1. */
                    if (k < n - 1) {
                        kp = ipiv[k];

                        /* Apply the transformation. */
                        cblas_sger(CblasColMajor, n - k - 1, nrhs, ONE,
                                   &A[(k + 1) + k * lda], 1, &B[k], ldb,
                                   &B[k + 1], ldb);

                        /* Interchange if a permutation was applied. */
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }
                    }
                    k--;
                } else {
                    /* 2x2 pivot block.
                     * Multiply by the diagonal block if forming L * D. */
                    if (nounit) {
                        d11 = A[(k - 1) + (k - 1) * lda];
                        d22 = A[k + k * lda];
                        d21 = A[k + (k - 1) * lda];
                        d12 = d21;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    /* Multiply by P(K) * inv(L(K)) if K < N-1. */
                    if (k < n - 1) {
                        /* Apply the transformation. */
                        cblas_sger(CblasColMajor, n - k - 1, nrhs, ONE,
                                   &A[(k + 1) + k * lda], 1, &B[k], ldb,
                                   &B[k + 1], ldb);
                        cblas_sger(CblasColMajor, n - k - 1, nrhs, ONE,
                                   &A[(k + 1) + (k - 1) * lda], 1, &B[k - 1], ldb,
                                   &B[k + 1], ldb);

                        /* Interchange if a permutation was applied.
                         * kp = abs(ipiv[k]) (Fortran) -> -(ipiv[k]+1) (0-based) */
                        kp = -(ipiv[k] + 1);
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }
                    }
                    k -= 2;
                }
            }
        }

    /*----------------------------------------
     * Compute  B := A' * B  (transpose)
     *----------------------------------------*/
    } else {

        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Form B := U'*B (or D*U'*B)
             * where U  = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
             * and   U' = inv(U'(1))*P(1)* ... *inv(U'(m))*P(m)
             * Loop backward applying the transformations. */
            k = n - 1;
            while (k >= 0) {
                if (ipiv[k] >= 0) {
                    /* 1x1 pivot block. */
                    if (k > 0) {
                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }

                        /* Apply the transformation.
                         * Fortran: DGEMV('T', K-1, NRHS, 1, B, LDB, A(1,K), 1, 1, B(K,1), LDB)
                         * 0-based: M=k, source=B(0:k-1,:), vector=A(0:k-1,k) */
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    k, nrhs, ONE, &B[0], ldb,
                                    &A[0 + k * lda], 1, ONE, &B[k], ldb);
                    }
                    if (nounit) {
                        cblas_sscal(nrhs, A[k + k * lda], &B[k], ldb);
                    }
                    k--;
                } else {
                    /* 2x2 pivot block. */
                    if (k > 1) {
                        /* Interchange if P(K) != I.
                         * For 2x2 upper: kp = abs(ipiv[k]) -> -(ipiv[k]+1)
                         * Swap row k-1 (Fortran K-1) with kp. */
                        kp = -(ipiv[k] + 1);
                        if (kp != k - 1) {
                            cblas_sswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                        }

                        /* Apply the transformations.
                         * Fortran: DGEMV('T', K-2, ..., A(1,K), ...)
                         *          DGEMV('T', K-2, ..., A(1,K-1), ...)
                         * 0-based: M=k-1 */
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    k - 1, nrhs, ONE, &B[0], ldb,
                                    &A[0 + k * lda], 1, ONE, &B[k], ldb);
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    k - 1, nrhs, ONE, &B[0], ldb,
                                    &A[0 + (k - 1) * lda], 1, ONE, &B[k - 1], ldb);
                    }

                    /* Multiply by the diagonal block if non-unit. */
                    if (nounit) {
                        d11 = A[(k - 1) + (k - 1) * lda];
                        d22 = A[k + k * lda];
                        d12 = A[(k - 1) + k * lda];
                        d21 = d12;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    k -= 2;
                }
            }

        } else {
            /* Form B := L'*B (or D*L'*B)
             * where L  = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
             * and   L' = inv(L'(m))*P(m)* ... *inv(L'(1))*P(1)
             * Loop forward applying the L-transformations. */
            k = 0;
            while (k < n) {
                if (ipiv[k] >= 0) {
                    /* 1x1 pivot block. */
                    if (k < n - 1) {
                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k) {
                            cblas_sswap(nrhs, &B[k], ldb, &B[kp], ldb);
                        }

                        /* Apply the transformation.
                         * Fortran: DGEMV('T', N-K, NRHS, 1, B(K+1,1), LDB, A(K+1,K), 1, 1, B(K,1), LDB)
                         * 0-based: M=n-k-1, source=B(k+1:n-1,:), vector=A(k+1:n-1,k) */
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    n - k - 1, nrhs, ONE, &B[k + 1], ldb,
                                    &A[(k + 1) + k * lda], 1, ONE, &B[k], ldb);
                    }
                    if (nounit) {
                        cblas_sscal(nrhs, A[k + k * lda], &B[k], ldb);
                    }
                    k++;
                } else {
                    /* 2x2 pivot block. */
                    if (k < n - 2) {
                        /* Interchange if P(K) != I.
                         * For 2x2 lower: kp = abs(ipiv[k]) -> -(ipiv[k]+1)
                         * Swap row k+1 with kp. */
                        kp = -(ipiv[k] + 1);
                        if (kp != k + 1) {
                            cblas_sswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                        }

                        /* Apply the transformation.
                         * Fortran: DGEMV('T', N-K-1, ..., A(K+2,K+1), ...)
                         *          DGEMV('T', N-K-1, ..., A(K+2,K), ...)
                         * 0-based: M=n-k-2, source=B(k+2:n-1,:) */
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    n - k - 2, nrhs, ONE, &B[k + 2], ldb,
                                    &A[(k + 2) + (k + 1) * lda], 1, ONE,
                                    &B[k + 1], ldb);
                        cblas_sgemv(CblasColMajor, CblasTrans,
                                    n - k - 2, nrhs, ONE, &B[k + 2], ldb,
                                    &A[(k + 2) + k * lda], 1, ONE,
                                    &B[k], ldb);
                    }

                    /* Multiply by the diagonal block if non-unit. */
                    if (nounit) {
                        d11 = A[k + k * lda];
                        d22 = A[(k + 1) + (k + 1) * lda];
                        d21 = A[(k + 1) + k * lda];
                        d12 = d21;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    k += 2;
                }
            }
        }
    }
}
