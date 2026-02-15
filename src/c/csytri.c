/**
 * @file csytri.c
 * @brief CSYTRI computes the inverse of a complex symmetric indefinite matrix
 *        using the factorization computed by CSYTRF.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CSYTRI computes the inverse of a complex symmetric indefinite matrix
 * A using the factorization A = U*D*U**T or A = L*D*L**T computed by
 * CSYTRF.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**T
 *                        = 'L': Lower triangular, A = L*D*L**T
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the factored matrix from CSYTRF.
 *                      On exit, the symmetric inverse of the original matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from CSYTRF.
 * @param[out]    work  Single complex array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void csytri(
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    const int* restrict ipiv,
    c64* restrict work,
    int* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CSYTRI", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        for (int k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    } else {
        for (int k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {
        /* Compute inv(A) from the factorization A = U*D*U**T.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        int k = 0;
        while (k < n) {
            int kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = ONE / A[k + k * lda];

                /* Compute column k of the inverse. */
                if (k > 0) {
                    c64 dotval;
                    cblas_ccopy(k, &A[0 + k * lda], 1, work, 1);
                    csymv(uplo, k, NEG_ONE, A, lda, work, 1,
                          ZERO, &A[0 + k * lda], 1);
                    cblas_cdotu_sub(k, work, 1, &A[0 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block. */
                c64 t = A[k + (k + 1) * lda];
                c64 ak = A[k + k * lda] / t;
                c64 akp1 = A[(k + 1) + (k + 1) * lda] / t;
                c64 akkp1 = A[k + (k + 1) * lda] / t;
                c64 d = t * (ak * akp1 - ONE);
                A[k + k * lda] = akp1 / d;
                A[(k + 1) + (k + 1) * lda] = ak / d;
                A[k + (k + 1) * lda] = -akkp1 / d;

                /* Compute columns k and k+1 of the inverse. */
                if (k > 0) {
                    c64 dotval;
                    cblas_ccopy(k, &A[0 + k * lda], 1, work, 1);
                    csymv(uplo, k, NEG_ONE, A, lda, work, 1,
                          ZERO, &A[0 + k * lda], 1);
                    cblas_cdotu_sub(k, work, 1, &A[0 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                    cblas_cdotu_sub(k, &A[0 + k * lda], 1,
                                   &A[0 + (k + 1) * lda], 1, &dotval);
                    A[k + (k + 1) * lda] -= dotval;
                    cblas_ccopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    csymv(uplo, k, NEG_ONE, A, lda, work, 1,
                          ZERO, &A[0 + (k + 1) * lda], 1);
                    cblas_cdotu_sub(k, work, 1,
                                   &A[0 + (k + 1) * lda], 1, &dotval);
                    A[(k + 1) + (k + 1) * lda] -= dotval;
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp. */
            int kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                if (kp > 0) {
                    cblas_cswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                }

                if (k - kp - 1 > 0) {
                    cblas_cswap(k - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[kp + (kp + 1) * lda], lda);
                }

                /* Swap diagonal elements */
                c64 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

                /* For 2x2 block, also swap A(k, k+1) with A(kp, k+1) */
                if (kstep == 2) {
                    temp = A[k + (k + 1) * lda];
                    A[k + (k + 1) * lda] = A[kp + (k + 1) * lda];
                    A[kp + (k + 1) * lda] = temp;
                }
            }

            k += kstep;
        }

    } else {
        /* Compute inv(A) from the factorization A = L*D*L**T.
         *
         * K decreases from n-1 to 0 in steps of 1 or 2. */
        int k = n - 1;
        while (k >= 0) {
            int kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = ONE / A[k + k * lda];

                /* Compute column k of the inverse. */
                if (k < n - 1) {
                    c64 dotval;
                    cblas_ccopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    csymv(uplo, n - k - 1, NEG_ONE, &A[(k + 1) + (k + 1) * lda], lda,
                          work, 1, ZERO, &A[(k + 1) + k * lda], 1);
                    cblas_cdotu_sub(n - k - 1, work, 1,
                                   &A[(k + 1) + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block. */
                c64 t = A[k + (k - 1) * lda];
                c64 ak = A[(k - 1) + (k - 1) * lda] / t;
                c64 akp1 = A[k + k * lda] / t;
                c64 akkp1 = A[k + (k - 1) * lda] / t;
                c64 d = t * (ak * akp1 - ONE);
                A[(k - 1) + (k - 1) * lda] = akp1 / d;
                A[k + k * lda] = ak / d;
                A[k + (k - 1) * lda] = -akkp1 / d;

                /* Compute columns k-1 and k of the inverse. */
                if (k < n - 1) {
                    c64 dotval;
                    cblas_ccopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    csymv(uplo, n - k - 1, NEG_ONE, &A[(k + 1) + (k + 1) * lda], lda,
                          work, 1, ZERO, &A[(k + 1) + k * lda], 1);
                    cblas_cdotu_sub(n - k - 1, work, 1,
                                   &A[(k + 1) + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                    cblas_cdotu_sub(n - k - 1,
                                   &A[(k + 1) + k * lda], 1,
                                   &A[(k + 1) + (k - 1) * lda], 1, &dotval);
                    A[k + (k - 1) * lda] -= dotval;
                    cblas_ccopy(n - k - 1, &A[(k + 1) + (k - 1) * lda], 1, work, 1);
                    csymv(uplo, n - k - 1, NEG_ONE, &A[(k + 1) + (k + 1) * lda], lda,
                          work, 1, ZERO, &A[(k + 1) + (k - 1) * lda], 1);
                    cblas_cdotu_sub(n - k - 1, work, 1,
                                   &A[(k + 1) + (k - 1) * lda], 1, &dotval);
                    A[(k - 1) + (k - 1) * lda] -= dotval;
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp. */
            int kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                if (kp < n - 1) {
                    cblas_cswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[(kp + 1) + kp * lda], 1);
                }

                if (kp - k - 1 > 0) {
                    cblas_cswap(kp - k - 1, &A[(k + 1) + k * lda], 1,
                                &A[kp + (k + 1) * lda], lda);
                }

                /* Swap diagonal elements */
                c64 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

                /* For 2x2 block, also swap A(k, k-1) with A(kp, k-1) */
                if (kstep == 2) {
                    temp = A[k + (k - 1) * lda];
                    A[k + (k - 1) * lda] = A[kp + (k - 1) * lda];
                    A[kp + (k - 1) * lda] = temp;
                }
            }

            k -= kstep;
        }
    }
}
