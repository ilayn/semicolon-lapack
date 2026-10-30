/**
 * @file dsytri.c
 * @brief DSYTRI computes the inverse of a real symmetric indefinite matrix
 *        using the factorization computed by DSYTRF.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSYTRI computes the inverse of a real symmetric indefinite matrix
 * A using the factorization A = U*D*U**T or A = L*D*L**T computed by
 * DSYTRF.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular, form is A = U*D*U**T
 *                      - `'L'`: Lower triangular, form is A = L*D*L**T
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, the block diagonal matrix D and the
 *                      multipliers used to obtain the factor U or L as
 *                      computed by `dsytrf`.
 *                      On exit, if `info=0`, the (symmetric) inverse of the
 *                      original matrix. If `uplo='U'`, the upper triangular
 *                      part of the inverse is formed and the part of A
 *                      below the diagonal is not referenced; if `uplo='L'`
 *                      the lower triangular part of the inverse is formed
 *                      and the part of A above the diagonal is not
 *                      referenced.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      Details of the interchanges and the block structure
 *                      of D as determined by `dsytrf`.
 * @param[out]    work  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, `D(i,i)=0`; the matrix is singular
 *                           and its inverse could not be computed.
 */
void dsytri(
    const char* uplo,
    const INT n,
    f64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    f64* restrict work,
    INT* info)
{
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DSYTRI", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        /* Upper triangular storage: examine D from bottom to top */
        for (INT k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == 0.0) {
                *info = k + 1;
                return;
            }
        }
    } else {
        /* Lower triangular storage: examine D from top to bottom. */
        for (INT k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == 0.0) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {
        /* Compute inv(A) from the factorization A = U*D*U**T.
         *
         * K is the main loop index, increasing from 0 to n-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks. */
        INT k = 0;
        while (k < n) {
            INT kstep;
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Invert the diagonal block. */
                A[k + k * lda] = 1.0 / A[k + k * lda];

                /* Compute column k of the inverse. */
                if (k > 0) {
                    cblas_dcopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasUpper,
                                k, -1.0, A, lda, work, 1,
                                0.0, &A[0 + k * lda], 1);
                    A[k + k * lda] -= cblas_ddot(k, work, 1, &A[0 + k * lda], 1);
                }
                kstep = 1;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Invert the diagonal block. */
                f64 t = fabs(A[k + (k + 1) * lda]);
                f64 ak = A[k + k * lda] / t;
                f64 akp1 = A[(k + 1) + (k + 1) * lda] / t;
                f64 akkp1 = A[k + (k + 1) * lda] / t;
                f64 d = t * (ak * akp1 - 1.0);
                A[k + k * lda] = akp1 / d;
                A[(k + 1) + (k + 1) * lda] = ak / d;
                A[k + (k + 1) * lda] = -akkp1 / d;

                /* Compute columns k and k+1 of the inverse. */
                if (k > 0) {
                    cblas_dcopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasUpper,
                                k, -1.0, A, lda, work, 1,
                                0.0, &A[0 + k * lda], 1);
                    A[k + k * lda] -= cblas_ddot(k, work, 1, &A[0 + k * lda], 1);
                    A[k + (k + 1) * lda] -= cblas_ddot(k, &A[0 + k * lda], 1,
                                                        &A[0 + (k + 1) * lda], 1);
                    cblas_dcopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasUpper,
                                k, -1.0, A, lda, work, 1,
                                0.0, &A[0 + (k + 1) * lda], 1);
                    A[(k + 1) + (k + 1) * lda] -= cblas_ddot(k, work, 1,
                                                              &A[0 + (k + 1) * lda], 1);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp in the leading
             * submatrix A(0:k+1,0:k+1) */
            INT kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                if (kp > 0) {
                    cblas_dswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                }

                if (k - kp - 1 > 0) {
                    cblas_dswap(k - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[kp + (kp + 1) * lda], lda);
                }

                f64 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

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
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks. */
        INT k = n - 1;
        while (k >= 0) {
            INT kstep;
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Invert the diagonal block. */
                A[k + k * lda] = 1.0 / A[k + k * lda];

                /* Compute column k of the inverse. */
                if (k < n - 1) {
                    cblas_dcopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0, &A[(k + 1) + k * lda], 1);
                    A[k + k * lda] -= cblas_ddot(n - k - 1, work, 1,
                                                  &A[(k + 1) + k * lda], 1);
                }
                kstep = 1;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Invert the diagonal block. */
                f64 t = fabs(A[k + (k - 1) * lda]);
                f64 ak = A[(k - 1) + (k - 1) * lda] / t;
                f64 akp1 = A[k + k * lda] / t;
                f64 akkp1 = A[k + (k - 1) * lda] / t;
                f64 d = t * (ak * akp1 - 1.0);
                A[(k - 1) + (k - 1) * lda] = akp1 / d;
                A[k + k * lda] = ak / d;
                A[k + (k - 1) * lda] = -akkp1 / d;

                /* Compute columns k-1 and k of the inverse. */
                if (k < n - 1) {
                    cblas_dcopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0, &A[(k + 1) + k * lda], 1);
                    A[k + k * lda] -= cblas_ddot(n - k - 1, work, 1,
                                                  &A[(k + 1) + k * lda], 1);
                    A[k + (k - 1) * lda] -= cblas_ddot(n - k - 1,
                                                        &A[(k + 1) + k * lda], 1,
                                                        &A[(k + 1) + (k - 1) * lda], 1);
                    cblas_dcopy(n - k - 1, &A[(k + 1) + (k - 1) * lda], 1, work, 1);
                    cblas_dsymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0, &A[(k + 1) + (k - 1) * lda], 1);
                    A[(k - 1) + (k - 1) * lda] -= cblas_ddot(n - k - 1, work, 1,
                                                              &A[(k + 1) + (k - 1) * lda], 1);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp in the trailing
             * submatrix A(k-1:n-1,k-1:n-1) */
            INT kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                if (kp < n - 1) {
                    cblas_dswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[(kp + 1) + kp * lda], 1);
                }

                if (kp - k - 1 > 0) {
                    cblas_dswap(kp - k - 1, &A[(k + 1) + k * lda], 1,
                                &A[kp + (k + 1) * lda], lda);
                }

                f64 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

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
