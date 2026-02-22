/**
 * @file cgeqrt3.c
 * @brief CGEQRT3 recursively computes a QR factorization of a general
 *        complex matrix using the compact WY representation of Q.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGEQRT3 recursively computes a QR factorization of a complex M-by-N
 * matrix A, using the compact WY representation of Q.
 *
 * Based on the algorithm of Elmroth and Gustavson,
 * IBM J. Res. Develop. Vol 44 No. 4 July 2000.
 *
 * The matrix V stores the elementary reflectors H(i) in the i-th column
 * below the diagonal. For example, if M=5 and N=3, the matrix V is
 *
 *              V = (  1       )
 *                  ( v1  1    )
 *                  ( v1 v2  1 )
 *                  ( v1 v2 v3 )
 *                  ( v1 v2 v3 )
 *
 * where the vi's represent the vectors which define H(i), which are returned
 * in the matrix A. The 1's along the diagonal of V are not stored in A. The
 * block reflector H is then given by
 *
 *              H = I - V * T * V**H
 *
 * where V**H is the conjugate transpose of V.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= n.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the complex M-by-N matrix A. On exit, the elements
 *                      on and above the diagonal contain the N-by-N upper
 *                      triangular matrix R; the elements below the diagonal are
 *                      the columns of V.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Single complex array, dimension (ldt, n).
 *                      The N-by-N upper triangular factor of the block reflector.
 *                      The elements on and above the diagonal contain the block
 *                      reflector T; the elements below the diagonal are not used.
 * @param[in]     ldt   The leading dimension of the array T. ldt >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgeqrt3(const INT m, const INT n,
             c64* restrict A, const INT lda,
             c64* restrict T, const INT ldt,
             INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);
    INT i, j, n1, n2, iinfo;

    /* Parameter validation */
    *info = 0;
    if (n < 0) {
        *info = -2;
    } else if (m < n) {
        *info = -1;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CGEQRT3", -(*info));
        return;
    }

    if (n == 1) {

        /* Compute Householder transform when N=1 */
        clarfg(m, &A[0], &A[(1 < m ? 1 : 0)], 1, &T[0]);

    } else {

        /* Otherwise, split A into blocks */
        n1 = n / 2;
        n2 = n - n1;

        /* Compute A(0:m-1, 0:n1-1) <- (Y1, R1, T1), where Q1 = I - Y1 T1 Y1^H */
        cgeqrt3(m, n1, A, lda, T, ldt, &iinfo);

        /*
         * Compute A(0:m-1, n1:n-1) = Q1^H * A(0:m-1, n1:n-1)
         * [workspace: T(0:n1-1, n1:n-1)]
         */

        /* T(0:n1-1, n1:n-1) = A(0:n1-1, n1:n-1) */
        for (j = 0; j < n2; j++) {
            for (i = 0; i < n1; i++) {
                T[i + (j + n1) * ldt] = A[i + (j + n1) * lda];
            }
        }

        /* T(0:n1-1, n1:n-1) = L1^H * T(0:n1-1, n1:n-1)
         * where L1 is the lower triangular part of A(0:n1-1, 0:n1-1) with unit diag */
        cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasUnit,
                    n1, n2, &ONE, A, lda, &T[n1 * ldt], ldt);

        /* T(0:n1-1, n1:n-1) += A(n1:m-1, 0:n1-1)^H * A(n1:m-1, n1:n-1) */
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    n1, n2, m - n1, &ONE,
                    &A[n1], lda, &A[n1 + n1 * lda], lda,
                    &ONE, &T[n1 * ldt], ldt);

        /* T(0:n1-1, n1:n-1) = T1^H * T(0:n1-1, n1:n-1)
         * where T1 is the upper triangular T(0:n1-1, 0:n1-1) */
        cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit,
                    n1, n2, &ONE, T, ldt, &T[n1 * ldt], ldt);

        /* A(n1:m-1, n1:n-1) -= A(n1:m-1, 0:n1-1) * T(0:n1-1, n1:n-1) */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - n1, n2, n1, &NEG_ONE,
                    &A[n1], lda, &T[n1 * ldt], ldt,
                    &ONE, &A[n1 + n1 * lda], lda);

        /* A(0:n1-1, n1:n-1) -= L1 * T(0:n1-1, n1:n-1) */
        cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n1, n2, &ONE, A, lda, &T[n1 * ldt], ldt);

        /* A(0:n1-1, n1:n-1) -= T(0:n1-1, n1:n-1) */
        for (j = 0; j < n2; j++) {
            for (i = 0; i < n1; i++) {
                A[i + (j + n1) * lda] -= T[i + (j + n1) * ldt];
            }
        }

        /* Compute A(n1:m-1, n1:n-1) <- (Y2, R2, T2) where Q2 = I - Y2 T2 Y2^H */
        cgeqrt3(m - n1, n2, &A[n1 + n1 * lda], lda,
                &T[n1 + n1 * ldt], ldt, &iinfo);

        /*
         * Compute T3 = T(0:n1-1, n1:n-1) = -T1 * Y1^H * Y2 * T2
         */

        /* T(0:n1-1, n1:n-1) = conj(A(n1:n-1, 0:n1-1))^T  (conjugated transposed copy) */
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                T[i + (j + n1) * ldt] = conjf(A[(j + n1) + i * lda]);
            }
        }

        /* T(0:n1-1, n1:n-1) = T(0:n1-1, n1:n-1) * L2
         * where L2 is the lower triangular part of A(n1:n1+n2-1, n1:n1+n2-1) with unit diag */
        cblas_ctrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                    n1, n2, &ONE, &A[n1 + n1 * lda], lda, &T[n1 * ldt], ldt);

        /* T(0:n1-1, n1:n-1) += A(n:m-1, 0:n1-1)^H * A(n:m-1, n1:n-1) */
        if (m > n) {
            cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        n1, n2, m - n, &ONE,
                        &A[n], lda, &A[n + n1 * lda], lda,
                        &ONE, &T[n1 * ldt], ldt);
        }

        /* T(0:n1-1, n1:n-1) = -T1 * T(0:n1-1, n1:n-1) */
        cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    n1, n2, &NEG_ONE, T, ldt, &T[n1 * ldt], ldt);

        /* T(0:n1-1, n1:n-1) = T(0:n1-1, n1:n-1) * T2 */
        cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    n1, n2, &ONE, &T[n1 + n1 * ldt], ldt, &T[n1 * ldt], ldt);

        /*
         * Result: Y = (Y1, Y2); R = [ R1  A(0:n1-1, n1:n-1) ]; T = [ T1  T3 ]
         *                            [  0       R2            ]      [  0  T2 ]
         */
    }
}
