/**
 * @file sgelqt3.c
 * @brief SGELQT3 recursively computes a LQ factorization of a general
 *        real matrix using the compact WY representation of Q.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGELQT3 recursively computes a LQ factorization of a real M-by-N
 * matrix A, using the compact WY representation of Q.
 *
 * Based on the algorithm of Elmroth and Gustavson,
 * IBM J. Res. Develop. Vol 44 No. 4 July 2000.
 *
 * The matrix V stores the elementary reflectors H(i) in the i-th row
 * above the diagonal. For example, if M=5 and N=3, the matrix V is
 *
 *              V = (  1  v1 v1 v1 v1 )
 *                  (     1  v2 v2 v2 )
 *                  (     1  v3 v3 v3 )
 *
 * where the vi's represent the vectors which define H(i), which are returned
 * in the matrix A. The 1's along the diagonal of V are not stored in A. The
 * block reflector H is then given by
 *
 *              H = I - V * T * V**T
 *
 * where V**T is the transpose of V.
 *
 * @param[in]     m     The number of rows of the matrix A. m <= n.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the real M-by-N matrix A. On exit, the elements
 *                      on and below the diagonal contain the M-by-M lower
 *                      triangular matrix L; the elements above the diagonal are
 *                      the rows of V.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Double precision array, dimension (ldt, m).
 *                      The M-by-M upper triangular factor of the block reflector.
 *                      The elements on and above the diagonal contain the block
 *                      reflector T; the elements below the diagonal are not used.
 * @param[in]     ldt   The leading dimension of the array T. ldt >= max(1, m).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgelqt3(const int m, const int n,
             float* const restrict A, const int lda,
             float* const restrict T, const int ldt,
             int* info)
{
    int i, j, m1, m2, i1, j1, iinfo;

    /* Parameter validation */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (ldt < (m > 1 ? m : 1)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SGELQT3", -(*info));
        return;
    }

    if (m == 1) {

        /* Compute Householder transform when M=1 */
        slarfg(n, &A[0], &A[0 + (1 < n ? 1 : 0) * lda], lda, &T[0]);

    } else {

        /* Otherwise, split A into blocks */
        m1 = m / 2;
        m2 = m - m1;
        i1 = m1;       /* 0-based: Fortran I1 = MIN(M1+1, M) maps to m1 */
        j1 = m;        /* 0-based: Fortran J1 = MIN(M+1, N) maps to m */

        /* Compute A(0:m1-1, 0:n-1) <- (Y1, R1, T1), where Q1 = I - Y1 T1 Y1^T */
        sgelqt3(m1, n, A, lda, T, ldt, &iinfo);

        /*
         * Compute A(m1:m-1, 0:n-1) = Q1^T * A(m1:m-1, 0:n-1)
         * [workspace: T(m1:m-1, 0:m1-1)]
         */

        /* T(m1:m-1, 0:m1-1) = A(m1:m-1, 0:m1-1) */
        for (i = 0; i < m2; i++) {
            for (j = 0; j < m1; j++) {
                T[(i + m1) + j * ldt] = A[(i + m1) + j * lda];
            }
        }

        /* T(m1:m-1, 0:m1-1) = T(m1:m-1, 0:m1-1) * U1^T
         * where U1 is the upper triangular part of A(0:m1-1, 0:m1-1) with unit diag */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasUnit,
                    m2, m1, 1.0f, A, lda, &T[i1], ldt);

        /* T(m1:m-1, 0:m1-1) += A(m1:m-1, m1:n-1) * A(0:m1-1, m1:n-1)^T */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m2, m1, n - m1, 1.0f,
                    &A[i1 + i1 * lda], lda, &A[0 + i1 * lda], lda,
                    1.0f, &T[i1], ldt);

        /* T(m1:m-1, 0:m1-1) = T(m1:m-1, 0:m1-1) * T1
         * where T1 is the upper triangular T(0:m1-1, 0:m1-1) */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m2, m1, 1.0f, T, ldt, &T[i1], ldt);

        /* A(m1:m-1, m1:n-1) -= T(m1:m-1, 0:m1-1) * A(0:m1-1, m1:n-1) */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m2, n - m1, m1, -1.0f,
                    &T[i1], ldt, &A[0 + i1 * lda], lda,
                    1.0f, &A[i1 + i1 * lda], lda);

        /* A(m1:m-1, 0:m1-1) -= T(m1:m-1, 0:m1-1) * U1 */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                    m2, m1, 1.0f, A, lda, &T[i1], ldt);

        /* A(m1:m-1, 0:m1-1) -= T(m1:m-1, 0:m1-1) and zero T(m1:m-1, 0:m1-1) */
        for (i = 0; i < m2; i++) {
            for (j = 0; j < m1; j++) {
                A[(i + m1) + j * lda] = A[(i + m1) + j * lda] - T[(i + m1) + j * ldt];
                T[(i + m1) + j * ldt] = 0.0f;
            }
        }

        /* Compute A(m1:m-1, m1:n-1) <- (Y2, R2, T2) where Q2 = I - Y2 T2 Y2^T */
        sgelqt3(m2, n - m1, &A[i1 + i1 * lda], lda,
                &T[i1 + i1 * ldt], ldt, &iinfo);

        /*
         * Compute T3 = T(0:m1-1, m1:m-1) = -T1 * Y1^T * Y2 * T2
         */

        /* T(0:m1-1, m1:m-1) = A(0:m1-1, m1:m-1)  (copy) */
        for (i = 0; i < m2; i++) {
            for (j = 0; j < m1; j++) {
                T[j + (i + m1) * ldt] = A[j + (i + m1) * lda];
            }
        }

        /* T(0:m1-1, m1:m-1) = T(0:m1-1, m1:m-1) * U2^T
         * where U2 is upper triangular part of A(m1:m-1, m1:m-1) with unit diag */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasUnit,
                    m1, m2, 1.0f, &A[i1 + i1 * lda], lda, &T[0 + i1 * ldt], ldt);

        /* T(0:m1-1, m1:m-1) += A(0:m1-1, m:n-1) * A(m1:m-1, m:n-1)^T */
        if (n > m) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        m1, m2, n - m, 1.0f,
                        &A[0 + j1 * lda], lda, &A[i1 + j1 * lda], lda,
                        1.0f, &T[0 + i1 * ldt], ldt);
        }

        /* T(0:m1-1, m1:m-1) = -T1 * T(0:m1-1, m1:m-1) */
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m1, m2, -1.0f, T, ldt, &T[0 + i1 * ldt], ldt);

        /* T(0:m1-1, m1:m-1) = T(0:m1-1, m1:m-1) * T2 */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m1, m2, 1.0f, &T[i1 + i1 * ldt], ldt, &T[0 + i1 * ldt], ldt);

        /*
         * Result: Y = (Y1, Y2); L = [ L1            0  ]; T = [ T1  T3 ]
         *                            [ A(0:m1-1, m1:n-1)  L2 ]      [  0  T2 ]
         */
    }
}
