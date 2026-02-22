/**
 * @file dgeqrt2.c
 * @brief DGEQRT2 computes a QR factorization of a general real matrix using
 *        the compact WY representation of Q.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DGEQRT2 computes a QR factorization of a real M-by-N matrix A,
 * using the compact WY representation of Q.
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
 *              H = I - V * T * V**T
 *
 * where V**T is the transpose of V.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= n.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the real M-by-N matrix A. On exit, the elements
 *                      on and above the diagonal contain the N-by-N upper
 *                      triangular matrix R; the elements below the diagonal are
 *                      the columns of V.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Double precision array, dimension (ldt, n).
 *                      The N-by-N upper triangular factor of the block reflector.
 *                      The elements on and above the diagonal contain the block
 *                      reflector T; the elements below the diagonal are not used.
 * @param[in]     ldt   The leading dimension of the array T. ldt >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgeqrt2(const INT m, const INT n,
             f64* restrict A, const INT lda,
             f64* restrict T, const INT ldt,
             INT* info)
{
    INT i, k;
    f64 aii, alpha;

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
        xerbla("DGEQRT2", -(*info));
        return;
    }

    k = m < n ? m : n;

    /*
     * First loop: generate reflectors and apply them.
     * tau(i) is stored temporarily in T(i, 0).
     */
    for (i = 0; i < k; i++) {
        /* Generate elementary reflector H(i) to annihilate A(i+1:m-1, i),
         * tau(i) -> T(i, 0) */
        dlarfg(m - i, &A[i + i * lda],
               &A[((i + 1) < m ? (i + 1) : i) + i * lda], 1,
               &T[i + 0 * ldt]);

        if (i < n - 1) {
            /* Apply H(i) to A(i:m-1, i+1:n-1) from the left */
            aii = A[i + i * lda];
            A[i + i * lda] = 1.0;

            /* W(0:n-i-2) := A(i:m-1, i+1:n-1)^T * A(i:m-1, i)
             * Store W in T(0:n-i-2, n-1) (last column of T as workspace) */
            cblas_dgemv(CblasColMajor, CblasTrans,
                        m - i, n - i - 1, 1.0,
                        &A[i + (i + 1) * lda], lda,
                        &A[i + i * lda], 1,
                        0.0, &T[0 + (n - 1) * ldt], 1);

            /* A(i:m-1, i+1:n-1) += alpha * A(i:m-1, i) * W^T */
            alpha = -(T[i + 0 * ldt]);
            cblas_dger(CblasColMajor,
                       m - i, n - i - 1, alpha,
                       &A[i + i * lda], 1,
                       &T[0 + (n - 1) * ldt], 1,
                       &A[i + (i + 1) * lda], lda);

            A[i + i * lda] = aii;
        }
    }

    /*
     * Second loop: build the T matrix from the stored tau values.
     * On entry, T(i, 0) holds tau(i) for i = 0..k-1.
     * On exit, T is upper triangular with T(i, i) = tau(i).
     */
    for (i = 1; i < n; i++) {
        aii = A[i + i * lda];
        A[i + i * lda] = 1.0;

        /* T(0:i-1, i) := -tau(i) * A(i:m-1, 0:i-1)^T * A(i:m-1, i) */
        alpha = -T[i + 0 * ldt];
        cblas_dgemv(CblasColMajor, CblasTrans,
                    m - i, i, alpha,
                    &A[i + 0 * lda], lda,
                    &A[i + i * lda], 1,
                    0.0, &T[0 + i * ldt], 1);

        A[i + i * lda] = aii;

        /* T(0:i-1, i) := T(0:i-1, 0:i-1) * T(0:i-1, i) */
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i, T, ldt, &T[0 + i * ldt], 1);

        /* T(i, i) = tau(i) */
        T[i + i * ldt] = T[i + 0 * ldt];
        T[i + 0 * ldt] = 0.0;
    }

    /* T(0, 0) = tau(0) is already in place from the first loop */
}
