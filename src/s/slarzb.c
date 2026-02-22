/**
 * @file slarzb.c
 * @brief SLARZB applies a block reflector or its transpose to a general
 *        distributed matrix.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLARZB applies a real block reflector H or its transpose H**T to a
 * real distributed M-by-N matrix C from either the left or the right.
 *
 * Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
 *
 * @param[in]     side    'L': apply H or H**T from the Left;
 *                        'R': apply H or H**T from the Right.
 * @param[in]     trans   'N': apply H (No transpose);
 *                        'C': apply H**T (Transpose).
 * @param[in]     direct  Indicates how H is formed from a product of
 *                        elementary reflectors.
 *                        'F': H = H(1) H(2) ... H(k) (Forward, not supported);
 *                        'B': H = H(k) ... H(2) H(1) (Backward).
 * @param[in]     storev  Indicates how the vectors which define the elementary
 *                        reflectors are stored:
 *                        'C': Columnwise (not supported);
 *                        'R': Rowwise.
 * @param[in]     m       Number of rows of the matrix C. m >= 0.
 * @param[in]     n       Number of columns of the matrix C. n >= 0.
 * @param[in]     k       The order of the matrix T (= the number of elementary
 *                        reflectors whose product defines the block reflector).
 * @param[in]     l       The number of columns of the matrix V containing the
 *                        meaningful part of the Householder reflectors.
 *                        If side = "L", m >= l >= 0; if side = "R", n >= l >= 0.
 * @param[in]     V       Double precision array, dimension (ldv, nv).
 *                        If storev = "C", nv = k; if storev = "R", nv = l.
 * @param[in]     ldv     Leading dimension of V.
 *                        If storev = "C", ldv >= l; if storev = "R", ldv >= k.
 * @param[in]     T       Double precision array, dimension (ldt, k).
 *                        The triangular k-by-k matrix T in the representation
 *                        of the block reflector.
 * @param[in]     ldt     Leading dimension of T. ldt >= k.
 * @param[in,out] C       Double precision array, dimension (ldc, n).
 *                        On entry, the m-by-n matrix C.
 *                        On exit, C is overwritten by H*C or H**T*C or C*H
 *                        or C*H**T.
 * @param[in]     ldc     Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work    Double precision array, dimension (ldwork, k).
 * @param[in]     ldwork  Leading dimension of work.
 *                        If side = "L", ldwork >= max(1, n);
 *                        if side = "R", ldwork >= max(1, m).
 */
void slarzb(const char* side, const char* trans, const char* direct,
            const char* storev, const INT m, const INT n,
            const INT k, const INT l,
            const f32* restrict V, const INT ldv,
            const f32* restrict T, const INT ldt,
            f32* restrict C, const INT ldc,
            f32* restrict work, const INT ldwork)
{
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;
    INT i, j;
    CBLAS_TRANSPOSE transt;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) return;


    /* Check for currently supported options */
    if (direct[0] != 'B' && direct[0] != 'b') return;
    if (storev[0] != 'R' && storev[0] != 'r') return;

    /* Determine the transpose to apply to T */
    if (trans[0] == 'N' || trans[0] == 'n') {
        transt = CblasTrans;
    } else {
        transt = CblasNoTrans;
    }

    if (side[0] == 'L' || side[0] == 'l') {
        /*
         * Form  H * C  or  H**T * C
         *
         * W(1:n, 1:k) = C(1:k, 1:n)**T
         */
        for (j = 0; j < k; j++) {
            cblas_scopy(n, &C[j + 0 * ldc], ldc, &work[0 + j * ldwork], 1);
        }

        /*
         * W(1:n, 1:k) = W(1:n, 1:k) +
         *               C(m-l+1:m, 1:n)**T * V(1:k, 1:l)**T
         */
        if (l > 0) {
            cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans,
                        n, k, l, ONE,
                        &C[(m - l) + 0 * ldc], ldc, V, ldv,
                        ONE, work, ldwork);
        }

        /*
         * W(1:n, 1:k) = W(1:n, 1:k) * T**T  or  W(1:n, 1:k) * T
         */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                    transt, CblasNonUnit,
                    n, k, ONE, T, ldt, work, ldwork);

        /*
         * C(1:k, 1:n) = C(1:k, 1:n) - W(1:n, 1:k)**T
         */
        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                C[i + j * ldc] -= work[j + i * ldwork];
            }
        }

        /*
         * C(m-l+1:m, 1:n) = C(m-l+1:m, 1:n) -
         *                    V(1:k, 1:l)**T * W(1:n, 1:k)**T
         */
        if (l > 0) {
            cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans,
                        l, n, k, NEG_ONE, V, ldv,
                        work, ldwork, ONE, &C[(m - l) + 0 * ldc], ldc);
        }

    } else if (side[0] == 'R' || side[0] == 'r') {
        /*
         * Form  C * H  or  C * H**T
         *
         * W(1:m, 1:k) = C(1:m, 1:k)
         */
        for (j = 0; j < k; j++) {
            cblas_scopy(m, &C[0 + j * ldc], 1, &work[0 + j * ldwork], 1);
        }

        /*
         * W(1:m, 1:k) = W(1:m, 1:k) +
         *               C(1:m, n-l+1:n) * V(1:k, 1:l)**T
         */
        if (l > 0) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        m, k, l, ONE,
                        &C[0 + (n - l) * ldc], ldc, V, ldv,
                        ONE, work, ldwork);
        }

        /*
         * W(1:m, 1:k) = W(1:m, 1:k) * T  or  W(1:m, 1:k) * T**T
         */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                    (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasTrans),
                    CblasNonUnit,
                    m, k, ONE, T, ldt, work, ldwork);

        /*
         * C(1:m, 1:k) = C(1:m, 1:k) - W(1:m, 1:k)
         */
        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] -= work[i + j * ldwork];
            }
        }

        /*
         * C(1:m, n-l+1:n) = C(1:m, n-l+1:n) -
         *                    W(1:m, 1:k) * V(1:k, 1:l)
         */
        if (l > 0) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, l, k, NEG_ONE,
                        work, ldwork, V, ldv,
                        ONE, &C[0 + (n - l) * ldc], ldc);
        }
    }
}
