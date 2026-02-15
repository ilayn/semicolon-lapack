/**
 * @file cunm22.c
 * @brief CUNM22 multiplies a general matrix by a banded unitary matrix.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CUNM22 overwrites the general complex M-by-N matrix C with
 *
 *                SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix of order NQ, with NQ = M if
 * SIDE = 'L' and NQ = N if SIDE = 'R'.
 * The unitary matrix Q processes a 2-by-2 block structure
 *
 *         [  Q11  Q12  ]
 *     Q = [            ]
 *         [  Q21  Q22  ],
 *
 * where Q12 is an N1-by-N1 lower triangular matrix and Q21 is an
 * N2-by-N2 upper triangular matrix.
 *
 * @param[in]     side    = 'L': apply Q or Q**H from the Left;
 *                         = 'R': apply Q or Q**H from the Right.
 * @param[in]     trans   = 'N': apply Q (No transpose);
 *                         = 'C': apply Q**H (Conjugate transpose).
 * @param[in]     m       The number of rows of the matrix C. m >= 0.
 * @param[in]     n       The number of columns of the matrix C. n >= 0.
 * @param[in]     n1      The dimension of Q12. n1 >= 0.
 * @param[in]     n2      The dimension of Q21. n2 >= 0.
 *                        n1 + n2 = m if SIDE = 'L' and n1 + n2 = n if SIDE = 'R'.
 * @param[in]     Q       Single complex array, dimension (ldq, m) if SIDE = 'L',
 *                        (ldq, n) if SIDE = 'R'.
 * @param[in]     ldq     The leading dimension of the array Q.
 *                        ldq >= max(1, m) if SIDE = 'L'; ldq >= max(1, n) if SIDE = 'R'.
 * @param[in,out] C       Single complex array, dimension (ldc, n).
 *                        On entry, the M-by-N matrix C.
 *                        On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.
 * @param[in]     ldc     The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work    Single complex array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of the array work.
 *                        If SIDE = 'L', lwork >= max(1, n);
 *                        if SIDE = 'R', lwork >= max(1, m).
 *                        For optimum performance lwork >= m*n.
 *                        If lwork = -1, then a workspace query is assumed.
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value
 */
void cunm22(const char* side, const char* trans,
            const int m, const int n, const int n1, const int n2,
            const c64* restrict Q, const int ldq,
            c64* restrict C, const int ldc,
            c64* restrict work, const int lwork,
            int* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    int left, lquery, notran;
    int i, ldwork, len, lwkopt, nb, nq, nw;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    if (left) {
        nq = m;
    } else {
        nq = n;
    }
    nw = nq;
    if (n1 == 0 || n2 == 0) nw = 1;

    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (n1 < 0 || n1 + n2 != nq) {
        *info = -5;
    } else if (n2 < 0) {
        *info = -6;
    } else if (ldq < (1 > nq ? 1 : nq)) {
        *info = -8;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -10;
    } else if (lwork < nw && !lquery) {
        *info = -12;
    }

    if (*info == 0) {
        lwkopt = m * n;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
    }

    if (*info != 0) {
        xerbla("CUNM22", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    /* Degenerate cases (n1 = 0 or n2 = 0) are handled using ZTRMM. */
    if (n1 == 0) {
        cblas_ctrmm(CblasColMajor,
                    left ? CblasLeft : CblasRight,
                    CblasUpper,
                    notran ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit,
                    m, n, &ONE, Q, ldq, C, ldc);
        work[0] = ONE;
        return;
    } else if (n2 == 0) {
        cblas_ctrmm(CblasColMajor,
                    left ? CblasLeft : CblasRight,
                    CblasLower,
                    notran ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit,
                    m, n, &ONE, Q, ldq, C, ldc);
        work[0] = ONE;
        return;
    }

    /* Compute the largest chunk size available from the workspace. */
    nb = 1 > ((lwork < lwkopt ? lwork : lwkopt) / nq)
         ? 1
         : (lwork < lwkopt ? lwork : lwkopt) / nq;

    if (left) {
        if (notran) {
            for (i = 0; i < n; i += nb) {
                len = nb < (n - i) ? nb : (n - i);
                ldwork = m;

                /* Multiply bottom part of C by Q12. */
                clacpy("A", n1, len, &C[n2 + i * ldc], ldc,
                       work, ldwork);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, len, &ONE, &Q[n2 * ldq], ldq,
                            work, ldwork);

                /* Multiply top part of C by Q11. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n1, len, n2,
                            &ONE, Q, ldq, &C[i * ldc], ldc,
                            &ONE, work, ldwork);

                /* Multiply top part of C by Q21. */
                clacpy("A", n2, len, &C[i * ldc], ldc,
                       work + n1, ldwork);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, len, &ONE, &Q[n1], ldq,
                            work + n1, ldwork);

                /* Multiply bottom part of C by Q22. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n2, len, n1,
                            &ONE, &Q[n1 + n2 * ldq], ldq,
                            &C[n2 + i * ldc], ldc,
                            &ONE, work + n1, ldwork);

                /* Copy everything back. */
                clacpy("A", m, len, work, ldwork, &C[i * ldc], ldc);
            }
        } else {
            for (i = 0; i < n; i += nb) {
                len = nb < (n - i) ? nb : (n - i);
                ldwork = m;

                /* Multiply bottom part of C by Q21**H. */
                clacpy("A", n2, len, &C[n1 + i * ldc], ldc,
                       work, ldwork);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            n2, len, &ONE, &Q[n1], ldq,
                            work, ldwork);

                /* Multiply top part of C by Q11**H. */
                cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            n2, len, n1,
                            &ONE, Q, ldq, &C[i * ldc], ldc,
                            &ONE, work, ldwork);

                /* Multiply top part of C by Q12**H. */
                clacpy("A", n1, len, &C[i * ldc], ldc,
                       work + n2, ldwork);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            n1, len, &ONE, &Q[n2 * ldq], ldq,
                            work + n2, ldwork);

                /* Multiply bottom part of C by Q22**H. */
                cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            n1, len, n2,
                            &ONE, &Q[n1 + n2 * ldq], ldq,
                            &C[n1 + i * ldc], ldc,
                            &ONE, work + n2, ldwork);

                /* Copy everything back. */
                clacpy("A", m, len, work, ldwork, &C[i * ldc], ldc);
            }
        }
    } else {
        if (notran) {
            for (i = 0; i < m; i += nb) {
                len = nb < (m - i) ? nb : (m - i);
                ldwork = len;

                /* Multiply right part of C by Q21. */
                clacpy("A", len, n2, &C[i + n1 * ldc], ldc,
                       work, ldwork);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            len, n2, &ONE, &Q[n1], ldq,
                            work, ldwork);

                /* Multiply left part of C by Q11. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            len, n2, n1,
                            &ONE, &C[i], ldc, Q, ldq,
                            &ONE, work, ldwork);

                /* Multiply left part of C by Q12. */
                clacpy("A", len, n1, &C[i], ldc,
                       work + n2 * ldwork, ldwork);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            len, n1, &ONE, &Q[n2 * ldq], ldq,
                            work + n2 * ldwork, ldwork);

                /* Multiply right part of C by Q22. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            len, n1, n2,
                            &ONE, &C[i + n1 * ldc], ldc,
                            &Q[n1 + n2 * ldq], ldq,
                            &ONE, work + n2 * ldwork, ldwork);

                /* Copy everything back. */
                clacpy("A", len, n, work, ldwork, &C[i], ldc);
            }
        } else {
            for (i = 0; i < m; i += nb) {
                len = nb < (m - i) ? nb : (m - i);
                ldwork = len;

                /* Multiply right part of C by Q12**H. */
                clacpy("A", len, n1, &C[i + n2 * ldc], ldc,
                       work, ldwork);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            len, n1, &ONE, &Q[n2 * ldq], ldq,
                            work, ldwork);

                /* Multiply left part of C by Q11**H. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            len, n1, n2,
                            &ONE, &C[i], ldc, Q, ldq,
                            &ONE, work, ldwork);

                /* Multiply left part of C by Q21**H. */
                clacpy("A", len, n2, &C[i], ldc,
                       work + n1 * ldwork, ldwork);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            len, n2, &ONE, &Q[n1], ldq,
                            work + n1 * ldwork, ldwork);

                /* Multiply right part of C by Q22**H. */
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            len, n2, n1,
                            &ONE, &C[i + n2 * ldc], ldc,
                            &Q[n1 + n2 * ldq], ldq,
                            &ONE, work + n1 * ldwork, ldwork);

                /* Copy everything back. */
                clacpy("A", len, n, work, ldwork, &C[i], ldc);
            }
        }
    }

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
