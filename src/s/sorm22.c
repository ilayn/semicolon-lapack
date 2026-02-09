/**
 * @file sorm22.c
 * @brief SORM22 multiplies a general matrix by a banded orthogonal matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SORM22 overwrites the general real M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'T':      Q**T * C       C * Q**T
 *
 * where Q is a real orthogonal matrix of order NQ, with NQ = M if
 * SIDE = 'L' and NQ = N if SIDE = 'R'.
 * The orthogonal matrix Q processes a 2-by-2 block structure
 *
 *        [  Q11  Q12  ]
 *    Q = [            ]
 *        [  Q21  Q22  ],
 *
 * where Q12 is an N1-by-N1 lower triangular matrix and Q21 is an
 * N2-by-N2 upper triangular matrix.
 *
 * @param[in]     side    = 'L': apply Q or Q**T from the Left;
 *                         = 'R': apply Q or Q**T from the Right.
 * @param[in]     trans   = 'N': apply Q (No transpose);
 *                         = 'T': apply Q**T (Transpose).
 * @param[in]     m       The number of rows of the matrix C. m >= 0.
 * @param[in]     n       The number of columns of the matrix C. n >= 0.
 * @param[in]     n1      The dimension of Q12. n1 >= 0.
 * @param[in]     n2      The dimension of Q21. n2 >= 0.
 *                        n1 + n2 = M if SIDE = 'L' and n1 + n2 = N if SIDE = 'R'.
 * @param[in]     Q       Array of dimension (ldq, M) if SIDE = 'L',
 *                        (ldq, N) if SIDE = 'R'.
 * @param[in]     ldq     The leading dimension of Q.
 *                        ldq >= max(1,M) if SIDE = 'L'; ldq >= max(1,N) if SIDE = 'R'.
 * @param[in,out] C       Array of dimension (ldc, n). On entry, the M-by-N matrix C.
 *                        On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1,m).
 * @param[out]    work    Workspace array of dimension (lwork).
 * @param[in]     lwork   The dimension of work.
 *                        If SIDE = 'L', lwork >= max(1,n);
 *                        if SIDE = 'R', lwork >= max(1,m).
 *                        For optimum performance lwork >= m*n.
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void sorm22(
    const char* side,
    const char* trans,
    const int m,
    const int n,
    const int n1,
    const int n2,
    const float* const restrict Q,
    const int ldq,
    float* const restrict C,
    const int ldc,
    float* const restrict work,
    const int lwork,
    int* info)
{
    const float ONE = 1.0f;

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
               !(trans[0] == 'T' || trans[0] == 't')) {
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
        work[0] = (float)lwkopt;
    }

    if (*info != 0) {
        xerbla("SORM22", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (m == 0 || n == 0) {
        work[0] = 1;
        return;
    }

    if (n1 == 0) {
        cblas_strmm(CblasColMajor,
                    left ? CblasLeft : CblasRight,
                    CblasUpper,
                    notran ? CblasNoTrans : CblasTrans,
                    CblasNonUnit,
                    m, n, ONE, Q, ldq, C, ldc);
        work[0] = ONE;
        return;
    } else if (n2 == 0) {
        cblas_strmm(CblasColMajor,
                    left ? CblasLeft : CblasRight,
                    CblasLower,
                    notran ? CblasNoTrans : CblasTrans,
                    CblasNonUnit,
                    m, n, ONE, Q, ldq, C, ldc);
        work[0] = ONE;
        return;
    }

    nb = lwkopt / nq;
    if (nb < 1) nb = 1;
    if (nb > lwork / nq) nb = lwork / nq;
    if (nb < 1) nb = 1;

    if (left) {
        if (notran) {
            for (i = 0; i < n; i += nb) {
                len = nb;
                if (len > n - i) len = n - i;
                ldwork = m;

                slacpy("A", n1, len, &C[n2 + i * ldc], ldc, work, ldwork);
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n1, len, ONE, &Q[n2 * ldq], ldq, work, ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n1, len, n2,
                            ONE, Q, ldq, &C[i * ldc], ldc, ONE, work, ldwork);

                slacpy("A", n2, len, &C[i * ldc], ldc, &work[n1], ldwork);
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            CblasNonUnit, n2, len, ONE, &Q[n1], ldq, &work[n1], ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, len, n1,
                            ONE, &Q[n1 + n2 * ldq], ldq, &C[n2 + i * ldc], ldc,
                            ONE, &work[n1], ldwork);

                slacpy("A", m, len, work, ldwork, &C[i * ldc], ldc);
            }
        } else {
            for (i = 0; i < n; i += nb) {
                len = nb;
                if (len > n - i) len = n - i;
                ldwork = m;

                slacpy("A", n2, len, &C[n1 + i * ldc], ldc, work, ldwork);
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                            CblasNonUnit, n2, len, ONE, &Q[n1], ldq, work, ldwork);

                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n2, len, n1,
                            ONE, Q, ldq, &C[i * ldc], ldc, ONE, work, ldwork);

                slacpy("A", n1, len, &C[i * ldc], ldc, &work[n2], ldwork);
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasNonUnit, n1, len, ONE, &Q[n2 * ldq], ldq, &work[n2], ldwork);

                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n1, len, n2,
                            ONE, &Q[n1 + n2 * ldq], ldq, &C[n1 + i * ldc], ldc,
                            ONE, &work[n2], ldwork);

                slacpy("A", m, len, work, ldwork, &C[i * ldc], ldc);
            }
        }
    } else {
        if (notran) {
            for (i = 0; i < m; i += nb) {
                len = nb;
                if (len > m - i) len = m - i;
                ldwork = len;

                slacpy("A", len, n2, &C[i + n1 * ldc], ldc, work, ldwork);
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                            CblasNonUnit, len, n2, ONE, &Q[n1], ldq, work, ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, len, n2, n1,
                            ONE, &C[i], ldc, Q, ldq, ONE, work, ldwork);

                slacpy("A", len, n1, &C[i], ldc, &work[n2 * ldwork], ldwork);
                cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                            CblasNonUnit, len, n1, ONE, &Q[n2 * ldq], ldq,
                            &work[n2 * ldwork], ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, len, n1, n2,
                            ONE, &C[i + n1 * ldc], ldc, &Q[n1 + n2 * ldq], ldq,
                            ONE, &work[n2 * ldwork], ldwork);

                slacpy("A", len, n, work, ldwork, &C[i], ldc);
            }
        } else {
            for (i = 0; i < m; i += nb) {
                len = nb;
                if (len > m - i) len = m - i;
                ldwork = len;

                slacpy("A", len, n1, &C[i + n2 * ldc], ldc, work, ldwork);
                cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                            CblasNonUnit, len, n1, ONE, &Q[n2 * ldq], ldq, work, ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, len, n1, n2,
                            ONE, &C[i], ldc, Q, ldq, ONE, work, ldwork);

                slacpy("A", len, n2, &C[i], ldc, &work[n1 * ldwork], ldwork);
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                            CblasNonUnit, len, n2, ONE, &Q[n1], ldq,
                            &work[n1 * ldwork], ldwork);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, len, n2, n1,
                            ONE, &C[i + n2 * ldc], ldc, &Q[n1 + n2 * ldq], ldq,
                            ONE, &work[n1 * ldwork], ldwork);

                slacpy("A", len, n, work, ldwork, &C[i], ldc);
            }
        }
    }

    work[0] = (float)lwkopt;
}
