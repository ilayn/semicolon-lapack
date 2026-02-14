/**
 * @file zunmqr.c
 * @brief ZUNMQR multiplies a general matrix by the unitary matrix from
 *        a QR factorization determined by ZGEQRF (blocked algorithm).
 */

#include <cblas.h>
#include <complex.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZUNMQR overwrites the general complex M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'C':   Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
 * if SIDE = 'R'.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     side   'L': apply Q or Q**H from the Left;
 *                       'R': apply Q or Q**H from the Right.
 * @param[in]     trans  'N': apply Q (No transpose);
 *                       'C': apply Q**H (Conjugate transpose).
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     k      The number of elementary reflectors.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     A      The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), as returned by ZGEQRF.
 *                       Dimension (lda, k).
 * @param[in]     lda    Leading dimension of A.
 *                       If SIDE = "L", lda >= max(1, m);
 *                       if SIDE = "R", lda >= max(1, n).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by ZGEQRF.
 * @param[in,out] C      On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work.
 *                       If SIDE = "L", lwork >= max(1, n);
 *                       if SIDE = "R", lwork >= max(1, m).
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zunmqr(const char* side, const char* trans,
            const int m, const int n, const int k,
            const c128* const restrict A, const int lda,
            const c128* const restrict tau,
            c128* const restrict C, const int ldc,
            c128* const restrict work, const int lwork,
            int* info)
{
    const int NBMAX = 64;
    const int LDT = NBMAX + 1;
    const int TSIZE = LDT * NBMAX;

    int left, notran, lquery;
    int i, ib, ic, jc, iinfo, iwt, ldwork, lwkopt;
    int mi, nb, nbmin, ni, nq, nw;
    int i1, i2, i3;

    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    if (left) {
        nq = m;
        nw = n > 1 ? n : 1;
    } else {
        nq = n;
        nw = m > 1 ? m : 1;
    }

    *info = 0;
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (lda < (nq > 1 ? nq : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (lwork < nw && !lquery) {
        *info = -12;
    }

    if (*info == 0) {
        nb = lapack_get_nb("ORMQR");
        if (nb > NBMAX) {
            nb = NBMAX;
        }
        lwkopt = nw * nb + TSIZE;
        work[0] = CMPLX((f64)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZUNMQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (m == 0 || n == 0 || k == 0) {
        work[0] = CMPLX(1.0, 0.0);
        return;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = (lwork - TSIZE) / ldwork;
            nbmin = lapack_get_nbmin("ORMQR");
        }
    }

    if (nb < nbmin || nb >= k) {
        zunm2r(side, trans, m, n, k, A, lda, tau, C, ldc, work, &iinfo);
    } else {
        iwt = nw * nb;

        if ((left && !notran) || (!left && notran)) {
            i1 = 0;
            i2 = k - 1;
            i3 = nb;
        } else {
            i1 = ((k - 1) / nb) * nb;
            i2 = 0;
            i3 = -nb;
        }

        if (left) {
            ni = n;
            jc = 0;
        } else {
            mi = m;
            ic = 0;
        }

        for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
            ib = nb < k - i ? nb : k - i;

            zlarft("F", "C", nq - i, ib,
                   &A[i + i * lda], lda,
                   &tau[i], &work[iwt], LDT);

            if (left) {
                mi = m - i;
                ic = i;
            } else {
                ni = n - i;
                jc = i;
            }

            zlarfb(side, trans, "F", "C", mi, ni, ib,
                   &A[i + i * lda], lda,
                   &work[iwt], LDT,
                   &C[ic + jc * ldc], ldc,
                   work, ldwork);
        }
    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
