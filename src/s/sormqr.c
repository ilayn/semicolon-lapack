/**
 * @file sormqr.c
 * @brief SORMQR multiplies a general matrix by the orthogonal matrix from
 *        a QR factorization determined by SGEQRF (blocked algorithm).
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SORMQR overwrites the general real M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'T':   Q^T * C        C * Q^T
 *
 * where Q is a real orthogonal matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by SGEQRF. Q is of order M if SIDE = 'L' and of order N
 * if SIDE = 'R'.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     side   'L': apply Q or Q^T from the Left;
 *                       'R': apply Q or Q^T from the Right.
 * @param[in]     trans  'N': apply Q (No transpose);
 *                       'T': apply Q^T (Transpose).
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     k      The number of elementary reflectors.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     A      The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), as returned by SGEQRF.
 *                       Dimension (lda, k).
 * @param[in]     lda    Leading dimension of A.
 *                       If SIDE = "L", lda >= max(1, m);
 *                       if SIDE = "R", lda >= max(1, n).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by SGEQRF.
 * @param[in,out] C      On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work.
 *                       If SIDE = "L", lwork >= max(1, n);
 *                       if SIDE = "R", lwork >= max(1, m).
 *                       If lwork == -1, workspace query only.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value.
 */
void sormqr(const char* side, const char* trans,
            const int m, const int n, const int k,
            const float * const restrict A, const int lda,
            const float * const restrict tau,
            float * const restrict C, const int ldc,
            float * const restrict work, const int lwork,
            int *info)
{
    /* NBMAX is the maximum block size; LDT is the leading dimension of
     * the T array stored in WORK; TSIZE is the size of the T array. */
    const int NBMAX = 64;
    const int LDT = NBMAX + 1;
    const int TSIZE = LDT * NBMAX;

    int left, notran, lquery;
    int i, ib, ic, jc, iinfo, iwt, ldwork, lwkopt;
    int mi, nb, nbmin, ni, nq, nw;
    int i1, i2, i3;

    /* Decode arguments */
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n > 1 ? n : 1;
    } else {
        nq = n;
        nw = m > 1 ? m : 1;
    }

    /* Parameter validation */
    *info = 0;
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
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
        /* Compute optimal workspace */
        nb = lapack_get_nb("ORMQR");
        if (nb > NBMAX) {
            nb = NBMAX;
        }
        lwkopt = nw * nb + TSIZE;
        work[0] = (float)lwkopt;
    }

    if (*info != 0) {
        xerbla("SORMQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        work[0] = 1.0f;
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
        /* Use unblocked code */
        sorm2r(side, trans, m, n, k, A, lda, tau, C, ldc, work, &iinfo);
    } else {
        /* Use blocked code */
        iwt = nw * nb;  /* offset into work for the T array */

        /* Determine loop direction:
         * Left+Trans or Right+NoTrans: i = 0, nb, 2*nb, ... (forward)
         * Left+NoTrans or Right+Trans: i = ..., 2*nb, nb, 0 (backward) */
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

            /* Form the triangular factor of the block reflector
             * H = H(i) H(i+1) . . . H(i+ib-1) */
            slarft("F", "C", nq - i, ib,
                   &A[i + i * lda], lda,
                   &tau[i], &work[iwt], LDT);

            if (left) {
                /* H or H^T is applied to C(i:m-1, 0:n-1) */
                mi = m - i;
                ic = i;
            } else {
                /* H or H^T is applied to C(0:m-1, i:n-1) */
                ni = n - i;
                jc = i;
            }

            /* Apply H or H^T */
            slarfb(side, trans, "F", "C", mi, ni, ib,
                   &A[i + i * lda], lda,
                   &work[iwt], LDT,
                   &C[ic + jc * ldc], ldc,
                   work, ldwork);
        }
    }

    work[0] = (float)lwkopt;
}
