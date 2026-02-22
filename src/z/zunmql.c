/**
 * @file zunmql.c
 * @brief ZUNMQL multiplies a general matrix by the unitary matrix from
 *        a QL factorization determined by ZGEQLF (blocked algorithm).
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include "../include/lapack_tuning.h"
#include <complex.h>

/**
 * ZUNMQL overwrites the general complex M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'C':   Q^H * C        C * Q^H
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by ZGEQLF. Q is of order M if SIDE = 'L' and of order N
 * if SIDE = 'R'.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     side   'L': apply Q or Q^H from the Left;
 *                       'R': apply Q or Q^H from the Right.
 * @param[in]     trans  'N': apply Q (No transpose);
 *                       'C': apply Q^H (Conjugate transpose).
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     k      The number of elementary reflectors.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     A      The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), as returned by ZGEQLF
 *                       in the last k columns. Dimension (lda, k).
 * @param[in]     lda    Leading dimension of A.
 *                       If SIDE = "L", lda >= max(1, m);
 *                       if SIDE = "R", lda >= max(1, n).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by ZGEQLF.
 * @param[in,out] C      On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q^H*C or C*Q^H or C*Q.
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
void zunmql(const char* side, const char* trans,
            const INT m, const INT n, const INT k,
            const c128* restrict A, const INT lda,
            const c128* restrict tau,
            c128* restrict C, const INT ldc,
            c128* restrict work, const INT lwork,
            INT* info)
{
    const INT NBMAX = 64;
    const INT LDT = NBMAX + 1;
    const INT TSIZE = LDT * NBMAX;

    INT left, notran, lquery;
    INT i, ib, iinfo, iwt, ldwork, lwkopt;
    INT mi, nb, nbmin, ni, nq, nw;
    INT i1, i2, i3;

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
        /* Compute optimal workspace */
        if (m == 0 || n == 0) {
            lwkopt = 1;
        } else {
            nb = lapack_get_nb("ORMQL");
            if (nb > NBMAX) {
                nb = NBMAX;
            }
            lwkopt = nw * nb + TSIZE;
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZUNMQL", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    nb = lapack_get_nb("ORMQL");
    if (nb > NBMAX) {
        nb = NBMAX;
    }
    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = (lwork - TSIZE) / ldwork;
            nbmin = lapack_get_nbmin("ORMQL");
        }
    }

    if (nb < nbmin || nb >= k) {
        /* Use unblocked code */
        zunm2l(side, trans, m, n, k, A, lda, tau, C, ldc, work, &iinfo);
    } else {
        /* Use blocked code */
        iwt = nw * nb;  /* offset into work for the T array */

        if ((left && notran) || (!left && !notran)) {
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
        } else {
            mi = m;
        }

        for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
            ib = nb < k - i ? nb : k - i;

            /* Form the triangular factor of the block reflector
             * H = H(i+ib-1) . . . H(i+1) H(i) */
            zlarft("B", "C", nq - k + i + ib, ib,
                   &A[0 + i * lda], lda,
                   &tau[i], &work[iwt], LDT);

            if (left) {
                /* H or H^H is applied to C(0:m-k+i+ib-1, 0:n-1) */
                mi = m - k + i + ib;
            } else {
                /* H or H^H is applied to C(0:m-1, 0:n-k+i+ib-1) */
                ni = n - k + i + ib;
            }

            /* Apply H or H^H */
            zlarfb(side, trans, "B", "C", mi, ni, ib,
                   &A[0 + i * lda], lda,
                   &work[iwt], LDT,
                   C, ldc,
                   work, ldwork);
        }
    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
