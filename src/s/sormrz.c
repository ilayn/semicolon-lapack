/**
 * @file sormrz.c
 * @brief SORMRZ multiplies a general matrix by the orthogonal matrix from
 *        an RZ factorization determined by STZRZF (blocked algorithm).
 */

#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SORMRZ overwrites the general real M-by-N matrix C with
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
 * as returned by STZRZF. Q is of order M if SIDE = 'L' and of order N
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
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     l      The number of columns of the matrix A containing
 *                       the meaningful part of the Householder reflectors.
 *                       If SIDE = "L", m >= l >= 0;
 *                       if SIDE = "R", n >= l >= 0.
 * @param[in,out] A      Double precision array, dimension
 *                       (lda, m) if SIDE = "L",
 *                       (lda, n) if SIDE = 'R'.
 *                       The i-th row must contain the vector which defines
 *                       the elementary reflector H(i), for i = 0,1,...,k-1,
 *                       as returned by STZRZF in the last k rows of its
 *                       array argument A.
 *                       A is modified by the routine but restored on exit.
 * @param[in]     lda    Leading dimension of A. lda >= max(1, k).
 * @param[in]     tau    Double precision array, dimension (k).
 *                       tau[i] must contain the scalar factor of the
 *                       elementary reflector H(i), as returned by STZRZF.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q^T*C or
 *                       C*Q^T or C*Q.
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Double precision array, dimension (max(1, lwork)).
 *                       On exit, work[0] returns the optimal lwork.
 * @param[in]     lwork  Dimension of work.
 *                       If SIDE = "L", lwork >= max(1, n);
 *                       if SIDE = "R", lwork >= max(1, m).
 *                       For good performance, lwork should generally be larger.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sormrz(const char* side, const char* trans,
            const int m, const int n, const int k, const int l,
            f32 * const restrict A, const int lda,
            const f32 * const restrict tau,
            f32 * const restrict C, const int ldc,
            f32 * const restrict work, const int lwork,
            int *info)
{
    /* NBMAX is the maximum block size (hardcoded in LAPACK Fortran source);
     * LDT is the leading dimension of the T array stored in WORK;
     * TSIZE is the size of the T array. */
    const int NBMAX = 64;
    const int ldt_val = NBMAX + 1;  /* 65 */
    const int TSIZE = ldt_val * NBMAX;  /* 65*64 = 4160 */

    int left, notran, lquery;
    int i, ib, ic, iinfo, iwt, ja, jc;
    int ldwork, lwkopt, mi = 0, nb, nbmin, ni = 0, nq, nw;
    int i1, i2, i3;
    char transt;

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
    if (!left && side[0] != 'R' && side[0] != 'r') {
        *info = -1;
    } else if (!notran && trans[0] != 'T' && trans[0] != 't') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (l < 0 || (left && l > m) || (!left && l > n)) {
        *info = -6;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -8;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -11;
    } else if (lwork < nw && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        /* Compute optimal workspace */
        if (m == 0 || n == 0) {
            lwkopt = 1;
        } else {
            nb = lapack_get_nb("ORMRQ");
            if (nb > NBMAX) {
                nb = NBMAX;
            }
            lwkopt = nw * nb + TSIZE;
        }
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SORMRZ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        work[0] = 1.0f;
        return;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = (lwork - TSIZE) / ldwork;
            nbmin = lapack_get_nbmin("ORMRQ");
        }
    }

    if (nb < nbmin || nb >= k) {
        /* Use unblocked code */
        sormr3(side, trans, m, n, k, l, A, lda, tau, C, ldc, work, &iinfo);
    } else {
        /* Use blocked code */
        iwt = nw * nb;  /* 0-based offset into work for the T array */

        /* Determine loop direction:
         * (Left && Trans) or (Right && NoTrans): forward (i1=0, step=nb)
         * (Left && NoTrans) or (Right && Trans): backward */
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
            ja = m - l;  /* 0-based: Fortran JA = M-L+1 -> ja = m-l */
        } else {
            mi = m;
            ic = 0;
            ja = n - l;  /* 0-based: Fortran JA = N-L+1 -> ja = n-l */
        }

        /* TRANS inversion for SLARZB:
         * If NOTRAN: transt = 'T'
         * If TRANS:  transt = 'N' */
        if (notran) {
            transt = 'T';
        } else {
            transt = 'N';
        }

        for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Form the triangular factor of the block reflector
             * H = H(i+ib-1) . . . H(i+1) H(i) */
            slarzt("B", "R", l, ib,
                   &A[i + ja * lda], lda,
                   &tau[i], &work[iwt], ldt_val);

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
            slarzb(side, &transt, "B", "R", mi, ni, ib, l,
                   &A[i + ja * lda], lda,
                   &work[iwt], ldt_val,
                   &C[ic + jc * ldc], ldc,
                   work, ldwork);
        }
    }

    work[0] = (f32)lwkopt;
}
