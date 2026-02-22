/**
 * @file cunmrz.c
 * @brief CUNMRZ overwrites a general complex M-by-N matrix C with Q*C,
 *        Q**H*C, C*Q**H, or C*Q where Q is from CTZRZF.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CUNMRZ overwrites the general complex M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *       Q = H(1) H(2) . . . H(k)
 *
 * as returned by CTZRZF. Q is of order M if SIDE = 'L' and of order N
 * if SIDE = 'R'.
 *
 * @param[in]     side   'L': apply Q or Q**H from the Left;
 *                       'R': apply Q or Q**H from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'C': Conjugate transpose, apply Q**H.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = 'L', m >= k >= 0;
 *                       if SIDE = 'R', n >= k >= 0.
 * @param[in]     l      The number of columns of the matrix A containing
 *                       the meaningful part of the Householder reflectors.
 *                       If SIDE = 'L', m >= l >= 0, if SIDE = 'R', n >= l >= 0.
 * @param[in]     A      Single complex array, dimension
 *                                    (lda, m) if SIDE = 'L',
 *                                    (lda, n) if SIDE = 'R'
 *                       The i-th row must contain the vector which defines the
 *                       elementary reflector H(i), for i = 0,1,...,k-1, as
 *                       returned by CTZRZF in the last k rows of its array
 *                       argument A.
 *                       A is modified by the routine but restored on exit.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, k).
 * @param[in]     tau    Single complex array, dimension (k).
 *                       tau[i] must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by CTZRZF.
 * @param[in,out] C      Single complex array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or
 *                       C*Q**H or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Single complex array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If SIDE = 'L', lwork >= max(1, n);
 *                       if SIDE = 'R', lwork >= max(1, m).
 *                       For good performance, lwork should generally be larger.
 *
 *                       If lwork = -1, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the work
 *                       array, returns this value as the first entry of the
 *                       work array, and no error message related to lwork is
 *                       issued by xerbla.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void cunmrz(const char* side, const char* trans,
            const INT m, const INT n, const INT k, const INT l,
            c64* restrict A, const INT lda,
            const c64* restrict tau,
            c64* restrict C, const INT ldc,
            c64* restrict work, const INT lwork,
            INT* info)
{
    const INT nbmax = 64;
    const INT ldt = nbmax + 1;
    const INT tsize = ldt * nbmax;

    INT left, notran, lquery;
    char transt;
    INT i, i1, i2, i3, ib, ic, iinfo, iwt, ja, jc;
    INT ldwork, lwkopt, mi = 0, nb, nbmin, ni = 0, nq, nw;

    *info = 0;
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
        /* Compute the workspace requirements */
        if (m == 0 || n == 0) {
            lwkopt = 1;
        } else {
            nb = nbmax < lapack_get_nb("ORMRQ") ? nbmax : lapack_get_nb("ORMRQ");
            lwkopt = nw * nb + tsize;
        }
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
    }

    if (*info != 0) {
        xerbla("CUNMRZ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Determine the block size. NB may be at most NBMAX, where NBMAX
       is used to define the local array T. */
    nb = nbmax < lapack_get_nb("ORMRQ") ? nbmax : lapack_get_nb("ORMRQ");
    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < k) {
        if (lwork < lwkopt) {
            nb = (lwork - tsize) / ldwork;
            nbmin = lapack_get_nbmin("ORMRQ") > 2 ? lapack_get_nbmin("ORMRQ") : 2;
        }
    }

    if (nb < nbmin || nb >= k) {
        /* Use unblocked code */
        cunmr3(side, trans, m, n, k, l, A, lda, tau, C, ldc, work, &iinfo);
    } else {
        /* Use blocked code */
        iwt = nw * nb;
        if ((left && !notran) || (!left && notran)) {
            i1 = 0;
            i2 = k;
            i3 = nb;
        } else {
            i1 = ((k - 1) / nb) * nb;
            i2 = 0;
            i3 = -nb;
        }

        if (left) {
            ni = n;
            jc = 0;
            ja = m - l;
        } else {
            mi = m;
            ic = 0;
            ja = n - l;
        }

        if (notran) {
            transt = 'C';
        } else {
            transt = 'N';
        }

        if (i3 > 0) {
            for (i = i1; i < i2; i += i3) {
                ib = nb < k - i ? nb : k - i;

                /* Form the triangular factor of the block reflector
                   H = H(i+ib-1) . . . H(i+1) H(i) */
                clarzt("B", "R", l, ib,
                       &A[i + ja * lda], lda, &tau[i], &work[iwt], ldt);

                if (left) {
                    /* H or H**H is applied to C(i:m-1, 0:n-1) */
                    mi = m - i;
                    ic = i;
                } else {
                    /* H or H**H is applied to C(0:m-1, i:n-1) */
                    ni = n - i;
                    jc = i;
                }

                /* Apply H or H**H */
                clarzb(side, &transt, "B", "R", mi, ni,
                       ib, l, &A[i + ja * lda], lda, &work[iwt], ldt,
                       &C[ic + jc * ldc], ldc, work, ldwork);
            }
        } else {
            for (i = i1; i >= i2; i += i3) {
                ib = nb < k - i ? nb : k - i;

                clarzt("B", "R", l, ib,
                       &A[i + ja * lda], lda, &tau[i], &work[iwt], ldt);

                if (left) {
                    mi = m - i;
                    ic = i;
                } else {
                    ni = n - i;
                    jc = i;
                }

                clarzb(side, &transt, "B", "R", mi, ni,
                       ib, l, &A[i + ja * lda], lda, &work[iwt], ldt,
                       &C[ic + jc * ldc], ldc, work, ldwork);
            }
        }
    }
    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
