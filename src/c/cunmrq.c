/**
 * @file cunmrq.c
 * @brief CUNMRQ overwrites a general complex M-by-N matrix C with Q*C,
 *        Q**H*C, C*Q**H, or C*Q where Q is from an RQ factorization.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CUNMRQ overwrites the general complex M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *       Q = H(1)**H H(2)**H . . . H(k)**H
 *
 * as returned by CGERQF. Q is of order M if SIDE = 'L' and of order N
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
 * @param[in]     A      Single complex array, dimension
 *                                    (lda, m) if SIDE = 'L',
 *                                    (lda, n) if SIDE = 'R'
 *                       The i-th row must contain the vector which defines the
 *                       elementary reflector H(i), for i = 0,1,...,k-1, as
 *                       returned by CGERQF in the last k rows of its array
 *                       argument A.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, k).
 * @param[in]     tau    Single complex array, dimension (k).
 *                       tau[i] must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by CGERQF.
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
void cunmrq(const char* side, const char* trans,
            const int m, const int n, const int k,
            c64* restrict A, const int lda,
            const c64* restrict tau,
            c64* restrict C, const int ldc,
            c64* restrict work, const int lwork,
            int* info)
{
    const int nbmax = 64;
    const int ldt = nbmax + 1;
    const int tsize = ldt * nbmax;

    int left, notran, lquery;
    char transt;
    int i, i1, i2, i3, ib, iinfo, iwt, ldwork, lwkopt;
    int mi = 0, nb, nbmin, ni = 0, nq, nw;

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
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (lwork < nw && !lquery) {
        *info = -12;
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
        xerbla("CUNMRQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

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
        cunmr2(side, trans, m, n, k, A, lda, tau, C, ldc, work, &iinfo);
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
        } else {
            mi = m;
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
                clarft("B", "R", nq - k + i + ib, ib,
                       &A[i + 0 * lda], lda, &tau[i], &work[iwt], ldt);
                if (left) {
                    /* H or H**H is applied to C(0:m-k+i+ib-1, 0:n-1) */
                    mi = m - k + i + ib;
                } else {
                    /* H or H**H is applied to C(0:m-1, 0:n-k+i+ib-1) */
                    ni = n - k + i + ib;
                }

                /* Apply H or H**H */
                clarfb(side, &transt, "B", "R", mi, ni, ib,
                       &A[i + 0 * lda], lda, &work[iwt], ldt,
                       C, ldc, work, ldwork);
            }
        } else {
            for (i = i1; i >= i2; i += i3) {
                ib = nb < k - i ? nb : k - i;

                clarft("B", "R", nq - k + i + ib, ib,
                       &A[i + 0 * lda], lda, &tau[i], &work[iwt], ldt);
                if (left) {
                    mi = m - k + i + ib;
                } else {
                    ni = n - k + i + ib;
                }

                clarfb(side, &transt, "B", "R", mi, ni, ib,
                       &A[i + 0 * lda], lda, &work[iwt], ldt,
                       C, ldc, work, ldwork);
            }
        }
    }
    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
