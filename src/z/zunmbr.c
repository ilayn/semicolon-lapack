/**
 * @file zunmbr.c
 * @brief ZUNMBR multiplies a general matrix by the unitary matrix Q or P
 *        determined by ZGEBRD.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * If VECT = 'Q', ZUNMBR overwrites the general complex M-by-N matrix C
 * with
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * If VECT = 'P', ZUNMBR overwrites the general complex M-by-N matrix C
 * with
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      P * C          C * P
 * TRANS = 'C':      P**H * C       C * P**H
 *
 * Here Q and P**H are the unitary matrices determined by ZGEBRD when
 * reducing a complex matrix A to bidiagonal form: A = Q * B * P**H. Q
 * and P**H are defined as products of elementary reflectors H(i) and
 * G(i) respectively.
 *
 * Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
 * order of the unitary matrix Q or P**H that is applied.
 *
 * If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
 * if nq >= k, Q = H(1) H(2) . . . H(k);
 * if nq < k, Q = H(1) H(2) . . . H(nq-1).
 *
 * If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
 * if k < nq, P = G(1) G(2) . . . G(k);
 * if k >= nq, P = G(1) G(2) . . . G(nq-1).
 *
 * @param[in]     vect   'Q': apply Q or Q**H;
 *                       'P': apply P or P**H.
 * @param[in]     side   'L': apply Q, Q**H, P or P**H from the Left;
 *                       'R': apply Q, Q**H, P or P**H from the Right.
 * @param[in]     trans  'N': No transpose, apply Q or P;
 *                       'C': Conjugate transpose, apply Q**H or P**H.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      If VECT = 'Q', the number of columns in the original
 *                       matrix reduced by ZGEBRD.
 *                       If VECT = 'P', the number of rows in the original
 *                       matrix reduced by ZGEBRD.
 *                       k >= 0.
 * @param[in]     A      Double complex array, dimension
 *                       (lda, min(nq,k)) if VECT = 'Q'
 *                       (lda, nq)        if VECT = 'P'
 *                       The vectors which define the elementary reflectors
 *                       H(i) and G(i), whose products determine the matrices
 *                       Q and P, as returned by ZGEBRD.
 * @param[in]     lda    The leading dimension of the array A.
 *                       If VECT = 'Q', lda >= max(1, nq);
 *                       if VECT = 'P', lda >= max(1, min(nq, k)).
 * @param[in]     tau    Double complex array, dimension (min(nq, k)).
 *                       tau[i] must contain the scalar factor of the
 *                       elementary reflector H(i) or G(i) which determines
 *                       Q or P, as returned by ZGEBRD in the array argument
 *                       TAUQ or TAUP.
 * @param[in,out] C      Double complex array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or C*Q**H
 *                       or C*Q or P*C or P**H*C or C*P or C*P**H.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double complex array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If SIDE = 'L', lwork >= max(1, n);
 *                       if SIDE = 'R', lwork >= max(1, m);
 *                       if n = 0 or m = 0, lwork >= 1.
 *                       For optimum performance lwork >= max(1, n*NB) if
 *                       SIDE = 'L', and lwork >= max(1, m*NB) if SIDE = 'R',
 *                       where NB is the optimal blocksize. (NB = 0 if M = 0
 *                       or N = 0.)
 *
 *                       If lwork = -1, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the work
 *                       array, returns this value as the first entry of the
 *                       work array, and no error message related to lwork is
 *                       issued by xerbla.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void zunmbr(const char* vect, const char* side, const char* trans,
            const int m, const int n, const int k,
            c128* restrict A, const int lda,
            const c128* restrict tau,
            c128* restrict C, const int ldc,
            c128* restrict work, const int lwork,
            int* info)
{
    int applyq, left, lquery, notran;
    int i1, i2, iinfo, lwkopt, mi, nb, ni, nq, nw;

    *info = 0;
    applyq = (vect[0] == 'Q' || vect[0] == 'q');
    left   = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    /* NQ is the order of Q or P and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n > 1 ? n : 1;
    } else {
        nq = n;
        nw = m > 1 ? m : 1;
    }
    if (!applyq && !(vect[0] == 'P' || vect[0] == 'p')) {
        *info = -1;
    } else if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -2;
    } else if (!notran && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (k < 0) {
        *info = -6;
    } else if ((applyq && lda < (nq > 1 ? nq : 1)) ||
               (!applyq && lda < ((nq < k ? nq : k) > 1 ? (nq < k ? nq : k) : 1))) {
        *info = -8;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -11;
    } else if (lwork < nw && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        if (m > 0 && n > 0) {
            if (applyq) {
                nb = lapack_get_nb("ORMQR");
            } else {
                nb = lapack_get_nb("ORMLQ");
            }
            lwkopt = nw * nb;
        } else {
            lwkopt = 1;
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZUNMBR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    if (applyq) {

        /* Apply Q */

        if (nq >= k) {

            /* Q was determined by a call to ZGEBRD with nq >= k */

            zunmqr(side, trans, m, n, k, A, lda, tau, C, ldc,
                   work, lwork, &iinfo);
        } else if (nq > 1) {

            /* Q was determined by a call to ZGEBRD with nq < k */

            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 1;
                i2 = 0;
            } else {
                mi = m;
                ni = n - 1;
                i1 = 0;
                i2 = 1;
            }
            zunmqr(side, trans, mi, ni, nq - 1, &A[1], lda, tau,
                   &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);
        }
    } else {

        /* Apply P */

        const char* transt;
        if (notran) {
            transt = "C";
        } else {
            transt = "N";
        }
        if (nq > k) {

            /* P was determined by a call to ZGEBRD with nq > k */

            zunmlq(side, transt, m, n, k, A, lda, tau, C, ldc,
                   work, lwork, &iinfo);
        } else if (nq > 1) {

            /* P was determined by a call to ZGEBRD with nq <= k */

            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 1;
                i2 = 0;
            } else {
                mi = m;
                ni = n - 1;
                i1 = 0;
                i2 = 1;
            }
            zunmlq(side, transt, mi, ni, nq - 1, &A[lda], lda,
                   tau, &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);
        }
    }
    work[0] = CMPLX((f64)lwkopt, 0.0);
}
