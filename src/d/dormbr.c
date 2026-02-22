/**
 * @file dormbr.c
 * @brief DORMBR applies the orthogonal matrix Q or P**T determined by DGEBRD.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * If VECT = 'Q', DORMBR overwrites the general real M-by-N matrix C
 * with
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'T':      Q**T * C       C * Q**T
 *
 * If VECT = 'P', DORMBR overwrites the general real M-by-N matrix C
 * with
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      P * C          C * P
 * TRANS = 'T':      P**T * C       C * P**T
 *
 * Here Q and P**T are the orthogonal matrices determined by DGEBRD when
 * reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
 * P**T are defined as products of elementary reflectors H(i) and G(i)
 * respectively.
 *
 * @param[in]     vect  = 'Q': apply Q or Q**T;
 *                      = 'P': apply P or P**T.
 * @param[in]     side  = 'L': apply Q, Q**T, P or P**T from the Left;
 *                      = 'R': apply Q, Q**T, P or P**T from the Right.
 * @param[in]     trans = 'N': No transpose, apply Q or P;
 *                      = 'T': Transpose, apply Q**T or P**T.
 * @param[in]     m     The number of rows of the matrix C. m >= 0.
 * @param[in]     n     The number of columns of the matrix C. n >= 0.
 * @param[in]     k     If vect = 'Q', the number of columns in the original
 *                      matrix reduced by DGEBRD.
 *                      If vect = 'P', the number of rows in the original
 *                      matrix reduced by DGEBRD.
 *                      k >= 0.
 * @param[in]     A     Double precision array, dimension
 *                      (lda, min(nq,k)) if vect = 'Q'
 *                      (lda, nq)        if vect = 'P'
 *                      The vectors which define the elementary reflectors H(i) and
 *                      G(i), whose products determine the matrices Q and P, as
 *                      returned by DGEBRD.
 * @param[in]     lda   The leading dimension of the array A.
 *                      If vect = 'Q', lda >= max(1,nq);
 *                      if vect = 'P', lda >= max(1,min(nq,k)).
 * @param[in]     tau   Double precision array, dimension (min(nq,k)).
 *                      tau[i] must contain the scalar factor of the elementary
 *                      reflector H(i) or G(i) which determines Q or P, as returned
 *                      by DGEBRD in the array argument TAUQ or TAUP.
 * @param[in,out] C     Double precision array, dimension (ldc, n).
 *                      On entry, the M-by-N matrix C.
 *                      On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q
 *                      or P*C or P**T*C or C*P or C*P**T.
 * @param[in]     ldc   The leading dimension of the array C. ldc >= max(1,m).
 * @param[out]    work  Double precision array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work.
 *                      If side = 'L', lwork >= max(1,n);
 *                      if side = 'R', lwork >= max(1,m).
 *                      For optimum performance lwork >= N*NB if side = 'L', and
 *                      lwork >= M*NB if side = 'R', where NB is the optimal
 *                      blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the work array, returns
 *                      this value as the first entry of the work array, and no error
 *                      message related to lwork is issued by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dormbr(const char* vect, const char* side, const char* trans,
            const INT m, const INT n, const INT k,
            const f64* restrict A, const INT lda,
            const f64* restrict tau,
            f64* restrict C, const INT ldc,
            f64* restrict work, const INT lwork, INT* info)
{
    INT applyq, left, lquery, notran;
    char transt;
    INT i1, i2, iinfo, lwkopt, mi, nb, ni, nq, nw;

    /* Test the input arguments */
    *info = 0;
    applyq = (vect[0] == 'Q' || vect[0] == 'q');
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    /* NQ is the order of Q or P and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = (1 > n) ? 1 : n;
    } else {
        nq = n;
        nw = (1 > m) ? 1 : m;
    }

    if (!applyq && !(vect[0] == 'P' || vect[0] == 'p')) {
        *info = -1;
    } else if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -2;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (k < 0) {
        *info = -6;
    } else if ((applyq && lda < ((1 > nq) ? 1 : nq)) ||
               (!applyq && lda < ((1 > ((nq < k) ? nq : k)) ? 1 : ((nq < k) ? nq : k)))) {
        *info = -8;
    } else if (ldc < ((1 > m) ? 1 : m)) {
        *info = -11;
    } else if (lwork < nw && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        if (applyq) {
            nb = lapack_get_nb("ORMQR");
        } else {
            nb = lapack_get_nb("ORMLQ");
        }
        if (nb < 1) nb = 1;
        lwkopt = nw * nb;
        work[0] = (f64)lwkopt;
    }

    if (*info != 0) {
        xerbla("DORMBR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    work[0] = 1.0;
    if (m == 0 || n == 0) {
        return;
    }

    if (applyq) {
        /* Apply Q */
        if (nq >= k) {
            /* Q was determined by a call to DGEBRD with nq >= k */
            dormqr(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &iinfo);
        } else if (nq > 1) {
            /* Q was determined by a call to DGEBRD with nq < k */
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
            dormqr(side, trans, mi, ni, nq - 1, &A[1], lda, tau,
                   &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);
        }
    } else {
        /* Apply P */
        if (notran) {
            transt = 'T';
        } else {
            transt = 'N';
        }
        if (nq > k) {
            /* P was determined by a call to DGEBRD with nq > k */
            dormlq(side, &transt, m, n, k, A, lda, tau, C, ldc, work, lwork, &iinfo);
        } else if (nq > 1) {
            /* P was determined by a call to DGEBRD with nq <= k */
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
            dormlq(side, &transt, mi, ni, nq - 1, &A[lda], lda, tau,
                   &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);
        }
    }
    work[0] = (f64)lwkopt;
}
