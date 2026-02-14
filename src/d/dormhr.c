/**
 * @file dormhr.c
 * @brief DORMHR overwrites matrix C with Q*C or C*Q or their transposes,
 *        where Q is from the Hessenberg reduction produced by DGEHRD.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DORMHR overwrites the general real M-by-N matrix C with
 *
 *     SIDE = 'L'     SIDE = 'R'
 *     Q * C          C * Q       (TRANS = 'N')
 *     Q^T * C        C * Q^T     (TRANS = 'T')
 *
 * where Q is a real orthogonal matrix of order nq (nq = m if SIDE = 'L',
 * nq = n if SIDE = 'R'). Q is defined as the product of IHI-ILO elementary
 * reflectors, as returned by DGEHRD:
 *
 *     Q = H(ilo) H(ilo+1) ... H(ihi-1)
 *
 * @param[in] side    'L' to apply Q or Q^T from the Left;
 *                    'R' to apply Q or Q^T from the Right.
 * @param[in] trans   'N' for no transpose; 'T' for transpose.
 * @param[in] m       Number of rows of C. m >= 0.
 * @param[in] n       Number of columns of C. n >= 0.
 * @param[in] ilo     Index ilo from DGEHRD (0-based).
 * @param[in] ihi     Index ihi from DGEHRD (0-based).
 * @param[in] A       Array containing the elementary reflectors from DGEHRD.
 * @param[in] lda     Leading dimension of A.
 * @param[in] tau     Array of scalar factors from DGEHRD.
 * @param[in,out] C   On entry, the M-by-N matrix C.
 *                    On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 * @param[in] ldc     Leading dimension of C. ldc >= max(1, m).
 * @param[out] work   Workspace array, dimension (max(1, lwork)).
 * @param[in] lwork   Dimension of work array.
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
SEMICOLON_API void dormhr(const char* side, const char* trans,
                          const int m, const int n,
                          const int ilo, const int ihi,
                          const f64* A, const int lda,
                          const f64* tau,
                          f64* C, const int ldc,
                          f64* work, const int lwork, int* info)
{
    int left, lquery;
    int i1, i2, mi, nb, nh, ni, nq, nw, lwkopt;

    *info = 0;
    nh = ihi - ilo;
    left = (side[0] == 'L' || side[0] == 'l');
    lquery = (lwork == -1);

    /* nq is the order of Q and nw is the minimum dimension of work */
    if (left) {
        nq = m;
        nw = n > 1 ? n : 1;
    } else {
        nq = n;
        nw = m > 1 ? m : 1;
    }

    /* Test the input arguments */
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 0 || ilo > (1 > nq ? 1 : nq) - 1) {
        *info = -5;
    } else if (ihi < (ilo < nq - 1 ? ilo : nq - 1) || ihi > nq - 1) {
        *info = -6;
    } else if (lda < (1 > nq ? 1 : nq)) {
        *info = -8;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -11;
    } else if (lwork < nw && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        /* Get optimal block size */
        nb = lapack_get_nb("ORMQR");
        lwkopt = nw * nb;
        work[0] = (f64)lwkopt;
    }

    if (*info != 0) {
        xerbla("DORMHR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nh == 0) {
        work[0] = 1.0;
        return;
    }

    if (left) {
        mi = nh;
        ni = n;
        i1 = ilo + 1;
        i2 = 0;
    } else {
        mi = m;
        ni = nh;
        i1 = 0;
        i2 = ilo + 1;
    }

    int iinfo;
    dormqr(side, trans, mi, ni, nh, &A[(ilo + 1) + ilo * lda], lda,
           &tau[ilo], &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);

    work[0] = (f64)lwkopt;
}
