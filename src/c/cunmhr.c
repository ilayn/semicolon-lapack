/**
 * @file cunmhr.c
 * @brief CUNMHR overwrites matrix C with Q*C or C*Q or their conjugate
 *        transposes, where Q is from the Hessenberg reduction produced by
 *        CGEHRD.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CUNMHR overwrites the general complex M-by-N matrix C with
 *
 *     SIDE = 'L'     SIDE = 'R'
 *     Q * C          C * Q       (TRANS = 'N')
 *     Q^H * C        C * Q^H    (TRANS = 'C')
 *
 * where Q is a complex unitary matrix of order nq (nq = m if SIDE = 'L',
 * nq = n if SIDE = 'R'). Q is defined as the product of IHI-ILO elementary
 * reflectors, as returned by CGEHRD:
 *
 *     Q = H(ilo) H(ilo+1) ... H(ihi-1)
 *
 * @param[in] side    'L' to apply Q or Q^H from the Left;
 *                    'R' to apply Q or Q^H from the Right.
 * @param[in] trans   'N' for no transpose; 'C' for conjugate transpose.
 * @param[in] m       Number of rows of C. m >= 0.
 * @param[in] n       Number of columns of C. n >= 0.
 * @param[in] ilo     Index ilo from CGEHRD (0-based).
 * @param[in] ihi     Index ihi from CGEHRD (0-based).
 * @param[in] A       Array containing the elementary reflectors from CGEHRD.
 * @param[in] lda     Leading dimension of A.
 * @param[in] tau     Array of scalar factors from CGEHRD.
 * @param[in,out] C   On entry, the M-by-N matrix C.
 *                    On exit, C is overwritten by Q*C or Q^H*C or C*Q^H or C*Q.
 * @param[in] ldc     Leading dimension of C. ldc >= max(1, m).
 * @param[out] work   Workspace array, dimension (max(1, lwork)).
 * @param[in] lwork   Dimension of work array.
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cunmhr(const char* side, const char* trans,
            const INT m, const INT n,
            const INT ilo, const INT ihi,
            const c64* A, const INT lda,
            const c64* tau,
            c64* C, const INT ldc,
            c64* work, const INT lwork, INT* info)
{
    INT left, lquery;
    INT i1, i2, mi, nb, nh, ni, nq, nw, lwkopt;

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
               !(trans[0] == 'C' || trans[0] == 'c')) {
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
        nb = lapack_get_nb("ORMQR");
        lwkopt = nw * nb;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
    }

    if (*info != 0) {
        xerbla("CUNMHR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nh == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
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

    INT iinfo;
    cunmqr(side, trans, mi, ni, nh, &A[(ilo + 1) + ilo * lda], lda,
           &tau[ilo], &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
