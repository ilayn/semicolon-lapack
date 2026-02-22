/**
 * @file sorghr.c
 * @brief SORGHR generates the orthogonal matrix Q from SGEHRD.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SORGHR generates a real orthogonal matrix Q which is defined as the
 * product of ihi-ilo elementary reflectors of order N, as returned by
 * SGEHRD:
 *
 * Q = H(ilo) H(ilo+1) . . . H(ihi-1).
 *
 * @param[in]     n      The order of the matrix Q. n >= 0.
 * @param[in]     ilo    ilo and ihi must have the same values as in the
 *                       previous call of SGEHRD. Q is equal to the unit
 *                       matrix except in the submatrix Q(ilo+1:ihi, ilo+1:ihi).
 *                       0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 *                       (0-based indexing)
 * @param[in]     ihi    See ilo. (0-based)
 * @param[in,out] A      On entry, the vectors which define the elementary
 *                       reflectors, as returned by SGEHRD.
 *                       On exit, the n-by-n orthogonal matrix Q. Dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in]     tau    tau(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by SGEHRD. Dimension (n-1).
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of work. lwork >= ihi-ilo.
 *                       For optimum performance lwork >= (ihi-ilo)*NB.
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sorghr(const INT n, const INT ilo, const INT ihi,
            f32* A, const INT lda, const f32* tau,
            f32* work, const INT lwork, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lquery;
    INT i, iinfo, j, lwkopt, nb, nh;
    INT max_n_1 = (n > 1) ? n : 1;
    INT nh_max_1;

    /* Test the input parameters */
    *info = 0;
    nh = ihi - ilo;
    nh_max_1 = (nh > 1) ? nh : 1;
    lquery = (lwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (ilo < 0 || ilo > (n > 0 ? n - 1 : 0)) {
        *info = -2;
    } else if (n > 0 && (ihi < ilo || ihi > n - 1)) {
        *info = -3;
    } else if (n == 0 && ihi != -1) {
        *info = -3;
    } else if (lda < max_n_1) {
        *info = -5;
    } else if (lwork < nh_max_1 && !lquery) {
        *info = -8;
    }

    if (*info == 0) {
        nb = lapack_get_nb("ORGQR");
        lwkopt = nh_max_1 * nb;
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SORGHR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        work[0] = 1.0f;
        return;
    }

    /* Shift the vectors which define the elementary reflectors one
       column to the right, and set the first ilo+1 and the last n-ihi-1
       rows and columns to those of the unit matrix */
    for (j = ihi; j >= ilo + 1; j--) {
        for (i = 0; i < j; i++) {
            A[i + j * lda] = ZERO;
        }
        for (i = j + 1; i <= ihi; i++) {
            A[i + j * lda] = A[i + (j - 1) * lda];
        }
        for (i = ihi + 1; i < n; i++) {
            A[i + j * lda] = ZERO;
        }
    }
    for (j = 0; j <= ilo; j++) {
        for (i = 0; i < n; i++) {
            A[i + j * lda] = ZERO;
        }
        A[j + j * lda] = ONE;
    }
    for (j = ihi + 1; j < n; j++) {
        for (i = 0; i < n; i++) {
            A[i + j * lda] = ZERO;
        }
        A[j + j * lda] = ONE;
    }

    if (nh > 0) {
        /* Generate Q(ilo+1:ihi, ilo+1:ihi) */
        sorgqr(nh, nh, nh, &A[(ilo + 1) + (ilo + 1) * lda], lda,
               &tau[ilo], work, lwork, &iinfo);
    }
    work[0] = (f32)lwkopt;
}
