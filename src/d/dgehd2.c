/**
 * @file dgehd2.c
 * @brief DGEHD2 reduces a general square matrix to upper Hessenberg form
 *        using an unblocked algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DGEHD2 reduces a real general matrix A to upper Hessenberg form H by
 * an orthogonal similarity transformation:  Q**T * A * Q = H .
 *
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     ilo    It is assumed that A is already upper triangular in
 *                       rows and columns 0:ilo-1 and ihi+1:n-1. ilo and ihi
 *                       are normally set by a previous call to DGEBAL;
 *                       otherwise they should be set to 0 and n-1 respectively.
 *                       0 <= ilo <= ihi <= max(0,n-1). (0-based)
 * @param[in]     ihi    See ilo. (0-based)
 * @param[in,out] A      On entry, the n by n general matrix to be reduced.
 *                       On exit, the upper triangle and the first subdiagonal
 *                       of A are overwritten with the upper Hessenberg matrix H,
 *                       and the elements below the first subdiagonal, with the
 *                       array tau, represent the orthogonal matrix Q as a
 *                       product of elementary reflectors. Dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[out]    tau    The scalar factors of the elementary reflectors.
 *                       Dimension (n-1).
 * @param[out]    work   Workspace array, dimension (n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value.
 *
 * The matrix Q is represented as a product of (ihi-ilo) elementary
 * reflectors
 *
 *    Q = H(ilo) H(ilo+1) . . . H(ihi-1).
 *
 * Each H(i) has the form
 *
 *    H(i) = I - tau * v * v**T
 *
 * where tau is a real scalar, and v is a real vector with
 * v(0:i) = 0, v(i+1) = 1 and v(ihi+1:n-1) = 0; v(i+2:ihi) is stored on
 * exit in A(i+2:ihi,i), and tau in tau(i).
 */
void dgehd2(const int n, const int ilo, const int ihi,
            double* A, const int lda, double* tau,
            double* work, int* info)
{
    int i;
    int max_n_1 = (n > 1) ? n : 1;

    /* Test the input parameters */
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (ilo < 0 || ilo > (n > 0 ? n - 1 : 0)) {
        *info = -2;
    } else if (ihi < (ilo < n - 1 ? ilo : n - 1) || ihi > n - 1) {
        /* ihi < min(ilo, n-1) or ihi > n-1 is invalid
           For n=0, this check is skipped since ihi should be -1 */
        if (n > 0)
            *info = -3;
    } else if (lda < max_n_1) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("DGEHD2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 1)
        return;

    for (i = ilo; i <= ihi - 1; i++) {
        /* Compute elementary reflector H(i) to annihilate A(i+2:ihi,i) */
        int len = ihi - i;  /* length of reflector vector */
        int start = (i + 2 < n) ? (i + 2) : (n - 1);

        dlarfg(len, &A[(i + 1) + i * lda], &A[start + i * lda], 1, &tau[i]);

        /* Apply H(i) to A(0:ihi, i+1:ihi) from the right */
        dlarf1f("Right", ihi + 1, len, &A[(i + 1) + i * lda], 1, tau[i],
                &A[(i + 1) * lda], lda, work);

        /* Apply H(i) to A(i+1:ihi, i+1:n-1) from the left */
        dlarf1f("Left", len, n - i - 1, &A[(i + 1) + i * lda], 1, tau[i],
                &A[(i + 1) + (i + 1) * lda], lda, work);
    }
}
