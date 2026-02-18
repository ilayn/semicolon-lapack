/**
 * @file sgehrd.c
 * @brief SGEHRD reduces a general matrix to upper Hessenberg form.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SGEHRD reduces a real general matrix A to upper Hessenberg form H by
 * an orthogonal similarity transformation:  Q**T * A * Q = H .
 *
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     ilo    It is assumed that A is already upper triangular in
 *                       rows and columns 0:ilo-1 and ihi+1:n-1. ilo and ihi
 *                       are normally set by a previous call to SGEBAL;
 *                       otherwise they should be set to 0 and n-1 respectively.
 *                       0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 *                       (0-based indexing)
 * @param[in]     ihi    See ilo. (0-based)
 * @param[in,out] A      On entry, the n by n general matrix to be reduced.
 *                       On exit, the upper triangle and the first subdiagonal
 *                       of A are overwritten with the upper Hessenberg matrix H,
 *                       and the elements below the first subdiagonal, with the
 *                       array tau, represent the orthogonal matrix Q as a
 *                       product of elementary reflectors. Dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[out]    tau    The scalar factors of the elementary reflectors.
 *                       Dimension (n-1). Elements 0:ilo-1 and ihi:n-2 are set
 *                       to zero.
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The length of work. lwork >= max(1, n).
 *                       For good performance, lwork should generally be larger.
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgehrd(const int n, const int ilo, const int ihi,
            f32* A, const int lda, f32* tau,
            f32* work, const int lwork, int* info)
{
    const int NBMAX = 64;
    const int LDT = NBMAX + 1;
    const int TSIZE = LDT * NBMAX;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int lquery;
    int i, ib, iinfo, iwt, j, ldwork, lwkopt, nb, nbmin, nh, nx = 0;
    f32 ei;
    int max_n_1 = (n > 1) ? n : 1;

    /* Test the input parameters */
    *info = 0;
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
    } else if (lwork < (n > 1 ? n : 1) && !lquery) {
        *info = -8;
    }

    nh = ihi - ilo + 1;
    if (*info == 0) {
        /* Compute the workspace requirements */
        if (nh <= 1) {
            lwkopt = 1;
        } else {
            nb = lapack_get_nb("GEHRD");
            if (nb > NBMAX) nb = NBMAX;
            lwkopt = n * nb + TSIZE;
        }
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SGEHRD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Set elements 0:ilo-1 and ihi:n-2 of tau to zero */
    for (i = 0; i < ilo; i++) {
        tau[i] = ZERO;
    }
    for (i = (ihi > 0 ? ihi : 0); i < n - 1; i++) {
        tau[i] = ZERO;
    }

    /* Quick return if possible */
    if (nh <= 1) {
        work[0] = 1.0f;
        return;
    }

    /* Determine the block size */
    nb = lapack_get_nb("GEHRD");
    if (nb > NBMAX) nb = NBMAX;
    nbmin = 2;

    if (nb > 1 && nb < nh) {
        /* Determine when to cross over from blocked to unblocked code
           (last block is always handled by unblocked code) */
        nx = lapack_get_nx("GEHRD");
        if (nx < nb) nx = nb;

        if (nx < nh) {
            /* Determine if workspace is large enough for blocked code */
            if (lwork < lwkopt) {
                /* Not enough workspace to use optimal NB: determine the
                   minimum value of NB, and reduce NB or force use of
                   unblocked code */
                nbmin = lapack_get_nbmin("GEHRD");
                if (lwork >= (n * nbmin + TSIZE)) {
                    nb = (lwork - TSIZE) / n;
                } else {
                    nb = 1;
                }
            }
        }
    }
    ldwork = n;

    if (nb < nbmin || nb >= nh) {
        /* Use unblocked code below */
        i = ilo;
    } else {
        /* Use blocked code */
        iwt = n * nb;  /* offset to T matrix in work array */

        for (i = ilo; i <= ihi - 1 - nx; i += nb) {
            ib = (nb < ihi - i) ? nb : (ihi - i);

            /* Reduce columns i:i+ib-1 to Hessenberg form, returning the
               matrices V and T of the block reflector H = I - V*T*V**T
               which performs the reduction, and also the matrix Y = A*V*T */
            slahr2(ihi + 1, i, ib, &A[i * lda], lda, &tau[i],
                   &work[iwt], LDT, work, ldwork);

            /* Apply the block reflector H to A(0:ihi, i+ib:ihi) from the
               right, computing  A := A - Y * V**T. V(i+ib, ib-1) must be set
               to 1 */
            ei = A[(i + ib) + (i + ib - 1) * lda];
            A[(i + ib) + (i + ib - 1) * lda] = ONE;
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        ihi + 1, ihi - i - ib + 1, ib,
                        -ONE, work, ldwork, &A[(i + ib) + i * lda], lda,
                        ONE, &A[(i + ib) * lda], lda);
            A[(i + ib) + (i + ib - 1) * lda] = ei;

            /* Apply the block reflector H to A(0:i, i+1:i+ib-1) from the right */
            cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                        i + 1, ib - 1, ONE, &A[(i + 1) + i * lda], lda, work, ldwork);
            for (j = 0; j < ib - 1; j++) {
                cblas_saxpy(i + 1, -ONE, &work[ldwork * j], 1, &A[(i + j + 1) * lda], 1);
            }

            /* Apply the block reflector H to A(i+1:ihi, i+ib:n-1) from the left */
            slarfb("Left", "Transpose", "Forward", "Columnwise",
                   ihi - i, n - i - ib, ib, &A[(i + 1) + i * lda], lda,
                   &work[iwt], LDT, &A[(i + 1) + (i + ib) * lda], lda,
                   work, ldwork);
        }
    }

    /* Use unblocked code to reduce the rest of the matrix */
    sgehd2(n, i, ihi, A, lda, tau, work, &iinfo);

    work[0] = (f32)lwkopt;
}
