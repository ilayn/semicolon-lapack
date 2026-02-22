/**
 * @file cgeqrfp.c
 * @brief CGEQRFP computes a QR factorization of a general M-by-N matrix
 *        with non-negative diagonal elements of R, using a blocked algorithm.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGEQRFP computes a QR factorization of a complex M-by-N matrix A:
 *
 *    A = Q * ( R ),
 *            ( 0 )
 *
 * where Q is a M-by-M unitary matrix, R is an upper-triangular N-by-N
 * matrix with nonnegative diagonal entries, and 0 is a (M-N)-by-N zero
 * matrix if M > N.
 *
 * This is the blocked Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in,out] A      On entry, the m-by-n matrix A.
 *                       On exit, the elements on and above the diagonal contain
 *                       the min(m,n)-by-n upper trapezoidal matrix R (R is
 *                       upper triangular if m >= n). The diagonal entries of R
 *                       are real and nonnegative. The elements below the
 *                       diagonal, with TAU, represent the unitary matrix Q as
 *                       a product of min(m,n) elementary reflectors.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau    Array of dimension min(m, n). The scalar factors
 *                       of the elementary reflectors.
 * @param[out]    work   Workspace, dimension (max(1, lwork)).
 *                       On exit, work[0] contains the optimal lwork.
 * @param[in]     lwork  Dimension of work.
 *                       lwork >= 1 if min(m,n) = 0, lwork >= n otherwise.
 *                       For optimal performance, lwork >= n*nb.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: success; < 0: -i means i-th argument was illegal.
 */
void cgeqrfp(const INT m, const INT n,
             c64* restrict A, const INT lda,
             c64* restrict tau,
             c64* restrict work, const INT lwork,
             INT* info)
{
    INT k, nb, nbmin, nx, iws, ldwork;
    INT i, ib, iinfo;
    INT lquery;

    /* Parameter validation */
    *info = 0;
    k = m < n ? m : n;
    lquery = (lwork == -1);

    /* Determine nb early for optimal workspace computation */
    nb = lapack_get_nb("GEQRF");

    if (k == 0) {
        iws = 1;
    } else {
        iws = n * nb;
    }
    work[0] = (c64)iws;

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (!lquery && lwork < (k == 0 ? 1 : n)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("CGEQRFP", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (k == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    nbmin = 2;
    nx = 0;
    iws = n;
    ldwork = n;

    if (nb > 1 && nb < k) {
        /* Determine crossover point */
        nx = lapack_get_nx("GEQRF");
        if (nx < k) {
            /* Determine if workspace is large enough for blocked code */
            iws = n * nb;
            if (lwork < iws) {
                /* Not enough workspace: reduce nb */
                nb = lwork / n;
                nbmin = lapack_get_nbmin("GEQRF");
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code */
        for (i = 0; i < k - nx; i += nb) {
            ib = (k - i) < nb ? (k - i) : nb;

            /* Compute the QR factorization of the current block
             * A(i:m-1, i:i+ib-1) */
            cgeqr2p(m - i, ib, &A[i + i * lda], lda, &tau[i], work, &iinfo);

            if (i + ib < n) {
                /* Form the triangular factor of the block reflector
                 * H = H(i) H(i+1) . . . H(i+ib-1) */
                clarft("F", "C", m - i, ib,
                       &A[i + i * lda], lda, &tau[i], work, ldwork);

                /* Apply H**H to A(i:m-1, i+ib:n-1) from the left */
                clarfb("L", "C", "F", "C",
                       m - i, n - i - ib, ib,
                       &A[i + i * lda], lda,
                       work, ldwork,
                       &A[i + (i + ib) * lda], lda,
                       &work[ib], ldwork);
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code for the remaining columns */
    if (i < k) {
        cgeqr2p(m - i, n - i, &A[i + i * lda], lda, &tau[i], work, &iinfo);
    }

    work[0] = (c64)iws;
}
