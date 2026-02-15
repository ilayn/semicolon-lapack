/**
 * @file chetrd.c
 * @brief CHETRD reduces a complex Hermitian matrix to real symmetric
 *        tridiagonal form by a unitary similarity transformation.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"
#include "../include/lapack_tuning.h"

/**
 * CHETRD reduces a complex Hermitian matrix A to real symmetric
 * tridiagonal form T by a unitary similarity transformation:
 * Q**H * A * Q = T.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the Hermitian matrix A is stored:
 *                       = 'U': Upper triangle of A is stored;
 *                       = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the Hermitian matrix A. If uplo = 'U', the
 *                       leading n-by-n upper triangular part of A contains the
 *                       upper triangular part of the matrix A, and the strictly
 *                       lower triangular part of A is not referenced. If
 *                       uplo = 'L', the leading n-by-n lower triangular part of
 *                       A contains the lower triangular part of the matrix A,
 *                       and the strictly upper triangular part of A is not
 *                       referenced.
 *                       On exit, if uplo = 'U', the diagonal and first
 *                       superdiagonal of A are overwritten by the corresponding
 *                       elements of the tridiagonal matrix T, and the elements
 *                       above the first superdiagonal, with the array tau,
 *                       represent the unitary matrix Q as a product of
 *                       elementary reflectors; if uplo = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T, and
 *                       the elements below the first subdiagonal, with the array
 *                       tau, represent the unitary matrix Q as a product of
 *                       elementary reflectors.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    D      Single precision array, dimension (n).
 *                       The diagonal elements of the tridiagonal matrix T:
 *                       D[i] = A[i + i*lda].
 * @param[out]    E      Single precision array, dimension (n-1).
 *                       The off-diagonal elements of the tridiagonal matrix T:
 *                       E[i] = A[i + (i+1)*lda] if uplo = 'U',
 *                       E[i] = A[(i+1) + i*lda] if uplo = 'L'.
 * @param[out]    tau    Complex*16 array, dimension (n-1).
 *                       The scalar factors of the elementary reflectors.
 * @param[out]    work   Complex*16 array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= 1.
 *                       For optimum performance lwork >= n*nb, where nb is the
 *                       optimal blocksize.
 *                       If lwork = -1, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the work
 *                       array, returns this value as the first entry of the work
 *                       array, and no error message related to lwork is issued
 *                       by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void chetrd(const char* uplo, const int n, c64* A, const int lda,
            f32* D, f32* E, c64* tau, c64* work,
            const int lwork, int* info)
{
    const f32 ONE = 1.0f;
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    int upper, lquery;
    int i, iinfo, iws, j, kk, ldwork = 1, lwkopt, nb, nbmin, nx;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < 1 && !lquery) {
        *info = -9;
    }

    if (*info == 0) {
        nb = lapack_get_nb("HETRD");
        lwkopt = (1 > n * nb) ? 1 : n * nb;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
    }

    if (*info != 0) {
        xerbla("CHETRD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    nx = n;
    iws = 1;
    if (nb > 1 && nb < n) {
        nx = lapack_get_nx("HETRD");
        if (nx > nb) {
            /* keep nx */
        } else {
            nx = nb;
        }
        if (nx < n) {
            ldwork = n;
            iws = ldwork * nb;
            if (lwork < iws) {
                nb = lwork / ldwork;
                if (nb < 1) nb = 1;
                nbmin = lapack_get_nbmin("HETRD");
                if (nb < nbmin) {
                    nx = n;
                }
            }
        } else {
            nx = n;
        }
    } else {
        nb = 1;
    }

    if (upper) {
        /* Reduce the upper triangle of A.
           Columns 0:kk-1 are handled by the unblocked method. */
        kk = n - ((n - nx + nb - 1) / nb) * nb;

        for (i = n - nb; i >= kk; i -= nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */
            clatrd(uplo, i + nb, nb, A, lda, E, tau, work, ldwork);

            /* Update the unreduced submatrix A(0:i-1,0:i-1), using an
               update of the form:  A := A - V*W**H - W*V**H */
            cblas_cher2k(CblasColMajor, CblasUpper, CblasNoTrans,
                         i, nb, &NEG_CONE, &A[i * lda], lda,
                         work, ldwork, ONE, A, lda);

            /* Copy superdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i + nb; j++) {
                A[(j - 1) + j * lda] = CMPLXF(E[j - 1], 0.0f);
                D[j] = crealf(A[j + j * lda]);
            }
        }

        /* Use unblocked code to reduce the last or only block */
        chetd2(uplo, kk, A, lda, D, E, tau, &iinfo);
    } else {
        /* Reduce the lower triangle of A */
        for (i = 0; i < n - nx; i += nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */
            clatrd(uplo, n - i, nb, &A[i + i * lda], lda, &E[i],
                   &tau[i], work, ldwork);

            /* Update the unreduced submatrix A(i+nb:n-1,i+nb:n-1), using
               an update of the form:  A := A - V*W**H - W*V**H */
            cblas_cher2k(CblasColMajor, CblasLower, CblasNoTrans,
                         n - i - nb, nb, &NEG_CONE,
                         &A[(i + nb) + i * lda], lda,
                         &work[nb], ldwork, ONE,
                         &A[(i + nb) + (i + nb) * lda], lda);

            /* Copy subdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i + nb; j++) {
                A[(j + 1) + j * lda] = CMPLXF(E[j], 0.0f);
                D[j] = crealf(A[j + j * lda]);
            }
        }

        /* Use unblocked code to reduce the last or only block */
        chetd2(uplo, n - i, &A[i + i * lda], lda, &D[i], &E[i],
               &tau[i], &iinfo);
    }

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
