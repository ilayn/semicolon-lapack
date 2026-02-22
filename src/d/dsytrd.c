/**
 * @file dsytrd.c
 * @brief DSYTRD reduces a real symmetric matrix to real symmetric
 *        tridiagonal form by an orthogonal similarity transformation.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"

/**
 * DSYTRD reduces a real symmetric matrix A to real symmetric
 * tridiagonal form T by an orthogonal similarity transformation:
 * Q**T * A * Q = T.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored:
 *                       = 'U': Upper triangle of A is stored;
 *                       = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the symmetric matrix A. If uplo = 'U', the
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
 *                       represent the orthogonal matrix Q as a product of
 *                       elementary reflectors; if uplo = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T, and
 *                       the elements below the first subdiagonal, with the array
 *                       tau, represent the orthogonal matrix Q as a product of
 *                       elementary reflectors.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    D      Double precision array, dimension (n).
 *                       The diagonal elements of the tridiagonal matrix T:
 *                       D[i] = A[i + i*lda].
 * @param[out]    E      Double precision array, dimension (n-1).
 *                       The off-diagonal elements of the tridiagonal matrix T:
 *                       E[i] = A[i + (i+1)*lda] if uplo = 'U',
 *                       E[i] = A[(i+1) + i*lda] if uplo = 'L'.
 * @param[out]    tau    Double precision array, dimension (n-1).
 *                       The scalar factors of the elementary reflectors.
 * @param[out]    work   Double precision array, dimension (max(1, lwork)).
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
void dsytrd(const char* uplo, const INT n, f64* A, const INT lda,
            f64* D, f64* E, f64* tau, f64* work,
            const INT lwork, INT* info)
{
    const f64 ONE = 1.0;

    INT upper, lquery;
    INT i, iinfo, iws, j, kk, ldwork = 1, lwkopt, nb, nbmin, nx;

    /* Test the input parameters */
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
        /* Determine the block size. */
        nb = lapack_get_nb("SYTRD");
        lwkopt = (1 > n * nb) ? 1 : n * nb;
        work[0] = (f64)lwkopt;
    }

    if (*info != 0) {
        xerbla("DSYTRD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        work[0] = 1.0;
        return;
    }

    nx = n;
    if (nb > 1 && nb < n) {
        /* Determine when to cross over from blocked to unblocked code
           (last block is always handled by unblocked code). */
        nx = lapack_get_nx("SYTRD");
        if (nx > nb) {
            /* keep nx */
        } else {
            nx = nb;
        }
        if (nx < n) {
            /* Determine if workspace is large enough for blocked code. */
            ldwork = n;
            iws = ldwork * nb;
            if (lwork < iws) {
                /* Not enough workspace to use optimal NB: determine the
                   minimum value of NB, and reduce NB or force use of
                   unblocked code by setting NX = N. */
                nb = lwork / ldwork;
                if (nb < 1) nb = 1;
                nbmin = lapack_get_nbmin("SYTRD");
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

        /* kk = n - ((n - nx + nb - 1) / nb) * nb  (Fortran 1-based: KK)
           In 0-based: same formula gives the number of columns handled
           by the unblocked method at the beginning. */
        kk = n - ((n - nx + nb - 1) / nb) * nb;

        /* Fortran loop: DO I = N-NB+1, KK+1, -NB (1-based)
           In 0-based: i goes from n-nb down to kk, step -nb */
        for (i = n - nb; i >= kk; i -= nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */

            /* dlatrd(uplo, n, nb, A, lda, E, tau, W, ldw)
               Fortran: DLATRD(UPLO, I+NB-1, NB, A, LDA, E, TAU, WORK, LDWORK)
               i+nb-1 in 1-based = (i+1)+(nb-1)-1 = i+nb-1 in 1-based = i+nb in 0-based count
               Actually: Fortran I is 1-based, our i is 0-based.
               Fortran I corresponds to our i+1 (since Fortran DO starts at N-NB+1).
               Fortran I+NB-1 = (our i+1)+NB-1 = i+nb in 0-based indexing of size.
               So the size argument is i+nb. */
            dlatrd(uplo, i + nb, nb, A, lda, E, tau, work, ldwork);

            /* Update the unreduced submatrix A(0:i-1, 0:i-1), using an
               update of the form: A := A - V*W**T - W*V**T
               Fortran: DSYR2K(UPLO, 'N', I-1, NB, -ONE, A(1,I), LDA, WORK, LDWORK, ONE, A, LDA)
               Fortran I-1 = number of rows/cols in the submatrix = our i
               Fortran A(1,I) = A[0 + i*lda] in 0-based
               Fortran WORK (start) = work[0] */
            cblas_dsyr2k(CblasColMajor, CblasUpper, CblasNoTrans,
                         i, nb, -ONE, &A[i * lda], lda,
                         work, ldwork, ONE, A, lda);

            /* Copy superdiagonal elements back into A, and diagonal
               elements into D */
            /* Fortran loop: DO J = I, I+NB-1 (1-based)
               0-based: j from i to i+nb-1 */
            for (j = i; j < i + nb; j++) {
                /* Fortran: A(J-1, J) = E(J-1)  (1-based)
                   0-based: A[(j-1) + j*lda] = E[j-1] */
                A[(j - 1) + j * lda] = E[j - 1];
                /* Fortran: D(J) = A(J, J)  (1-based)
                   0-based: D[j] = A[j + j*lda] */
                D[j] = A[j + j * lda];
            }
        }

        /* Use unblocked code to reduce the last or only block */
        /* Fortran: DSYTD2(UPLO, KK, A, LDA, D, E, TAU, IINFO)
           kk is the size of the remaining block (same in both bases) */
        dsytd2(uplo, kk, A, lda, D, E, tau, &iinfo);
    } else {
        /* Reduce the lower triangle of A */

        /* Fortran loop: DO I = 1, N-NX, NB (1-based)
           0-based: i from 0, while i < n-nx, step nb */
        for (i = 0; i < n - nx; i += nb) {
            /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */

            /* Fortran: DLATRD(UPLO, N-I+1, NB, A(I,I), LDA, E(I), TAU(I), WORK, LDWORK)
               N-I+1 (1-based) with I=our_i+1 gives N-(our_i+1)+1 = n-i
               A(I,I) 1-based = A[i + i*lda] 0-based
               E(I) 1-based = E[i] 0-based
               TAU(I) 1-based = tau[i] 0-based */
            dlatrd(uplo, n - i, nb, &A[i + i * lda], lda, &E[i],
                   &tau[i], work, ldwork);

            /* Update the unreduced submatrix A(i+nb:n-1, i+nb:n-1), using
               an update of the form: A := A - V*W**T - W*V**T
               Fortran: DSYR2K(UPLO, 'N', N-I-NB+1, NB, -ONE,
                               A(I+NB, I), LDA, WORK(NB+1), LDWORK, ONE,
                               A(I+NB, I+NB), LDA)
               N-I-NB+1 (1-based I) = n-(i+1)-nb+1 = n-i-nb
               A(I+NB, I) 1-based = A[(i+nb-1+1-1) + (i+1-1)*lda] = wait...
               Actually Fortran I is 1-based. Our i is 0-based: Fortran I = our_i + 1.
               N - I - NB + 1 = n - (our_i+1) - nb + 1 = n - our_i - nb
               A(I+NB, I) in 1-based = A[(I+NB-1) + (I-1)*lda] 0-based
                 = A[(our_i+1+nb-1) + (our_i+1-1)*lda] = A[(our_i+nb) + our_i*lda]
               WORK(NB+1) in 1-based = work[nb] in 0-based
               A(I+NB, I+NB) in 1-based = A[(our_i+nb) + (our_i+nb)*lda] */
            cblas_dsyr2k(CblasColMajor, CblasLower, CblasNoTrans,
                         n - i - nb, nb, -ONE,
                         &A[(i + nb) + i * lda], lda,
                         &work[nb], ldwork, ONE,
                         &A[(i + nb) + (i + nb) * lda], lda);

            /* Copy subdiagonal elements back into A, and diagonal
               elements into D */
            /* Fortran loop: DO J = I, I+NB-1 (1-based)
               With Fortran I = our_i+1: j from our_i+1 to our_i+nb (1-based)
               0-based: j from our_i to our_i+nb-1 */
            for (j = i; j < i + nb; j++) {
                /* Fortran: A(J+1, J) = E(J) (1-based)
                   0-based: A[(j+1-1) + (j-1)*lda] = ... wait, need careful conversion.
                   Fortran J is 1-based here (from I to I+NB-1 where I=our_i+1).
                   Let j_f = Fortran J = our_j + 1.
                   A(J_f+1, J_f) = A[J_f + (J_f-1)*lda] 0-based = A[(our_j+1) + our_j*lda]
                   E(J_f) = E[our_j] 0-based
                   D(J_f) = D[our_j]
                   A(J_f, J_f) = A[our_j + our_j*lda] */
                A[(j + 1) + j * lda] = E[j];
                D[j] = A[j + j * lda];
            }
        }

        /* Use unblocked code to reduce the last or only block */
        /* Fortran: DSYTD2(UPLO, N-I+1, A(I,I), LDA, D(I), E(I), TAU(I), IINFO)
           At loop exit, Fortran I = last value + NB. Our i is 0-based.
           N-I+1 (1-based) = n - (our_i+1) + 1 = n - our_i
           A(I,I) = A[i + i*lda], E(I)=E[i], D(I)=D[i], TAU(I)=tau[i] */
        dsytd2(uplo, n - i, &A[i + i * lda], lda, &D[i], &E[i],
               &tau[i], &iinfo);
    }

    work[0] = (f64)lwkopt;
}
