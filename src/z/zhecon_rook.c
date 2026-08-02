/**
 * @file zhecon_rook.c
 * @brief ZHECON_ROOK estimates the reciprocal of the condition number of a Hermitian matrix using the factorization computed by ZHETRF_ROOK.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZHECON_ROOK estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian matrix A using the factorization
 * A = U*D*U**H or A = L*D*L**H computed by `zhetrf_rook`.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular, form is A = U*D*U**H
 *                      - `'L'`: Lower triangular, form is A = L*D*L**H
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in]     A     Double complex array of dimension `(lda,n)`.
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain U or L, as computed by `zhetrf_rook`.
 * @param[in]     lda   The leading dimension of A. `lda>=max(1,n)`.
 * @param[in]     ipiv  Integer array of dimension `n`. Details of the
 *                      interchanges and block structure of D determined by
 *                      `zhetrf_rook`.
 * @param[in]     anorm The 1-norm of the original matrix A.
 * @param[out]    rcond The reciprocal condition number, computed as
 *                      `rcond=1/(anorm*ainvnm)`, where `ainvnm` estimates
 *                      the 1-norm of `inv(A)`.
 * @param[out]    work  Double complex workspace of dimension `2*n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an
 *                           illegal value
 */
void zhecon_rook(
    const char* uplo,
    const INT n,
    const c128* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    const f64 anorm,
    f64* rcond,
    c128* restrict work,
    INT* info)
{
    INT upper;
    INT i, kase;
    f64 ainvnm;
    INT isave[3];
    INT dummy_info;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (anorm < 0.0) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("ZHECON_ROOK", -(*info));
        return;
    }

    *rcond = 0.0;
    if (n == 0) {
        *rcond = 1.0;
        return;
    } else if (anorm <= 0.0) {
        return;
    }

    if (upper) {

        for (i = n - 1; i >= 0; i--) {
            if (ipiv[i] >= 0 && A[i + i * lda] == CMPLX(0.0, 0.0)) {
                return;
            }
        }

    } else {

        for (i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && A[i + i * lda] == CMPLX(0.0, 0.0)) {
                return;
            }
        }
    }

    kase = 0;
    for (;;) {
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase == 0) {
            break;
        }

        zhetrs_rook(uplo, n, 1, A, lda, ipiv, work, n, &dummy_info);
    }

    if (ainvnm != 0.0) {
        *rcond = (1.0 / ainvnm) / anorm;
    }
}
