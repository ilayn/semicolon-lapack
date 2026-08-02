/**
 * @file zhecon_3.c
 * @brief ZHECON_3 estimates the reciprocal of the condition number of a Hermitian matrix using the factorization computed by ZHETRF_RK or ZHETRF_BK.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZHECON_3 estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian matrix A using the factorization
 * computed by `zhetrf_rk` or `zhetrf_bk`:
 *
 *    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate transpose of U (or L), P is a
 * permutation matrix, P**T is the transpose of P, and D is Hermitian
 * and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 * This routine uses BLAS3 solver ZHETRS_3.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular, form is A = P*U*D*(U**H)*(P**T)
 *                      - `'L'`: Lower triangular, form is A = P*L*D*(L**H)*(P**T)
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in]     A     Array of dimension `(lda,n)`.
 *                      Diagonal of the block diagonal matrix D and factors
 *                      U or L as computed by `zhetrf_rk` and `zhetrf_bk`:
 *                      - Only diagonal elements of the Hermitian block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D should be provided on entry in
 *                        array `E`), and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in]     E     Array of dimension `n`.
 *                      On entry, contains the superdiagonal (or
 *                      subdiagonal) elements of the Hermitian block
 *                      diagonal matrix D with 1-by-1 or 2-by-2 diagonal
 *                      blocks, where if `uplo='U'`: `E[i]=D(i-1,i)`,
 *                      `i=1:n-1`, `E[0]` not referenced; if `uplo='L'`:
 *                      `E[i]=D(i+1,i)`, `i=0:n-2`, `E[n-1]` not referenced.
 *                      For a 1-by-1 diagonal block `D(k)`, the element
 *                      `E[k]` is not referenced in both `uplo='U'` or
 *                      `uplo='L'` cases.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      Details of the interchanges and the block structure
 *                      of D as determined by `zhetrf_rk` or `zhetrf_bk`.
 * @param[in]     anorm The 1-norm of the original matrix A.
 * @param[out]    rcond The reciprocal of the condition number of the matrix
 *                      A, computed as `rcond=1/(anorm*ainvnm)`, where
 *                      `ainvnm` is an estimate of the 1-norm of `inv(A)`
 *                      computed in this routine.
 * @param[out]    work  Array of dimension `2*n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void zhecon_3(
    const char* uplo,
    const INT n,
    const c128* restrict A,
    const INT lda,
    const c128* restrict E,
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
        *info = -7;
    }
    if (*info != 0) {
        xerbla("ZHECON_3", -(*info));
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

        zhetrs_3(uplo, n, 1, A, lda, E, ipiv, work, n, &dummy_info);
    }

    if (ainvnm != 0.0) {
        *rcond = (1.0 / ainvnm) / anorm;
    }
}
