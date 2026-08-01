/**
 * @file ssytri_3.c
 * @brief SSYTRI_3 computes the inverse of a real symmetric indefinite matrix using the factorization computed by SSYTRF_RK or DSYTRF_BK.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYTRI_3 computes the inverse of a real symmetric indefinite
 * matrix A using the factorization computed by SSYTRF_RK or SSYTRF_BK:
 *
 *     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * SSYTRI_3 sets the leading dimension of the workspace before calling
 * SSYTRI_3X that actually computes the inverse. This is the blocked
 * version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangle of A is stored
 *                      - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, diagonal of the block diagonal matrix D
 *                      and factors U or L as computed by `ssytrf_rk` or
 *                      `ssytrf_bk`:
 *
 *                      - Only diagonal elements of the symmetric block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D should be provided on entry in
 *                        array `E`), and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 *
 *                      On exit, if `info=0`, the symmetric inverse of the
 *                      original matrix.
 *                      If `uplo='U'`: the upper triangular part of the
 *                      inverse is formed and the part of A below the
 *                      diagonal is not referenced;
 *                      If `uplo='L'`: the lower triangular part of the
 *                      inverse is formed and the part of A above the
 *                      diagonal is not referenced.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in]     E     Array of dimension `n`.
 *                      On entry, contains the superdiagonal (or
 *                      subdiagonal) elements of the symmetric block
 *                      diagonal matrix D with 1-by-1 or 2-by-2 diagonal
 *                      blocks, where if `uplo='U'`: `E[i]=D(i-1,i)`,
 *                      `i=1:n-1`, `E[0]` not referenced; if `uplo='L'`:
 *                      `E[i]=D(i+1,i)`, `i=0:n-2`, `E[n-1]` not referenced.
 *
 *                      For a 1-by-1 diagonal block `D(k)`, the element
 *                      `E[k]` is not referenced in both `uplo='U'` or
 *                      `uplo='L'` cases.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      Details of the interchanges and the block structure
 *                      of D as determined by `ssytrf_rk` or `ssytrf_bk`.
 * @param[out]    work  Array of dimension `max(1,lwork)`.
 *                      On exit, if `info=0`, `work[0]` returns the optimal
 *                      `lwork`.
 * @param[in]     lwork The length of `work`.
 *                      If `n=0`, `lwork>=1`, else `lwork>=(n+nb+1)*(nb+3)`.
 *                      If `lwork=-1`, then a workspace query is assumed; the
 *                      routine only calculates the optimal size of the
 *                      `work` array, returns this value as the first entry
 *                      of the `work` array, and no error message related to
 *                      `lwork` is issued.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, `D(i,i)=0`; the matrix is singular
 *                           and its inverse could not be computed.
 */
void ssytri_3(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    const f32* restrict E,
    const INT* restrict ipiv,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    INT upper, lquery;
    INT lwkopt, nb;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (n == 0) {
        lwkopt = 1;
    } else {
        nb = 1;
        if (nb < 1) {
            nb = 1;
        }
        lwkopt = (n + nb + 1) * (nb + 3);
    }
    work[0] = (f32)lwkopt;

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < lwkopt && !lquery) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("SSYTRI_3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    ssytri_3x(uplo, n, A, lda, E, ipiv, work, nb, info);

    work[0] = (f32)lwkopt;
}
