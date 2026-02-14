/**
 * @file zhecon_3.c
 * @brief ZHECON_3 estimates the reciprocal of the condition number of a Hermitian matrix using the factorization computed by ZHETRF_RK or ZHETRF_BK.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZHECON_3 estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian matrix A using the factorization
 * computed by ZHETRF_RK or ZHETRF_BK:
 *
 *    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is Hermitian and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 * This routine uses BLAS3 solver ZHETRS_3.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix:
 *          = 'U':  Upper triangular, form is A = P*U*D*(U**H)*(P**T);
 *          = 'L':  Lower triangular, form is A = P*L*D*(L**H)*(P**T).
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] A
 *          Double complex array, dimension (lda, n).
 *          Diagonal of the block diagonal matrix D and factors U or L
 *          as computed by ZHETRF_RK and ZHETRF_BK:
 *            a) ONLY diagonal elements of the Hermitian block diagonal
 *               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
 *               (superdiagonal (or subdiagonal) elements of D
 *                should be provided on entry in array E), and
 *            b) If UPLO = 'U': factor U in the superdiagonal part of A.
 *               If UPLO = 'L': factor L in the subdiagonal part of A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Double complex array, dimension (n).
 *          Contains the superdiagonal (or subdiagonal) elements of the
 *          Hermitian block diagonal matrix D with 1-by-1 or 2-by-2
 *          diagonal blocks.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by ZHETRF_RK or ZHETRF_BK.
 *
 * @param[in] anorm
 *          The 1-norm of the original matrix A.
 *
 * @param[out] rcond
 *          The reciprocal of the condition number of the matrix A,
 *          computed as rcond = 1/(anorm * ainvnm).
 *
 * @param[out] work
 *          Double complex array, dimension (2*n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zhecon_3(
    const char* uplo,
    const int n,
    const c128* restrict A,
    const int lda,
    const c128* restrict E,
    const int* restrict ipiv,
    const f64 anorm,
    f64* rcond,
    c128* restrict work,
    int* info)
{
    int upper;
    int i, kase;
    f64 ainvnm;
    int isave[3];
    int dummy_info;

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
