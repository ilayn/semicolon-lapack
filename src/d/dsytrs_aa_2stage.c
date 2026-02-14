/**
 * @file dsytrs_aa_2stage.c
 * @brief DSYTRS_AA_2STAGE solves a system of linear equations A*X = B using the factorization computed by DSYTRF_AA_2STAGE.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSYTRS_AA_2STAGE solves a system of linear equations A*X = B with a real
 * symmetric matrix A using the factorization A = U**T*T*U or
 * A = L*T*L**T computed by DSYTRF_AA_2STAGE.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U**T*T*U;
 *          = 'L':  Lower triangular, form is A = L*T*L**T.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, n).
 *          Details of factors computed by DSYTRF_AA_2STAGE.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] TB
 *          Double precision array, dimension (ltb).
 *          Details of factors computed by DSYTRF_AA_2STAGE.
 *
 * @param[in] ltb
 *          The size of the array TB. ltb >= 4*n.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges as computed by DSYTRF_AA_2STAGE.
 *
 * @param[in] ipiv2
 *          Integer array, dimension (n).
 *          Details of the interchanges as computed by DSYTRF_AA_2STAGE.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void dsytrs_aa_2stage(
    const char* uplo,
    const int n,
    const int nrhs,
    const f64* restrict A,
    const int lda,
    f64* restrict TB,
    const int ltb,
    const int* restrict ipiv,
    const int* restrict ipiv2,
    f64* restrict B,
    const int ldb,
    int* info)
{
    int ldtb, nb;
    int upper;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ltb < 4 * n) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -11;
    }

    if (*info != 0) {
        xerbla("DSYTRS_AA_2STAGE", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    nb = (int)TB[0];
    ldtb = ltb / n;

    if (upper) {

        if (n > nb) {

            dlaswp(nrhs, B, ldb, nb, n - 1, ipiv, 1);

            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                        n - nb, nrhs, 1.0, &A[0 + nb * lda], lda, &B[nb + 0 * ldb], ldb);

        }

        dgbtrs("N", n, nb, nb, nrhs, TB, ldtb, ipiv2, B, ldb, info);

        if (n > nb) {

            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                        n - nb, nrhs, 1.0, &A[0 + nb * lda], lda, &B[nb + 0 * ldb], ldb);

            dlaswp(nrhs, B, ldb, nb, n - 1, ipiv, -1);

        }

    } else {

        if (n > nb) {

            dlaswp(nrhs, B, ldb, nb, n - 1, ipiv, 1);

            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        n - nb, nrhs, 1.0, &A[nb + 0 * lda], lda, &B[nb + 0 * ldb], ldb);

        }

        dgbtrs("N", n, nb, nb, nrhs, TB, ldtb, ipiv2, B, ldb, info);

        if (n > nb) {

            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                        n - nb, nrhs, 1.0, &A[nb + 0 * lda], lda, &B[nb + 0 * ldb], ldb);

            dlaswp(nrhs, B, ldb, nb, n - 1, ipiv, -1);

        }
    }
}
