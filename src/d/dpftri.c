/**
 * @file dpftri.c
 * @brief DPFTRI computes the inverse of a symmetric positive definite matrix in RFP format.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DPFTRI computes the inverse of a (real) symmetric positive definite
 * matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
 * computed by DPFTRF.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'T':  The Transpose TRANSR of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (n*(n+1)/2).
 *          On entry, the Cholesky factor in RFP format.
 *          On exit, the inverse of the original matrix.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the (i,i) element of the factor is zero
 */
void dpftri(
    const char* transr,
    const char* uplo,
    const int n,
    f64* restrict A,
    int* info)
{
    int lower, nisodd, normaltransr;
    int n1, n2, k;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("DPFTRI", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    dtftri(transr, uplo, "N", n, A, info);
    if (*info > 0) {
        return;
    }

    if (n % 2 == 0) {
        k = n / 2;
        nisodd = 0;
    } else {
        nisodd = 1;
    }

    if (lower) {
        n2 = n / 2;
        n1 = n - n2;
    } else {
        n1 = n / 2;
        n2 = n - n1;
    }

    if (nisodd) {

        if (normaltransr) {

            if (lower) {

                dlauum("L", n1, A, n, info);
                cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                            n1, n2, 1.0, A + n1, n, 1.0, A, n);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, 1.0, A + n, n, A + n1, n);
                dlauum("U", n2, A + n, n, info);

            } else {

                dlauum("L", n1, A + n2, n, info);
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            n1, n2, 1.0, A, n, 1.0, A + n2, n);
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasNonUnit,
                            n1, n2, 1.0, A + n1, n, A, n);
                dlauum("U", n2, A + n1, n, info);

            }

        } else {

            if (lower) {

                dlauum("U", n1, A, n1, info);
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n1, n2, 1.0, A + n1 * n1, n1, 1.0, A, n1);
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, 1.0, A + 1, n1, A + n1 * n1, n1);
                dlauum("L", n2, A + 1, n1, info);

            } else {

                dlauum("U", n1, A + n2 * n2, n2, info);
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                            n1, n2, 1.0, A, n2, 1.0, A + n2 * n2, n2);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasTrans, CblasNonUnit,
                            n2, n1, 1.0, A + n1 * n2, n2, A, n2);
                dlauum("L", n2, A + n1 * n2, n2, info);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                dlauum("L", k, A + 1, n + 1, info);
                cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                            k, k, 1.0, A + k + 1, n + 1, 1.0, A + 1, n + 1);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, 1.0, A, n + 1, A + k + 1, n + 1);
                dlauum("U", k, A, n + 1, info);

            } else {

                dlauum("L", k, A + k + 1, n + 1, info);
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, 1.0, A, n + 1, 1.0, A + k + 1, n + 1);
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasNonUnit,
                            k, k, 1.0, A + k, n + 1, A, n + 1);
                dlauum("U", k, A + k, n + 1, info);

            }

        } else {

            if (lower) {

                dlauum("U", k, A + k, k, info);
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, 1.0, A + k * (k + 1), k, 1.0, A + k, k);
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, 1.0, A, k, A + k * (k + 1), k);
                dlauum("L", k, A, k, info);

            } else {

                dlauum("U", k, A + k * (k + 1), k, info);
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                            k, k, 1.0, A, k, 1.0, A + k * (k + 1), k);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasTrans, CblasNonUnit,
                            k, k, 1.0, A + k * k, k, A, k);
                dlauum("L", k, A + k * k, k, info);

            }

        }

    }
}
