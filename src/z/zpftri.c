/**
 * @file zpftri.c
 * @brief ZPFTRI computes the inverse of a Hermitian positive definite matrix in RFP format.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZPFTRI computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by ZPFTRF.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'C':  The Conjugate-transpose TRANSR of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (n*(n+1)/2).
 *          On entry, the Cholesky factor in RFP format.
 *          On exit, the inverse of the original matrix.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the (i,i) element of the factor is zero
 */
void zpftri(
    const char* transr,
    const char* uplo,
    const int n,
    c128* const restrict A,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);

    int lower, nisodd, normaltransr;
    int n1, n2, k;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("ZPFTRI", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    ztftri(transr, uplo, "N", n, A, info);
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

                zlauum("L", n1, A, n, info);
                cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans,
                            n1, n2, 1.0, A + n1, n, 1.0, A, n);
                cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, &CONE, A + n, n, A + n1, n);
                zlauum("U", n2, A + n, n, info);

            } else {

                zlauum("L", n1, A + n2, n, info);
                cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                            n1, n2, 1.0, A, n, 1.0, A + n2, n);
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            n1, n2, &CONE, A + n1, n, A, n);
                zlauum("U", n2, A + n1, n, info);

            }

        } else {

            if (lower) {

                zlauum("U", n1, A, n1, info);
                cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n1, n2, 1.0, A + n1 * n1, n1, 1.0, A, n1);
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, &CONE, A + 1, n1, A + n1 * n1, n1);
                zlauum("L", n2, A + 1, n1, info);

            } else {

                zlauum("U", n1, A + n2 * n2, n2, info);
                cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            n1, n2, 1.0, A, n2, 1.0, A + n2 * n2, n2);
                cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            n2, n1, &CONE, A + n1 * n2, n2, A, n2);
                zlauum("L", n2, A + n1 * n2, n2, info);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                zlauum("L", k, A + 1, n + 1, info);
                cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans,
                            k, k, 1.0, A + k + 1, n + 1, 1.0, A + 1, n + 1);
                cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, n + 1, A + k + 1, n + 1);
                zlauum("U", k, A, n + 1, info);

            } else {

                zlauum("L", k, A + k + 1, n + 1, info);
                cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, 1.0, A, n + 1, 1.0, A + k + 1, n + 1);
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, A + k, n + 1, A, n + 1);
                zlauum("U", k, A + k, n + 1, info);

            }

        } else {

            if (lower) {

                zlauum("U", k, A + k, k, info);
                cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, 1.0, A + k * (k + 1), k, 1.0, A + k, k);
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, k, A + k * (k + 1), k);
                zlauum("L", k, A, k, info);

            } else {

                zlauum("U", k, A + k * (k + 1), k, info);
                cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            k, k, 1.0, A, k, 1.0, A + k * (k + 1), k);
                cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, A + k * k, k, A, k);
                zlauum("L", k, A + k * k, k, info);

            }

        }

    }
}
