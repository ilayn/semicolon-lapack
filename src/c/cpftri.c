/**
 * @file cpftri.c
 * @brief CPFTRI computes the inverse of a Hermitian positive definite matrix in RFP format.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CPFTRI computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by CPFTRF.
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
 *          Single complex array, dimension (n*(n+1)/2).
 *          On entry, the Cholesky factor in RFP format.
 *          On exit, the inverse of the original matrix.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the (i,i) element of the factor is zero
 */
void cpftri(
    const char* transr,
    const char* uplo,
    const int n,
    c64* restrict A,
    int* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);

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
        xerbla("CPFTRI", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    ctftri(transr, uplo, "N", n, A, info);
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

                clauum("L", n1, A, n, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                            n1, n2, 1.0f, A + n1, n, 1.0f, A, n);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, &CONE, A + n, n, A + n1, n);
                clauum("U", n2, A + n, n, info);

            } else {

                clauum("L", n1, A + n2, n, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                            n1, n2, 1.0f, A, n, 1.0f, A + n2, n);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            n1, n2, &CONE, A + n1, n, A, n);
                clauum("U", n2, A + n1, n, info);

            }

        } else {

            if (lower) {

                clauum("U", n1, A, n1, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n1, n2, 1.0f, A + n1 * n1, n1, 1.0f, A, n1);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, &CONE, A + 1, n1, A + n1 * n1, n1);
                clauum("L", n2, A + 1, n1, info);

            } else {

                clauum("U", n1, A + n2 * n2, n2, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            n1, n2, 1.0f, A, n2, 1.0f, A + n2 * n2, n2);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            n2, n1, &CONE, A + n1 * n2, n2, A, n2);
                clauum("L", n2, A + n1 * n2, n2, info);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                clauum("L", k, A + 1, n + 1, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                            k, k, 1.0f, A + k + 1, n + 1, 1.0f, A + 1, n + 1);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, n + 1, A + k + 1, n + 1);
                clauum("U", k, A, n + 1, info);

            } else {

                clauum("L", k, A + k + 1, n + 1, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, 1.0f, A, n + 1, 1.0f, A + k + 1, n + 1);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, A + k, n + 1, A, n + 1);
                clauum("U", k, A + k, n + 1, info);

            }

        } else {

            if (lower) {

                clauum("U", k, A + k, k, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, 1.0f, A + k * (k + 1), k, 1.0f, A + k, k);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, k, A + k * (k + 1), k);
                clauum("L", k, A, k, info);

            } else {

                clauum("U", k, A + k * (k + 1), k, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            k, k, 1.0f, A, k, 1.0f, A + k * (k + 1), k);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, A + k * k, k, A, k);
                clauum("L", k, A + k * k, k, info);

            }

        }

    }
}
