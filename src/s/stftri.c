/**
 * @file stftri.c
 * @brief STFTRI computes the inverse of a triangular matrix stored in RFP format.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>

/**
 * STFTRI computes the inverse of a triangular matrix A stored in RFP
 * format.
 *
 * This is a Level 3 BLAS version of the algorithm.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'T':  The Transpose TRANSR of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 * @param[in] diag
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (n*(n+1)/2).
 *          On entry, the triangular matrix in RFP format.
 *          On exit, the (triangular) inverse of the original matrix.
 *
 * @param[out] info
 *          = 0: successful exit
 *          < 0: if info = -i, the i-th argument had an illegal value
 *          > 0: if info = i, A(i,i) is exactly zero. The triangular
 *               matrix is singular and its inverse can not be computed.
 */
void stftri(
    const char* transr,
    const char* uplo,
    const char* diag,
    const int n,
    float* const restrict A,
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
    } else if (!(diag[0] == 'N' || diag[0] == 'n') &&
               !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("STFTRI", -(*info));
        return;
    }

    if (n == 0) {
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

                strtri("L", diag, n1, A, n, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, -1.0f, A, n, A + n1, n);
                strtri("U", diag, n2, A + n, n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, 1.0f, A + n, n, A + n1, n);

            } else {

                strtri("L", diag, n1, A + n2, n, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, -1.0f, A + n2, n, A, n);
                strtri("U", diag, n2, A + n1, n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, 1.0f, A + n1, n, A, n);

            }

        } else {

            if (lower) {

                strtri("U", diag, n1, A, n1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, -1.0f, A, n1, A + n1 * n1, n1);
                strtri("L", diag, n2, A + 1, n1, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, 1.0f, A + 1, n1, A + n1 * n1, n1);

            } else {

                strtri("U", diag, n1, A + n2 * n2, n2, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, -1.0f, A + n2 * n2, n2, A, n2);
                strtri("L", diag, n2, A + n1 * n2, n2, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, 1.0f, A + n1 * n2, n2, A, n2);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                strtri("L", diag, k, A + 1, n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, -1.0f, A + 1, n + 1, A + k + 1, n + 1);
                strtri("U", diag, k, A, n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, 1.0f, A, n + 1, A + k + 1, n + 1);

            } else {

                strtri("L", diag, k, A + k + 1, n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, -1.0f, A + k + 1, n + 1, A, n + 1);
                strtri("U", diag, k, A + k, n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, 1.0f, A + k, n + 1, A, n + 1);

            }

        } else {

            if (lower) {

                strtri("U", diag, k, A + k, k, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, -1.0f, A + k, k, A + k * (k + 1), k);
                strtri("L", diag, k, A, k, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, 1.0f, A, k, A + k * (k + 1), k);

            } else {

                strtri("U", diag, k, A + k * (k + 1), k, info);
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, -1.0f, A + k * (k + 1), k, A, k);
                strtri("L", diag, k, A + k * k, k, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, 1.0f, A + k * k, k, A, k);

            }

        }

    }
}
