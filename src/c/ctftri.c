/**
 * @file ctftri.c
 * @brief CTFTRI computes the inverse of a triangular matrix stored in RFP format.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * CTFTRI computes the inverse of a triangular matrix A stored in RFP
 * format.
 *
 * This is a Level 3 BLAS version of the algorithm.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'C':  The Conjugate-transpose TRANSR of RFP A is stored.
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
 *          Complex*16 array, dimension (n*(n+1)/2).
 *          On entry, the triangular matrix in RFP format.
 *          On exit, the (triangular) inverse of the original matrix.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, A(i,i) is exactly zero. The triangular
 *                           matrix is singular and its inverse can not be computed.
 */
void ctftri(
    const char* transr,
    const char* uplo,
    const char* diag,
    const INT n,
    c64* restrict A,
    INT* info)
{
    const c64 cone = CMPLXF(1.0f, 0.0f);
    const c64 neg_cone = CMPLXF(-1.0f, 0.0f);

    INT lower, nisodd, normaltransr;
    INT n1, n2, k;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
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
        xerbla("CTFTRI", -(*info));
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

                ctrtri("L", diag, n1, A, n, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, &neg_cone, A, n, &A[n1], n);
                ctrtri("U", diag, n2, &A[n], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, &cone, &A[n], n, &A[n1], n);

            } else {

                ctrtri("L", diag, n1, &A[n2], n, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, &neg_cone, &A[n2], n, A, n);
                ctrtri("U", diag, n2, &A[n1], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, &cone, &A[n1], n, A, n);

            }

        } else {

            if (lower) {

                ctrtri("U", diag, n1, A, n1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, &neg_cone, A, n1, &A[n1 * n1], n1);
                ctrtri("L", diag, n2, &A[1], n1, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n1, n2, &cone, &A[1], n1, &A[n1 * n1], n1);

            } else {

                ctrtri("U", diag, n1, &A[n2 * n2], n2, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, &neg_cone, &A[n2 * n2], n2, A, n2);
                ctrtri("L", diag, n2, &A[n1 * n2], n2, info);
                if (*info > 0) {
                    *info = *info + n1;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            n2, n1, &cone, &A[n1 * n2], n2, A, n2);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                ctrtri("L", diag, k, &A[1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &neg_cone, &A[1], n + 1, &A[k + 1], n + 1);
                ctrtri("U", diag, k, A, n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &cone, A, n + 1, &A[k + 1], n + 1);

            } else {

                ctrtri("L", diag, k, &A[k + 1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &neg_cone, &A[k + 1], n + 1, A, n + 1);
                ctrtri("U", diag, k, &A[k], n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &cone, &A[k], n + 1, A, n + 1);

            }

        } else {

            if (lower) {

                ctrtri("U", diag, k, &A[k], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &neg_cone, &A[k], k, &A[k * (k + 1)], k);
                ctrtri("L", diag, k, A, k, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &cone, A, k, &A[k * (k + 1)], k);

            } else {

                ctrtri("U", diag, k, &A[k * (k + 1)], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &neg_cone, &A[k * (k + 1)], k, A, k);
                ctrtri("L", diag, k, &A[k * k], k, info);
                if (*info > 0) {
                    *info = *info + k;
                }
                if (*info > 0) {
                    return;
                }
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans,
                            (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                            k, k, &cone, &A[k * k], k, A, k);

            }

        }

    }
}
