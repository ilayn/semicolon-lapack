/**
 * @file cpftri.c
 * @brief CPFTRI computes the inverse of a Hermitian positive definite matrix in RFP format.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * CPFTRI computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by CPFTRF.
 *
 * @param[in]     transr
 *                       - `'N'`: The Normal TRANSR of RFP A is stored
 *                       - `'C'`: The Conjugate-transpose TRANSR of RFP A is stored
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in,out] A      Complex array of dimension `n*(n+1)/2`.
 *                       On entry, the Cholesky factor in RFP format.
 *                       On exit, the inverse of the original matrix.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the (i,i) element of the factor is
 *                           zero
 *
 * @par Further Details:
 * @rst
 * We first consider Rectangular Full Packed (RFP) Format when ``n`` is
 * even. We give an example where ``n=6``.
 *
 * .. code-block:: text
 *
 *      AP is Upper             AP is Lower
 *
 *   00 01 02 03 04 05       00
 *      11 12 13 14 15       10 11
 *         22 23 24 25       20 21 22
 *            33 34 35       30 31 32 33
 *               44 45       40 41 42 43 44
 *                  55       50 51 52 53 54 55
 *
 * Let ``transr='N'``. RFP holds AP as follows:
 * For ``uplo='U'`` the upper trapezoid A(0:5,0:2) consists of the last
 * three columns of AP upper. The lower triangle A(4:6,0:2) consists of
 * conjugate-transpose of the first three columns of AP upper.
 * For ``uplo='L'`` the lower trapezoid A(1:6,0:2) consists of the first
 * three columns of AP lower. The upper triangle A(0:2,0:2) consists of
 * conjugate-transpose of the last three columns of AP lower.
 * To denote conjugate we place -- above the element. This covers the
 * case n even and ``transr='N'``.
 *
 * .. code-block:: text
 *
 *          RFP A                   RFP A
 *
 *                                -- -- --
 *         03 04 05                33 43 53
 *                                    -- --
 *         13 14 15                00 44 54
 *                                       --
 *         23 24 25                10 11 55
 *
 *         33 34 35                20 21 22
 *         --
 *         00 44 45                30 31 32
 *         -- --
 *         01 11 55                40 41 42
 *         -- -- --
 *         02 12 22                50 51 52
 *
 * Now let ``transr='C'``. RFP A in both `uplo` cases is just the conjugate-
 * transpose of RFP A above. One therefore gets:
 *
 * .. code-block:: text
 *
 *            RFP A                   RFP A
 *
 *      -- -- -- --                -- -- -- -- -- --
 *      03 13 23 33 00 01 02    33 00 10 20 30 40 50
 *      -- -- -- -- --                -- -- -- -- --
 *      04 14 24 34 44 11 12    43 44 11 21 31 41 51
 *      -- -- -- -- -- --                -- -- -- --
 *      05 15 25 35 45 55 22    53 54 55 22 32 42 52
 *
 * We then consider Rectangular Full Packed (RFP) Format when ``n`` is
 * odd. We give an example where ``n=5``.
 *
 * .. code-block:: text
 *
 *     AP is Upper                 AP is Lower
 *
 *   00 01 02 03 04              00
 *      11 12 13 14              10 11
 *         22 23 24              20 21 22
 *            33 34              30 31 32 33
 *               44              40 41 42 43 44
 *
 * Let ``transr='N'``. RFP holds AP as follows:
 * For ``uplo='U'`` the upper trapezoid A(0:4,0:2) consists of the last
 * three columns of AP upper. The lower triangle A(3:4,0:1) consists of
 * conjugate-transpose of the first two columns of AP upper.
 * For ``uplo='L'`` the lower trapezoid A(0:4,0:2) consists of the first
 * three columns of AP lower. The upper triangle A(0:1,1:2) consists of
 * conjugate-transpose of the last two columns of AP lower.
 * To denote conjugate we place -- above the element. This covers the
 * case n odd and ``transr='N'``.
 *
 * .. code-block:: text
 *
 *          RFP A                   RFP A
 *
 *                                   -- --
 *         02 03 04                00 33 43
 *                                      --
 *         12 13 14                10 11 44
 *
 *         22 23 24                20 21 22
 *         --
 *         00 33 34                30 31 32
 *         -- --
 *         01 11 44                40 41 42
 *
 * Now let ``transr='C'``. RFP A in both `uplo` cases is just the conjugate-
 * transpose of RFP A above. One therefore gets:
 *
 * .. code-block:: text
 *
 *            RFP A                   RFP A
 *
 *      -- -- --                   -- -- -- -- -- --
 *      02 12 22 00 01             00 10 20 30 40 50
 *      -- -- -- --                   -- -- -- -- --
 *      03 13 23 33 11             33 11 21 31 41 51
 *      -- -- -- -- --                   -- -- -- --
 *      04 14 24 34 44             43 44 22 32 42 52
 * @endrst
 */
void cpftri(
    const char* transr,
    const char* uplo,
    const INT n,
    c64* restrict A,
    INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT lower, nisodd, normaltransr;
    INT n1, n2, k;

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
                            n1, n2, 1.0f, &A[n1], n, 1.0f, A, n);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, &CONE, &A[n], n, &A[n1], n);
                clauum("U", n2, &A[n], n, info);

            } else {

                clauum("L", n1, &A[n2], n, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                            n1, n2, 1.0f, A, n, 1.0f, &A[n2], n);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            n1, n2, &CONE, &A[n1], n, A, n);
                clauum("U", n2, &A[n1], n, info);

            }

        } else {

            if (lower) {

                clauum("U", n1, A, n1, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n1, n2, 1.0f, &A[n1 * n1], n1, 1.0f, A, n1);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, &CONE, &A[1], n1, &A[n1 * n1], n1);
                clauum("L", n2, &A[1], n1, info);

            } else {

                clauum("U", n1, &A[n2 * n2], n2, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            n1, n2, 1.0f, A, n2, 1.0f, &A[n2 * n2], n2);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            n2, n1, &CONE, &A[n1 * n2], n2, A, n2);
                clauum("L", n2, &A[n1 * n2], n2, info);

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                clauum("L", k, &A[1], n + 1, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                            k, k, 1.0f, &A[k + 1], n + 1, 1.0f, &A[1], n + 1);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, n + 1, &A[k + 1], n + 1);
                clauum("U", k, A, n + 1, info);

            } else {

                clauum("L", k, &A[k + 1], n + 1, info);
                cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, 1.0f, A, n + 1, 1.0f, &A[k + 1], n + 1);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, &A[k], n + 1, A, n + 1);
                clauum("U", k, &A[k], n + 1, info);

            }

        } else {

            if (lower) {

                clauum("U", k, &A[k], k, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, 1.0f, &A[k * (k + 1)], k, 1.0f, &A[k], k);
                cblas_ctrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, A, k, &A[k * (k + 1)], k);
                clauum("L", k, A, k, info);

            } else {

                clauum("U", k, &A[k * (k + 1)], k, info);
                cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            k, k, 1.0f, A, k, 1.0f, &A[k * (k + 1)], k);
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, &A[k * k], k, A, k);
                clauum("L", k, &A[k * k], k, info);

            }

        }

    }
}
