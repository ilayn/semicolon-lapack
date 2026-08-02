/**
 * @file zpftrf.c
 * @brief ZPFTRF computes the Cholesky factorization of a Hermitian positive definite matrix in RFP format.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * ZPFTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite matrix A.
 *
 * The factorization has the form
 * @rst
 * .. code-block:: text
 *
 *     A = U**H * U,  if uplo = 'U', or
 *     A = L  * L**H,  if uplo = 'L',
 * @endrst
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the block version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in]     transr
 *                       - `'N'`: The Normal TRANSR of RFP A is stored
 *                       - `'C'`: The Conjugate-transpose TRANSR of RFP A is stored
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of RFP A is stored
 *                       - `'L'`: Lower triangle of RFP A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in,out] A      Complex array of dimension `n*(n+1)/2`.
 *                       On entry, the Hermitian matrix A in RFP format. RFP format is
 *                       described by `transr`, `uplo`, and `n` as follows: if `transr='N'`
 *                       then RFP A is `(0:n,0:k-1)` when n is even; `k=n/2`. RFP A is
 *                       `(0:n-1,0:k)` when n is odd; `k=n/2`. If `transr='C'` then RFP is
 *                       the conjugate-transpose of RFP A as defined when `transr='N'`. The
 *                       contents of RFP A are defined by `uplo` as follows: if `uplo='U'`
 *                       the RFP A contains the nt elements of upper packed A. If `uplo='L'`
 *                       the RFP A contains the elements of lower packed A. The LDA of RFP A
 *                       is `(n+1)/2` when `transr='C'`. When `transr` is `'N'` the LDA is
 *                       `n+1` when n is even and n is odd. See below for further details.
 *                       On exit, if `info=0`, the factor U or L from the Cholesky
 *                       factorization RFP A = U**H*U or RFP A = L*L**H.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the leading principal minor of order
 *                           i is not positive, and the factorization could not be
 *                           completed.
 *
 * @par Further Details:
 * @rst
 * We first consider Standard Packed Format when ``n`` is even.
 * We give an example where ``n=6``.
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
 * We next consider Standard Packed Format when ``n`` is odd.
 * We give an example where ``n=5``.
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
void zpftrf(
    const char* transr,
    const char* uplo,
    const INT n,
    c128* restrict A,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);

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
        xerbla("ZPFTRF", -(*info));
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

                zpotrf("L", n1, A, n, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            n2, n1, &CONE, A, n, &A[n1], n);
                cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n2, n1, -1.0, &A[n1], n, 1.0, &A[n], n);
                zpotrf("U", n2, &A[n], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            } else {

                zpotrf("L", n1, &A[n2], n, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, &CONE, &A[n2], n, A, n);
                cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            n2, n1, -1.0, A, n, 1.0, &A[n1], n);
                zpotrf("U", n2, &A[n1], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            }

        } else {

            if (lower) {

                zpotrf("U", n1, A, n1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            n1, n2, &CONE, A, n1, &A[n1 * n1], n1);
                cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans,
                            n2, n1, -1.0, &A[n1 * n1], n1, 1.0, &A[1], n1);
                zpotrf("L", n2, &A[1], n1, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            } else {

                zpotrf("U", n1, &A[n2 * n2], n2, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, &CONE, &A[n2 * n2], n2, A, n2);
                cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                            n2, n1, -1.0, A, n2, 1.0, &A[n1 * n2], n2);
                zpotrf("L", n2, &A[n1 * n2], n2, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                zpotrf("L", k, &A[1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, &A[1], n + 1, &A[k + 1], n + 1);
                cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, -1.0, &A[k + 1], n + 1, 1.0, A, n + 1);
                zpotrf("U", k, A, n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            } else {

                zpotrf("L", k, &A[k + 1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, &A[k + 1], n + 1, A, n + 1);
                cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            k, k, -1.0, A, n + 1, 1.0, &A[k], n + 1);
                zpotrf("U", k, &A[k], n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            }

        } else {

            if (lower) {

                zpotrf("U", k, &A[k], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasConjTrans, CblasNonUnit,
                            k, k, &CONE, &A[k], k, &A[k * (k + 1)], k);
                cblas_zherk(CblasColMajor, CblasLower, CblasConjTrans,
                            k, k, -1.0, &A[k * (k + 1)], k, 1.0, A, k);
                zpotrf("L", k, A, k, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            } else {

                zpotrf("U", k, &A[k * (k + 1)], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, &CONE, &A[k * (k + 1)], k, A, k);
                cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, -1.0, A, k, 1.0, &A[k * k], k);
                zpotrf("L", k, &A[k * k], k, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            }

        }

    }
}
