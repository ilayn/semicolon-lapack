/**
 * @file spftrs.c
 * @brief SPFTRS solves a system of linear equations using Cholesky factorization in RFP format.
 */

#include "semicolon_lapack_single.h"

/**
 * SPFTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by SPFTRF.
 *
 * @param[in]     transr
 *                       - `'N'`: The Normal TRANSR of RFP A is stored
 *                       - `'T'`: The Transpose TRANSR of RFP A is stored
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of RFP A is stored
 *                       - `'L'`: Lower triangle of RFP A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     A      Array of dimension `n*(n+1)/2`.
 *                       The triangular factor U or L from the Cholesky factorization
 *                       of RFP A = U**T*U or RFP A = L*L**T, as computed by `spftrf`.
 *                       See below for more details about RFP A.
 * @param[in,out] B      Array of dimension `(ldb,nrhs)`.
 *                       On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
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
 * the transpose of the first three columns of AP upper.
 * For ``uplo='L'`` the lower trapezoid A(1:6,0:2) consists of the first
 * three columns of AP lower. The upper triangle A(0:2,0:2) consists of
 * the transpose of the last three columns of AP lower.
 * This covers the case n even and ``transr='N'``.
 *
 * .. code-block:: text
 *
 *          RFP A                   RFP A
 *
 *         03 04 05                33 43 53
 *         13 14 15                00 44 54
 *         23 24 25                10 11 55
 *         33 34 35                20 21 22
 *         00 44 45                30 31 32
 *         01 11 55                40 41 42
 *         02 12 22                50 51 52
 *
 * Now let ``transr='T'``. RFP A in both `uplo` cases is just the
 * transpose of RFP A above. One therefore gets:
 *
 * .. code-block:: text
 *
 *            RFP A                   RFP A
 *
 *      03 13 23 33 00 01 02    33 00 10 20 30 40 50
 *      04 14 24 34 44 11 12    43 44 11 21 31 41 51
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
 * the transpose of the first two columns of AP upper.
 * For ``uplo='L'`` the lower trapezoid A(0:4,0:2) consists of the first
 * three columns of AP lower. The upper triangle A(0:1,1:2) consists of
 * the transpose of the last two columns of AP lower.
 * This covers the case n odd and ``transr='N'``.
 *
 * .. code-block:: text
 *
 *          RFP A                   RFP A
 *
 *         02 03 04                00 33 43
 *         12 13 14                10 11 44
 *         22 23 24                20 21 22
 *         00 33 34                30 31 32
 *         01 11 44                40 41 42
 *
 * Now let ``transr='T'``. RFP A in both `uplo` cases is just the
 * transpose of RFP A above. One therefore gets:
 *
 * .. code-block:: text
 *
 *            RFP A                   RFP A
 *
 *      02 12 22 00 01             00 10 20 30 40 50
 *      03 13 23 33 11             33 11 21 31 41 51
 *      04 14 24 34 44             43 44 22 32 42 52
 * @endrst
 */
void spftrs(
    const char* transr,
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32* restrict A,
    f32* restrict B,
    const INT ldb,
    INT* info)
{
    INT lower, normaltransr;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("SPFTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (lower) {
        stfsm(transr, "L", uplo, "N", "N", n, nrhs, 1.0f, A, B, ldb);
        stfsm(transr, "L", uplo, "T", "N", n, nrhs, 1.0f, A, B, ldb);
    } else {
        stfsm(transr, "L", uplo, "T", "N", n, nrhs, 1.0f, A, B, ldb);
        stfsm(transr, "L", uplo, "N", "N", n, nrhs, 1.0f, A, B, ldb);
    }
}
