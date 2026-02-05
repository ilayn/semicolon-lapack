/**
 * @file dpftrs.c
 * @brief DPFTRS solves a system of linear equations using Cholesky factorization in RFP format.
 */

#include "semicolon_lapack_double.h"

/**
 * DPFTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by DPFTRF.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'T':  The Transpose TRANSR of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of RFP A is stored;
 *          = 'L':  Lower triangle of RFP A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] A
 *          Double precision array, dimension (n*(n+1)/2).
 *          The triangular factor U or L from the Cholesky factorization.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of B. ldb >= max(1, n).
 *
 * @param[out] info
 *          = 0: successful exit
 *          < 0: if info = -i, the i-th argument had an illegal value
 */
void dpftrs(
    const char* transr,
    const char* uplo,
    const int n,
    const int nrhs,
    const double* const restrict A,
    double* const restrict B,
    const int ldb,
    int* info)
{
    int lower, normaltransr;

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
        xerbla("DPFTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (lower) {
        dtfsm(transr, "L", uplo, "N", "N", n, nrhs, 1.0, A, B, ldb);
        dtfsm(transr, "L", uplo, "T", "N", n, nrhs, 1.0, A, B, ldb);
    } else {
        dtfsm(transr, "L", uplo, "T", "N", n, nrhs, 1.0, A, B, ldb);
        dtfsm(transr, "L", uplo, "N", "N", n, nrhs, 1.0, A, B, ldb);
    }
}
