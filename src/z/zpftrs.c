/**
 * @file zpftrs.c
 * @brief ZPFTRS solves a system of linear equations using Cholesky factorization in RFP format.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZPFTRS solves a system of linear equations A*X = B with a Hermitian
 * positive definite matrix A using the Cholesky factorization
 * A = U**H*U or A = L*L**H computed by ZPFTRF.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'C':  The Conjugate-transpose TRANSR of RFP A is stored.
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
 *          Complex*16 array, dimension (n*(n+1)/2).
 *          The triangular factor U or L from the Cholesky factorization.
 *
 * @param[in,out] B
 *          Complex*16 array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of B. ldb >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zpftrs(
    const char* transr,
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict A,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower, normaltransr;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
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
        xerbla("ZPFTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (lower) {
        ztfsm(transr, "L", uplo, "N", "N", n, nrhs, CONE, A, B, ldb);
        ztfsm(transr, "L", uplo, "C", "N", n, nrhs, CONE, A, B, ldb);
    } else {
        ztfsm(transr, "L", uplo, "C", "N", n, nrhs, CONE, A, B, ldb);
        ztfsm(transr, "L", uplo, "N", "N", n, nrhs, CONE, A, B, ldb);
    }
}
