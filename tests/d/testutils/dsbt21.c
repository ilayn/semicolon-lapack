/**
 * @file dsbt21.c
 * @brief DSBT21 checks a decomposition of the form A = U S U'
 *        where A is symmetric banded, U is orthogonal, and S is diagonal
 *        or symmetric tridiagonal.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DSBT21 generally checks a decomposition of the form
 *
 *    A = U S U'
 *
 * where ' means transpose, A is symmetric banded, U is orthogonal,
 * and S is diagonal (if KS=0) or symmetric tridiagonal (if KS=1).
 *
 * Specifically:
 *    RESULT[0] = | A - U S U' | / ( |A| n ulp )
 *    RESULT[1] = | I - U U' | / ( n ulp )
 *
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle storage.
 * @param[in]     n      The size of the matrix.
 * @param[in]     ka     The bandwidth of A. Clamped to max(0, min(n-1, ka)).
 * @param[in]     ks     The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      Symmetric banded matrix in SB format, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1). Not referenced if ks=0.
 * @param[in]     U      Orthogonal matrix, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U.
 * @param[out]    work   Workspace, dimension (n*n + n).
 * @param[out]    result Test ratios, dimension (2).
 */
void dsbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const f64* A, const INT lda,
            const f64* D, const f64* E,
            const f64* U, const INT ldu,
            f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT lower;
    char cuplo;
    INT ika, j, jc, jr, lw;
    f64 anorm, ulp, unfl, wnorm;
    CBLAS_UPLO cblas_uplo;

    /* Constants */

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    ika = ka;
    if (ika > n - 1)
        ika = n - 1;
    if (ika < 0)
        ika = 0;
    lw = (n * (n + 1)) / 2;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        lower = 0;
        cuplo = 'U';
        cblas_uplo = CblasUpper;
    } else {
        lower = 1;
        cuplo = 'L';
        cblas_uplo = CblasLower;
    }

    unfl = dlamch("S");
    ulp = dlamch("E") * dlamch("B");

    /* Do Test 1 */

    /* Norm of A: */

    anorm = dlansb("1", &cuplo, n, ika, A, lda, work);
    if (anorm < unfl)
        anorm = unfl;

    /* Compute error matrix:    Error = A - U S U'

       Copy A from SB to SP storage format. */

    j = 0;
    for (jc = 0; jc < n; jc++) {
        if (lower) {
            INT nelem = ika + 1;
            if (nelem > n - jc)
                nelem = n - jc;
            for (jr = 0; jr < nelem; jr++) {
                work[j] = A[jr + jc * lda];
                j++;
            }
            for (jr = nelem; jr < n - jc; jr++) {
                work[j] = ZERO;
                j++;
            }
        } else {
            INT nzeros = (jc > ika) ? jc - ika : 0;
            for (jr = 0; jr < nzeros; jr++) {
                work[j] = ZERO;
                j++;
            }
            INT top = (ika < jc) ? ika : jc;
            for (jr = top; jr >= 0; jr--) {
                work[j] = A[ika - jr + jc * lda];
                j++;
            }
        }
    }

    for (j = 0; j < n; j++) {
        cblas_dspr(CblasColMajor, cblas_uplo, n, -D[j], &U[j * ldu], 1, work);
    }

    if (n > 1 && ks == 1) {
        for (j = 0; j < n - 1; j++) {
            cblas_dspr2(CblasColMajor, cblas_uplo, n, -E[j],
                        &U[j * ldu], 1, &U[(j + 1) * ldu], 1, work);
        }
    }
    wnorm = dlansp("1", &cuplo, n, work, &work[lw]);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f64 tmp = wnorm;
            if (tmp > (f64)n * anorm)
                tmp = (f64)n * anorm;
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f64 tmp = wnorm / anorm;
            if (tmp > (f64)n)
                tmp = (f64)n;
            result[0] = tmp / (n * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U U' - I */

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

    for (j = 0; j < n; j++) {
        work[(n + 1) * j] -= ONE;
    }

    {
        f64 tmp = dlange("1", n, n, work, n, &work[n * n]);
        if (tmp > (f64)n)
            tmp = (f64)n;
        result[1] = tmp / (n * ulp);
    }
}
