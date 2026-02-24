/**
 * @file zhbt21.c
 * @brief ZHBT21 checks a decomposition of the form A = U S U**H
 *        where A is hermitian banded, U is unitary, and S is diagonal
 *        or hermitian tridiagonal.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZHBT21 generally checks a decomposition of the form
 *
 *    A = U S U**H
 *
 * where **H means conjugate transpose, A is hermitian banded, U is
 * unitary, and S is diagonal (if KS=0) or hermitian
 * tridiagonal (if KS=1).
 *
 * Specifically:
 *    RESULT[0] = | A - U S U**H | / ( |A| n ulp )
 *    RESULT[1] = | I - U U**H | / ( n ulp )
 *
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle storage.
 * @param[in]     n      The size of the matrix.
 * @param[in]     ka     The bandwidth of A. Clamped to max(0, min(n-1, ka)).
 * @param[in]     ks     The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      Hermitian banded matrix in HB format, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1). Not referenced if ks=0.
 * @param[in]     U      Unitary matrix, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U.
 * @param[out]    work   Complex workspace, dimension (n*n + n*(n+1)/2).
 * @param[out]    rwork  Real workspace, dimension (n).
 * @param[out]    result Test ratios, dimension (2).
 */
void zhbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const c128* A, const INT lda,
            const f64* D, const f64* E,
            const c128* U, const INT ldu,
            c128* work, f64* rwork, f64* result)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT lower;
    char cuplo;
    INT ika, j, jc, jr;
    f64 anorm, ulp, unfl, wnorm;
    CBLAS_UPLO cblas_uplo;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    ika = ka;
    if (ika > n - 1)
        ika = n - 1;
    if (ika < 0)
        ika = 0;

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

    anorm = zlanhb("1", &cuplo, n, ika, A, lda, rwork);
    if (anorm < unfl)
        anorm = unfl;

    /* Compute error matrix:    Error = A - U S U**H

       Copy A from HB to HP storage format. */

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
                work[j] = CZERO;
                j++;
            }
        } else {
            INT nzeros = (jc > ika) ? jc - ika : 0;
            for (jr = 0; jr < nzeros; jr++) {
                work[j] = CZERO;
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
        cblas_zhpr(CblasColMajor, cblas_uplo, n, -D[j], &U[j * ldu], 1, work);
    }

    if (n > 1 && ks == 1) {
        for (j = 0; j < n - 1; j++) {
            c128 neg_ej = CMPLX(-E[j], 0.0);
            cblas_zhpr2(CblasColMajor, cblas_uplo, n, &neg_ej,
                        &U[j * ldu], 1, &U[(j + 1) * ldu], 1, work);
        }
    }
    wnorm = zlanhp("1", &cuplo, n, work, rwork);

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

    /* Compute  U U**H - I */

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

    for (j = 0; j < n; j++) {
        work[(n + 1) * j] -= CONE;
    }

    {
        f64 tmp = zlange("1", n, n, work, n, rwork);
        if (tmp > (f64)n)
            tmp = (f64)n;
        result[1] = tmp / (n * ulp);
    }
}
