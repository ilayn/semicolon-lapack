/**
 * @file dspt21.c
 * @brief DSPT21 checks a decomposition of the form A = U S U'
 *        where A is symmetric (stored in packed format), U is orthogonal,
 *        and S is diagonal or symmetric tridiagonal.
 *
 * Port of LAPACK's TESTING/EIG/dspt21.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DSPT21 generally checks a decomposition of the form
 *
 *    A = U S U'
 *
 * where ' means transpose, A is symmetric (stored in packed format),
 * U is orthogonal, and S is diagonal (if KBAND=0) or symmetric
 * tridiagonal (if KBAND=1). If ITYPE=1, then U is represented as a
 * dense matrix, otherwise the U is expressed as a product of Householder
 * transformations, whose vectors are stored in the array "VP" and whose
 * scaling constants are in "TAU".
 *
 * Specifically, if ITYPE=1, then:
 *    RESULT[0] = | A - U S U' | / ( |A| n ulp )
 *    RESULT[1] = | I - U U' | / ( n ulp )
 *
 * If ITYPE=2, then:
 *    RESULT[0] = | A - V S V' | / ( |A| n ulp )
 *
 * If ITYPE=3, then:
 *    RESULT[0] = | I - V U' | / ( n ulp )
 *
 * @param[in]     itype  Type of test (1, 2, or 3).
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle.
 * @param[in]     n      The size of the matrix.
 * @param[in]     kband  Bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     AP     Packed symmetric matrix, dimension (n*(n+1)/2).
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1). Not referenced if kband=0.
 * @param[in]     U      Dense orthogonal matrix, dimension (ldu, n). Not referenced if itype=2.
 * @param[in]     ldu    Leading dimension of U.
 * @param[in,out] VP     Householder vectors in packed format, dimension (n*(n+1)/2).
 *                       Modified temporarily if itype >= 2. Not referenced if itype=1.
 * @param[in]     tau    Householder scaling factors, dimension (n). Not referenced if itype < 2.
 * @param[out]    work   Workspace, dimension (n*n + n).
 * @param[out]    result Test ratios, dimension (2). result[1] only set if itype=1.
 */
void dspt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const f64* AP, const f64* D, const f64* E,
            const f64* U, const INT ldu,
            f64* VP, const f64* tau,
            f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 HALF = 0.5;

    INT lower;
    char cuplo;
    INT iinfo, j, jp, jp1, jr, lap;
    f64 anorm, temp, ulp, unfl, vsave, wnorm;

    /* 1) Constants */

    result[0] = ZERO;
    if (itype == 1)
        result[1] = ZERO;
    if (n <= 0)
        return;

    lap = (n * (n + 1)) / 2;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        lower = 0;
        cuplo = 'U';
    } else {
        lower = 1;
        cuplo = 'L';
    }

    unfl = dlamch("S");
    ulp = dlamch("E") * dlamch("B");

    /* Some Error Checks */

    if (itype < 1 || itype > 3) {
        result[0] = TEN / ulp;
        return;
    }

    /* Do Test 1 */

    /* Norm of A: */

    if (itype == 3) {
        anorm = ONE;
    } else {
        anorm = dlansp("1", &cuplo, n, AP, work);
        if (anorm < unfl)
            anorm = unfl;
    }

    /* Compute error matrix: */

    if (itype == 1) {

        /* ITYPE=1: error = A - U S U' */

        dlaset("F", n, n, ZERO, ZERO, work, n);
        cblas_dcopy(lap, AP, 1, work, 1);

        for (j = 0; j < n; j++) {
            cblas_dspr(CblasColMajor,
                       lower ? CblasLower : CblasUpper,
                       n, -D[j], &U[j * ldu], 1, work);
        }

        if (n > 1 && kband == 1) {
            for (j = 0; j < n - 1; j++) {
                cblas_dspr2(CblasColMajor,
                            lower ? CblasLower : CblasUpper,
                            n, -E[j], &U[j * ldu], 1, &U[(j + 1) * ldu], 1,
                            work);
            }
        }
        wnorm = dlansp("1", &cuplo, n, work, &work[n * n]);

    } else if (itype == 2) {

        /* ITYPE=2: error = V S V' - A */

        dlaset("F", n, n, ZERO, ZERO, work, n);

        if (lower) {
            work[lap - 1] = D[n - 1];
            for (j = n - 2; j >= 0; j--) {
                /* Fortran J (1-based) = j+1 */
                jp = ((2 * n - (j + 1)) * j) / 2;
                jp1 = jp + n - j - 1;
                if (kband == 1) {
                    work[jp + j + 1] = (ONE - tau[j]) * E[j];
                    for (jr = j + 2; jr < n; jr++) {
                        work[jp + jr] = -tau[j] * E[j] * VP[jp + jr];
                    }
                }

                if (tau[j] != ZERO) {
                    vsave = VP[jp + j + 1];
                    VP[jp + j + 1] = ONE;
                    cblas_dspmv(CblasColMajor, CblasLower,
                                n - j - 1, ONE, &work[jp + n],
                                &VP[jp + j + 1], 1, ZERO, &work[lap], 1);
                    temp = -HALF * tau[j] * cblas_ddot(n - j - 1,
                                &work[lap], 1, &VP[jp + j + 1], 1);
                    cblas_daxpy(n - j - 1, temp, &VP[jp + j + 1], 1,
                                &work[lap], 1);
                    cblas_dspr2(CblasColMajor, CblasLower,
                                n - j - 1, -tau[j], &VP[jp + j + 1], 1,
                                &work[lap], 1, &work[jp + n]);
                    VP[jp + j + 1] = vsave;
                }
                work[jp + j] = D[j];
            }
        } else {
            work[0] = D[0];
            for (j = 0; j < n - 1; j++) {
                /* Fortran J (1-based) = j+1 */
                jp = ((j + 1) * j) / 2;
                jp1 = jp + j + 1;
                if (kband == 1) {
                    work[jp1 + j] = (ONE - tau[j]) * E[j];
                    for (jr = 0; jr < j; jr++) {
                        work[jp1 + jr] = -tau[j] * E[j] * VP[jp1 + jr];
                    }
                }

                if (tau[j] != ZERO) {
                    vsave = VP[jp1 + j];
                    VP[jp1 + j] = ONE;
                    cblas_dspmv(CblasColMajor, CblasUpper,
                                j + 1, ONE, work,
                                &VP[jp1], 1, ZERO, &work[lap], 1);
                    temp = -HALF * tau[j] * cblas_ddot(j + 1,
                                &work[lap], 1, &VP[jp1], 1);
                    cblas_daxpy(j + 1, temp, &VP[jp1], 1, &work[lap], 1);
                    cblas_dspr2(CblasColMajor, CblasUpper,
                                j + 1, -tau[j], &VP[jp1], 1,
                                &work[lap], 1, work);
                    VP[jp1 + j] = vsave;
                }
                work[jp1 + j + 1] = D[j + 1];
            }
        }

        for (j = 0; j < lap; j++) {
            work[j] -= AP[j];
        }
        wnorm = dlansp("1", &cuplo, n, work, &work[lap]);

    } else if (itype == 3) {

        /* ITYPE=3: error = U V' - I */

        if (n < 2)
            return;
        dlacpy(" ", n, n, U, ldu, work, n);
        dopmtr("R", &cuplo, "T", n, n, VP, tau, work, n,
               &work[n * n], &iinfo);
        if (iinfo != 0) {
            result[0] = TEN / ulp;
            return;
        }

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= ONE;
        }

        wnorm = dlange("1", n, n, work, n, &work[n * n]);
    } else {
        wnorm = ZERO;
    }

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f64 tmp = fmin(wnorm, (f64)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f64 tmp = fmin(wnorm / anorm, (f64)n);
            result[0] = tmp / (n * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U U' - I */

    if (itype == 1) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= ONE;
        }

        {
            f64 tmp = dlange("1", n, n, work, n, &work[n * n]);
            result[1] = fmin(tmp, (f64)n) / (n * ulp);
        }
    }
}
