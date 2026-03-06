/**
 * @file chpt21.c
 * @brief CHPT21 checks a decomposition of the form A = U S U**H
 *        where A is Hermitian (packed format), U is unitary, and S is
 *        diagonal or tridiagonal.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CHPT21 generally checks a decomposition of the form
 *
 *         A = U S U**H
 *
 * where **H means conjugate transpose, A is Hermitian, U is unitary,
 * and S is diagonal (if KBAND=0) or (real) symmetric tridiagonal
 * (if KBAND=1). If ITYPE=1, then U is represented as a dense matrix,
 * otherwise the U is expressed as a product of Householder
 * transformations, whose vectors are stored in the array "VP" and
 * whose scaling constants are in "TAU".
 *
 * Specifically, if ITYPE=1, then:
 *    RESULT(1) = | A - U S U**H | / ( |A| n ulp )
 *    RESULT(2) = | I - U U**H | / ( n ulp )
 *
 * If ITYPE=2, then:
 *    RESULT(1) = | A - V S V**H | / ( |A| n ulp )
 *
 * If ITYPE=3, then:
 *    RESULT(1) = | I - U V**H | / ( n ulp )
 *
 * @param[in]     itype  Type of test (1, 2, or 3).
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle.
 * @param[in]     n      The size of the matrix.
 * @param[in]     kband  Bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     AP     Packed Hermitian matrix, dimension (n*(n+1)/2).
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n). Not referenced if kband=0.
 * @param[in]     U      Unitary matrix, dimension (ldu, n). Not referenced if itype=2.
 * @param[in]     ldu    Leading dimension of U.
 * @param[in,out] VP     Householder vectors in packed format, dimension (n*(n+1)/2).
 * @param[in]     tau    Householder scaling factors, dimension (n).
 * @param[out]    work   Workspace, dimension (n*n).
 * @param[out]    rwork  Real workspace, dimension (n).
 * @param[out]    result Test ratios, dimension (2). result[1] only set if itype=1.
 */
void chpt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c64* AP, const f32* D, const f32* E,
            const c64* U, const INT ldu,
            c64* VP, const c64* tau,
            c64* work, f32* rwork, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TEN = 10.0f;
    const f32 HALF = 0.5f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT lower;
    char cuplo;
    INT iinfo, j, jp, jp1, jr, lap;
    f32 anorm, ulp, unfl, wnorm;
    c64 temp, vsave;

    /* Constants */

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

    unfl = slamch("S");
    ulp = slamch("E") * slamch("B");

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
        anorm = clanhp("1", &cuplo, n, AP, rwork);
        if (anorm < unfl)
            anorm = unfl;
    }

    /* Compute error matrix: */

    if (itype == 1) {

        /* ITYPE=1: error = A - U S U**H */

        claset("F", n, n, CZERO, CZERO, work, n);
        cblas_ccopy(lap, AP, 1, work, 1);

        for (j = 0; j < n; j++) {
            cblas_chpr(CblasColMajor,
                       lower ? CblasLower : CblasUpper,
                       n, -D[j], &U[j * ldu], 1, work);
        }

        if (n > 1 && kband == 1) {
            for (j = 1; j < n - 1; j++) {
                c64 neg_ej = CMPLXF(-E[j], 0.0f);
                cblas_chpr2(CblasColMajor,
                            lower ? CblasLower : CblasUpper,
                            n, &neg_ej, &U[j * ldu], 1,
                            &U[(j - 1) * ldu], 1, work);
            }
        }
        wnorm = clanhp("1", &cuplo, n, work, rwork);

    } else if (itype == 2) {

        /* ITYPE=2: error = V S V**H - A */

        claset("F", n, n, CZERO, CZERO, work, n);

        if (lower) {
            work[lap - 1] = D[n - 1];
            for (j = n - 2; j >= 0; j--) {
                jp = ((2 * n - (j + 1)) * j) / 2;
                jp1 = jp + n - j - 1;
                if (kband == 1) {
                    work[jp + j + 1] = (CONE - tau[j]) * E[j];
                    for (jr = j + 2; jr < n; jr++) {
                        work[jp + jr] = -tau[j] * E[j] * VP[jp + jr];
                    }
                }

                if (crealf(tau[j]) != 0.0f || cimagf(tau[j]) != 0.0f) {
                    vsave = VP[jp + j + 1];
                    VP[jp + j + 1] = CONE;
                    cblas_chpmv(CblasColMajor, CblasLower,
                                n - j - 1, &CONE, &work[jp + n],
                                &VP[jp + j + 1], 1, &CZERO, &work[lap], 1);
                    c64 dotc;
                    cblas_cdotc_sub(n - j - 1, &work[lap], 1,
                                    &VP[jp + j + 1], 1, &dotc);
                    temp = -HALF * tau[j] * dotc;
                    cblas_caxpy(n - j - 1, &temp, &VP[jp + j + 1], 1,
                                &work[lap], 1);
                    c64 neg_tau = -tau[j];
                    cblas_chpr2(CblasColMajor, CblasLower,
                                n - j - 1, &neg_tau, &VP[jp + j + 1], 1,
                                &work[lap], 1, &work[jp + n]);
                    VP[jp + j + 1] = vsave;
                }
                work[jp + j] = D[j];
            }
        } else {
            work[0] = D[0];
            for (j = 0; j < n - 1; j++) {
                jp = ((j + 1) * j) / 2;
                jp1 = jp + j + 1;
                if (kband == 1) {
                    work[jp1 + j] = (CONE - tau[j]) * E[j];
                    for (jr = 0; jr < j; jr++) {
                        work[jp1 + jr] = -tau[j] * E[j] * VP[jp1 + jr];
                    }
                }

                if (crealf(tau[j]) != 0.0f || cimagf(tau[j]) != 0.0f) {
                    vsave = VP[jp1 + j];
                    VP[jp1 + j] = CONE;
                    cblas_chpmv(CblasColMajor, CblasUpper,
                                j + 1, &CONE, work,
                                &VP[jp1], 1, &CZERO, &work[lap], 1);
                    c64 dotc;
                    cblas_cdotc_sub(j + 1, &work[lap], 1,
                                    &VP[jp1], 1, &dotc);
                    temp = -HALF * tau[j] * dotc;
                    cblas_caxpy(j + 1, &temp, &VP[jp1], 1, &work[lap], 1);
                    c64 neg_tau = -tau[j];
                    cblas_chpr2(CblasColMajor, CblasUpper,
                                j + 1, &neg_tau, &VP[jp1], 1,
                                &work[lap], 1, work);
                    VP[jp1 + j] = vsave;
                }
                work[jp1 + j + 1] = D[j + 1];
            }
        }

        for (j = 0; j < lap; j++) {
            work[j] -= AP[j];
        }
        wnorm = clanhp("1", &cuplo, n, work, rwork);

    } else if (itype == 3) {

        /* ITYPE=3: error = U V**H - I */

        if (n < 2)
            return;
        clacpy(" ", n, n, U, ldu, work, n);
        cupmtr("R", &cuplo, "C", n, n, VP, tau, work, n,
               &work[n * n], &iinfo);
        if (iinfo != 0) {
            result[0] = TEN / ulp;
            return;
        }

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        wnorm = clange("1", n, n, work, n, rwork);
    } else {
        wnorm = ZERO;
    }

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f32 tmp = fminf(wnorm, (f32)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f32 tmp = fminf(wnorm / anorm, (f32)n);
            result[0] = tmp / (n * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U U**H - I */

    if (itype == 1) {
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        f32 tmp = clange("1", n, n, work, n, rwork);
        result[1] = fminf(tmp, (f32)n) / (n * ulp);
    }
}
