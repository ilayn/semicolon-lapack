/**
 * @file zhpt21.c
 * @brief ZHPT21 checks a decomposition of the form A = U S U**H
 *        where A is Hermitian (packed format), U is unitary, and S is
 *        diagonal or tridiagonal.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZHPT21 generally checks a decomposition of the form
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
void zhpt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c128* AP, const f64* D, const f64* E,
            const c128* U, const INT ldu,
            c128* VP, const c128* tau,
            c128* work, f64* rwork, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 HALF = 0.5;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower;
    char cuplo;
    INT iinfo, j, jp, jp1, jr, lap;
    f64 anorm, ulp, unfl, wnorm;
    c128 temp, vsave;

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
        anorm = zlanhp("1", &cuplo, n, AP, rwork);
        if (anorm < unfl)
            anorm = unfl;
    }

    /* Compute error matrix: */

    if (itype == 1) {

        /* ITYPE=1: error = A - U S U**H */

        zlaset("F", n, n, CZERO, CZERO, work, n);
        cblas_zcopy(lap, AP, 1, work, 1);

        for (j = 0; j < n; j++) {
            cblas_zhpr(CblasColMajor,
                       lower ? CblasLower : CblasUpper,
                       n, -D[j], &U[j * ldu], 1, work);
        }

        if (n > 1 && kband == 1) {
            for (j = 1; j < n - 1; j++) {
                c128 neg_ej = CMPLX(-E[j], 0.0);
                cblas_zhpr2(CblasColMajor,
                            lower ? CblasLower : CblasUpper,
                            n, &neg_ej, &U[j * ldu], 1,
                            &U[(j - 1) * ldu], 1, work);
            }
        }
        wnorm = zlanhp("1", &cuplo, n, work, rwork);

    } else if (itype == 2) {

        /* ITYPE=2: error = V S V**H - A */

        zlaset("F", n, n, CZERO, CZERO, work, n);

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

                if (creal(tau[j]) != 0.0 || cimag(tau[j]) != 0.0) {
                    vsave = VP[jp + j + 1];
                    VP[jp + j + 1] = CONE;
                    cblas_zhpmv(CblasColMajor, CblasLower,
                                n - j - 1, &CONE, &work[jp + n],
                                &VP[jp + j + 1], 1, &CZERO, &work[lap], 1);
                    c128 dotc;
                    cblas_zdotc_sub(n - j - 1, &work[lap], 1,
                                    &VP[jp + j + 1], 1, &dotc);
                    temp = -HALF * tau[j] * dotc;
                    cblas_zaxpy(n - j - 1, &temp, &VP[jp + j + 1], 1,
                                &work[lap], 1);
                    c128 neg_tau = -tau[j];
                    cblas_zhpr2(CblasColMajor, CblasLower,
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

                if (creal(tau[j]) != 0.0 || cimag(tau[j]) != 0.0) {
                    vsave = VP[jp1 + j];
                    VP[jp1 + j] = CONE;
                    cblas_zhpmv(CblasColMajor, CblasUpper,
                                j + 1, &CONE, work,
                                &VP[jp1], 1, &CZERO, &work[lap], 1);
                    c128 dotc;
                    cblas_zdotc_sub(j + 1, &work[lap], 1,
                                    &VP[jp1], 1, &dotc);
                    temp = -HALF * tau[j] * dotc;
                    cblas_zaxpy(j + 1, &temp, &VP[jp1], 1, &work[lap], 1);
                    c128 neg_tau = -tau[j];
                    cblas_zhpr2(CblasColMajor, CblasUpper,
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
        wnorm = zlanhp("1", &cuplo, n, work, rwork);

    } else if (itype == 3) {

        /* ITYPE=3: error = U V**H - I */

        if (n < 2)
            return;
        zlacpy(" ", n, n, U, ldu, work, n);
        zupmtr("R", &cuplo, "C", n, n, VP, tau, work, n,
               &work[n * n], &iinfo);
        if (iinfo != 0) {
            result[0] = TEN / ulp;
            return;
        }

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        wnorm = zlange("1", n, n, work, n, rwork);
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

    /* Compute  U U**H - I */

    if (itype == 1) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        f64 tmp = zlange("1", n, n, work, n, rwork);
        result[1] = fmin(tmp, (f64)n) / (n * ulp);
    }
}
