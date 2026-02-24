/**
 * @file zhet21.c
 * @brief ZHET21 checks a decomposition of the form A = U S U**H
 *        where A is Hermitian, U is unitary, and S is diagonal or tridiagonal.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZHET21 generally checks a decomposition of the form
 *
 *    A = U S U**H
 *
 * where **H means conjugate transpose, A is Hermitian, U is unitary, and
 * S is diagonal (if KBAND=0) or (real) symmetric tridiagonal (if KBAND=1).
 *
 * If ITYPE=1, then U is represented as a dense matrix; otherwise U is
 * expressed as a product of Householder transformations, whose vectors
 * are stored in the array "V" and whose scaling constants are in "TAU".
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
 * @param[in]     uplo   'U' for upper triangle, 'L' for lower triangle.
 * @param[in]     n      The size of the matrix.
 * @param[in]     kband  Bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      Hermitian matrix, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1).
 * @param[in]     U      Unitary matrix, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U.
 * @param[in,out] V      Householder vectors, dimension (ldv, n).
 * @param[in]     ldv    Leading dimension of V.
 * @param[in]     tau    Householder scaling factors, dimension (n).
 * @param[out]    work   Workspace, dimension (2*n*n).
 * @param[out]    rwork  Real workspace, dimension (n).
 * @param[out]    result Test ratios, dimension (2).
 */
void zhet21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c128* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict U, const INT ldu,
            c128* restrict V, const INT ldv,
            const c128* const restrict tau,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower;
    char cuplo;
    INT j, jcol, jrow, iinfo;
    f64 anorm, ulp, unfl, wnorm;
    c128 vsave;

    result[0] = ZERO;
    if (itype == 1)
        result[1] = ZERO;
    if (n <= 0)
        return;

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

    /* Norm of A */
    if (itype == 3) {
        anorm = ONE;
    } else {
        anorm = zlanhe("1", &cuplo, n, A, lda, rwork);
        if (anorm < unfl)
            anorm = unfl;
    }

    /* Compute error matrix */
    if (itype == 1) {
        /*
         * ITYPE=1: error = A - U S U**H
         */
        zlaset("F", n, n, CZERO, CZERO, work, n);
        zlacpy(&cuplo, n, n, A, lda, work, n);

        for (j = 0; j < n; j++) {
            cblas_zher(CblasColMajor,
                       lower ? CblasLower : CblasUpper,
                       n, -D[j], &U[j * ldu], 1, work, n);
        }

        if (n > 1 && kband == 1) {
            for (j = 0; j < n - 1; j++) {
                c128 neg_ej = CMPLX(-E[j], 0.0);
                cblas_zher2(CblasColMajor,
                            lower ? CblasLower : CblasUpper,
                            n, &neg_ej, &U[j * ldu], 1, &U[(j + 1) * ldu], 1,
                            work, n);
            }
        }
        wnorm = zlanhe("1", &cuplo, n, work, n, rwork);

    } else if (itype == 2) {
        /*
         * ITYPE=2: error = V S V**H - A
         */
        zlaset("F", n, n, CZERO, CZERO, work, n);

        if (lower) {
            work[n * n - 1] = D[n - 1];
            for (j = n - 2; j >= 0; j--) {
                if (kband == 1) {
                    work[(n + 1) * j + 1] = (CONE - tau[j]) * E[j];
                    for (INT jr = j + 2; jr < n; jr++) {
                        work[j * n + jr] = -tau[j] * E[j] * V[jr + j * ldv];
                    }
                }

                vsave = V[(j + 1) + j * ldv];
                V[(j + 1) + j * ldv] = ONE;
                zlarfy("L", n - j - 1, &V[(j + 1) + j * ldv], 1, tau[j],
                       &work[(n + 1) * (j + 1)], n, &work[n * n]);
                V[(j + 1) + j * ldv] = vsave;
                work[(n + 1) * j] = D[j];
            }
        } else {
            work[0] = D[0];
            for (j = 0; j < n - 1; j++) {
                if (kband == 1) {
                    work[(n + 1) * (j + 1) - 1] = (CONE - tau[j]) * E[j];
                    for (INT jr = 0; jr < j; jr++) {
                        work[(j + 1) * n + jr] = -tau[j] * E[j] * V[jr + (j + 1) * ldv];
                    }
                }

                vsave = V[j + (j + 1) * ldv];
                V[j + (j + 1) * ldv] = ONE;
                zlarfy("U", j + 1, &V[(j + 1) * ldv], 1, tau[j],
                       work, n, &work[n * n]);
                V[j + (j + 1) * ldv] = vsave;
                work[(n + 1) * (j + 1)] = D[j + 1];
            }
        }

        /* Subtract A */
        for (jcol = 0; jcol < n; jcol++) {
            if (lower) {
                for (jrow = jcol; jrow < n; jrow++) {
                    work[jrow + n * jcol] -= A[jrow + lda * jcol];
                }
            } else {
                for (jrow = 0; jrow <= jcol; jrow++) {
                    work[jrow + n * jcol] -= A[jrow + lda * jcol];
                }
            }
        }
        wnorm = zlanhe("1", &cuplo, n, work, n, rwork);

    } else if (itype == 3) {
        /*
         * ITYPE=3: error = U V**H - I
         */
        if (n < 2)
            return;

        zlacpy(" ", n, n, U, ldu, work, n);
        if (lower) {
            zunm2r("R", "C", n, n - 1, n - 1, &V[1], ldv, tau,
                   &work[n], n, &work[n * n], &iinfo);
        } else {
            zunm2l("R", "C", n, n - 1, n - 1, &V[ldv], ldv, tau,
                   work, n, &work[n * n], &iinfo);
        }
        if (iinfo != 0) {
            result[0] = TEN / ulp;
            return;
        }

        /* Subtract I from diagonal */
        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        wnorm = zlange("1", n, n, work, n, rwork);
    } else {
        wnorm = ZERO;
    }

    /* Compute result[0] */
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

    /* Test 2: U U**H - I (only for itype=1) */
    if (itype == 1) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= CONE;
        }

        f64 tmp = zlange("1", n, n, work, n, rwork);
        result[1] = fmin((f64)n, tmp) / (n * ulp);
    }
}
