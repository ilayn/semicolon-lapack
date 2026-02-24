/**
 * @file zget52.c
 * @brief ZGET52 does an eigenvector check for the generalized eigenvalue problem.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

void zget52(const INT left, const INT n,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* E, const INT lde,
            const c128* alpha, const c128* beta,
            c128* work, f64* rwork, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    f64 safmin = dlamch("Safe minimum");
    f64 safmax = ONE / safmin;
    f64 ulp = dlamch("Epsilon") * dlamch("Base");

    enum CBLAS_TRANSPOSE trans;
    const char* normab;
    if (left) {
        trans = CblasConjTrans;
        normab = "I";
    } else {
        trans = CblasNoTrans;
        normab = "O";
    }

    /* Norm of A, B, and E: */
    f64 anorm = fmax(zlange(normab, n, n, A, lda, rwork), safmin);
    f64 bnorm = fmax(zlange(normab, n, n, B, ldb, rwork), safmin);
    f64 enorm = fmax(zlange("O", n, n, E, lde, rwork), ulp);
    f64 alfmax = safmax / fmax(ONE, bnorm);
    f64 betmax = safmax / fmax(ONE, anorm);

    /*
     * Compute error matrix.
     * Column i = ( b(i) A - a(i) B ) E(i) / max( |a(i) B|, |b(i) A| )
     */
    for (INT jvec = 0; jvec < n; jvec++) {
        c128 alphai = alpha[jvec];
        c128 betai = beta[jvec];
        f64 abmax = fmax(cabs1(alphai), cabs1(betai));
        if (cabs1(alphai) > alfmax || cabs1(betai) > betmax ||
            abmax < ONE) {
            f64 scale = ONE / fmax(abmax, safmin);
            alphai = scale * alphai;
            betai = scale * betai;
        }
        f64 scale = ONE / fmax(cabs1(alphai) * bnorm,
                           fmax(cabs1(betai) * anorm, safmin));
        c128 acoeff = scale * betai;
        c128 bcoeff = scale * alphai;
        if (left) {
            acoeff = conj(acoeff);
            bcoeff = conj(bcoeff);
        }
        c128 neg_bcoeff = -bcoeff;
        cblas_zgemv(CblasColMajor, trans, n, n, &acoeff, A, lda,
                    &E[lde * jvec], 1, &CZERO,
                    &work[n * jvec], 1);
        cblas_zgemv(CblasColMajor, trans, n, n, &neg_bcoeff, B, ldb,
                    &E[lde * jvec], 1, &CONE,
                    &work[n * jvec], 1);
    }

    f64 errnrm = zlange("One", n, n, work, n, rwork) / enorm;

    /* Compute RESULT(1) */
    result[0] = errnrm / ulp;

    /* Normalization of E: */
    f64 enrmer = ZERO;
    for (INT jvec = 0; jvec < n; jvec++) {
        f64 temp1 = ZERO;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, cabs1(E[j + lde * jvec]));
        }
        enrmer = fmax(enrmer, fabs(temp1 - ONE));
    }

    /* Compute RESULT(2) : the normalization error in E. */
    result[1] = enrmer / ((f64)n * ulp);
}
