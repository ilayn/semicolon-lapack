/**
 * @file cget52.c
 * @brief CGET52 does an eigenvector check for the generalized eigenvalue problem.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

void cget52(const INT left, const INT n,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            const c64* E, const INT lde,
            const c64* alpha, const c64* beta,
            c64* work, f32* rwork, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    f32 safmin = slamch("Safe minimum");
    f32 safmax = ONE / safmin;
    f32 ulp = slamch("Epsilon") * slamch("Base");

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
    f32 anorm = fmaxf(clange(normab, n, n, A, lda, rwork), safmin);
    f32 bnorm = fmaxf(clange(normab, n, n, B, ldb, rwork), safmin);
    f32 enorm = fmaxf(clange("O", n, n, E, lde, rwork), ulp);
    f32 alfmax = safmax / fmaxf(ONE, bnorm);
    f32 betmax = safmax / fmaxf(ONE, anorm);

    /*
     * Compute error matrix.
     * Column i = ( b(i) A - a(i) B ) E(i) / max( |a(i) B|, |b(i) A| )
     */
    for (INT jvec = 0; jvec < n; jvec++) {
        c64 alphai = alpha[jvec];
        c64 betai = beta[jvec];
        f32 abmax = fmaxf(cabs1f(alphai), cabs1f(betai));
        if (cabs1f(alphai) > alfmax || cabs1f(betai) > betmax ||
            abmax < ONE) {
            f32 scale = ONE / fmaxf(abmax, safmin);
            alphai = scale * alphai;
            betai = scale * betai;
        }
        f32 scale = ONE / fmaxf(cabs1f(alphai) * bnorm,
                           fmaxf(cabs1f(betai) * anorm, safmin));
        c64 acoeff = scale * betai;
        c64 bcoeff = scale * alphai;
        if (left) {
            acoeff = conjf(acoeff);
            bcoeff = conjf(bcoeff);
        }
        c64 neg_bcoeff = -bcoeff;
        cblas_cgemv(CblasColMajor, trans, n, n, &acoeff, A, lda,
                    &E[lde * jvec], 1, &CZERO,
                    &work[n * jvec], 1);
        cblas_cgemv(CblasColMajor, trans, n, n, &neg_bcoeff, B, ldb,
                    &E[lde * jvec], 1, &CONE,
                    &work[n * jvec], 1);
    }

    f32 errnrm = clange("One", n, n, work, n, rwork) / enorm;

    /* Compute RESULT(1) */
    result[0] = errnrm / ulp;

    /* Normalization of E: */
    f32 enrmer = ZERO;
    for (INT jvec = 0; jvec < n; jvec++) {
        f32 temp1 = ZERO;
        for (INT j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, cabs1f(E[j + lde * jvec]));
        }
        enrmer = fmaxf(enrmer, fabsf(temp1 - ONE));
    }

    /* Compute RESULT(2) : the normalization error in E. */
    result[1] = enrmer / ((f32)n * ulp);
}
