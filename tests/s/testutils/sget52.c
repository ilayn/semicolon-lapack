/**
 * @file sget52.c
 * @brief SGET52 does an eigenvector check for the generalized eigenvalue problem.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * SGET52 does an eigenvector check for the generalized eigenvalue
 * problem.
 *
 * The basic test for right eigenvectors is:
 *
 *                           | b(j) A E(j) -  a(j) B E(j) |
 *         RESULT(1) = max   -------------------------------
 *                      j    n ulp max( |b(j) A|, |a(j) B| )
 *
 * using the 1-norm.  Here, a(j)/b(j) = w is the j-th generalized
 * eigenvalue of A - w B, or, equivalently, b(j)/a(j) = m is the j-th
 * generalized eigenvalue of m A - B.
 *
 * For left eigenvectors, A', B', a-bar, and b-bar are used.
 *
 * SGET52 also tests the normalization of E.  Each eigenvector is
 * supposed to be normalized so that the maximum "absolute value"
 * of its elements is 1, where "absolute value" of a complex value x
 * is |Re(x)| + |Im(x)|.
 *
 *         RESULT(2) =      max       | M(v(j)) - 1 | / ( n ulp )
 *                    eigenvectors v(j)
 *
 * @param[in]     left    If nonzero, test left eigenvectors; otherwise right.
 * @param[in]     n       The size of the matrices. n >= 0.
 * @param[in]     A       Matrix A, dimension (lda, n).
 * @param[in]     lda     Leading dimension of A.
 * @param[in]     B       Matrix B, dimension (ldb, n).
 * @param[in]     ldb     Leading dimension of B.
 * @param[in]     E       Matrix of eigenvectors, dimension (lde, n).
 * @param[in]     lde     Leading dimension of E.
 * @param[in]     alphar  Real parts of eigenvalue numerators, dimension (n).
 * @param[in]     alphai  Imaginary parts of eigenvalue numerators, dimension (n).
 * @param[in]     beta    Eigenvalue denominators, dimension (n).
 * @param[out]    work    Workspace, dimension (n*n + n).
 * @param[out]    result  Array of dimension (2). Test results.
 */
void sget52(const INT left, const INT n,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32* E, const INT lde,
            const f32* alphar, const f32* alphai, const f32* beta,
            f32* work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TEN = 10.0f;

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
        trans = CblasTrans;
        normab = "I";
    } else {
        trans = CblasNoTrans;
        normab = "O";
    }

    f32 anorm = fmaxf(slange(normab, n, n, A, lda, work), safmin);
    f32 bnorm = fmaxf(slange(normab, n, n, B, ldb, work), safmin);
    f32 enorm = fmaxf(slange("O", n, n, E, lde, work), ulp);
    f32 alfmax = safmax / fmaxf(ONE, bnorm);
    f32 betmax = safmax / fmaxf(ONE, anorm);

    INT ilcplx = 0;
    for (INT jvec = 0; jvec < n; jvec++) {
        if (ilcplx) {
            ilcplx = 0;
        } else {
            f32 salfr = alphar[jvec];
            f32 salfi = alphai[jvec];
            f32 sbeta = beta[jvec];
            if (salfi == ZERO) {
                /* Real eigenvalue and -vector */
                f32 abmax = fmaxf(fabsf(salfr), fabsf(sbeta));
                if (fabsf(salfr) > alfmax || fabsf(sbeta) > betmax ||
                    abmax < ONE) {
                    f32 scale = ONE / fmaxf(abmax, safmin);
                    salfr = scale * salfr;
                    sbeta = scale * sbeta;
                }
                f32 scale = ONE / fmaxf(fabsf(salfr) * bnorm,
                                  fmaxf(fabsf(sbeta) * anorm, safmin));
                f32 acoef = scale * sbeta;
                f32 bcoefr = scale * salfr;
                cblas_sgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * jvec], 1, ZERO,
                            &work[n * jvec], 1);
                cblas_sgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * jvec], 1);
            } else {
                /* Complex conjugate pair */
                ilcplx = 1;
                if (jvec == n - 1) {
                    result[0] = TEN / ulp;
                    return;
                }
                f32 abmax = fmaxf(fabsf(salfr) + fabsf(salfi), fabsf(sbeta));
                if (fabsf(salfr) + fabsf(salfi) > alfmax ||
                    fabsf(sbeta) > betmax || abmax < ONE) {
                    f32 scale = ONE / fmaxf(abmax, safmin);
                    salfr = scale * salfr;
                    salfi = scale * salfi;
                    sbeta = scale * sbeta;
                }
                f32 scale = ONE / fmaxf((fabsf(salfr) + fabsf(salfi)) * bnorm,
                                  fmaxf(fabsf(sbeta) * anorm, safmin));
                f32 acoef = scale * sbeta;
                f32 bcoefr = scale * salfr;
                f32 bcoefi = scale * salfi;
                if (left) {
                    bcoefi = -bcoefi;
                }

                cblas_sgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * jvec], 1, ZERO,
                            &work[n * jvec], 1);
                cblas_sgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * jvec], 1);
                cblas_sgemv(CblasColMajor, trans, n, n, bcoefi, B, ldb,
                            &E[lde * (jvec + 1)], 1, ONE,
                            &work[n * jvec], 1);

                cblas_sgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * (jvec + 1)], 1, ZERO,
                            &work[n * (jvec + 1)], 1);
                cblas_sgemv(CblasColMajor, trans, n, n, -bcoefi, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * (jvec + 1)], 1);
                cblas_sgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * (jvec + 1)], 1, ONE,
                            &work[n * (jvec + 1)], 1);
            }
        }
    }

    f32 errnrm = slange("One", n, n, work, n, &work[n * n]) / enorm;

    result[0] = errnrm / ulp;

    /* Normalization of E */
    f32 enrmer = ZERO;
    ilcplx = 0;
    for (INT jvec = 0; jvec < n; jvec++) {
        if (ilcplx) {
            ilcplx = 0;
        } else {
            f32 temp1 = ZERO;
            if (alphai[jvec] == ZERO) {
                for (INT j = 0; j < n; j++) {
                    temp1 = fmaxf(temp1, fabsf(E[j + lde * jvec]));
                }
                enrmer = fmaxf(enrmer, fabsf(temp1 - ONE));
            } else {
                ilcplx = 1;
                for (INT j = 0; j < n; j++) {
                    temp1 = fmaxf(temp1, fabsf(E[j + lde * jvec]) +
                                       fabsf(E[j + lde * (jvec + 1)]));
                }
                enrmer = fmaxf(enrmer, fabsf(temp1 - ONE));
            }
        }
    }

    result[1] = enrmer / ((f32)n * ulp);
}
