/**
 * @file dget52.c
 * @brief DGET52 does an eigenvector check for the generalized eigenvalue problem.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * DGET52 does an eigenvector check for the generalized eigenvalue
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
 * DGET52 also tests the normalization of E.  Each eigenvector is
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
void dget52(const INT left, const INT n,
            const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64* E, const INT lde,
            const f64* alphar, const f64* alphai, const f64* beta,
            f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;

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
        trans = CblasTrans;
        normab = "I";
    } else {
        trans = CblasNoTrans;
        normab = "O";
    }

    f64 anorm = fmax(dlange(normab, n, n, A, lda, work), safmin);
    f64 bnorm = fmax(dlange(normab, n, n, B, ldb, work), safmin);
    f64 enorm = fmax(dlange("O", n, n, E, lde, work), ulp);
    f64 alfmax = safmax / fmax(ONE, bnorm);
    f64 betmax = safmax / fmax(ONE, anorm);

    INT ilcplx = 0;
    for (INT jvec = 0; jvec < n; jvec++) {
        if (ilcplx) {
            ilcplx = 0;
        } else {
            f64 salfr = alphar[jvec];
            f64 salfi = alphai[jvec];
            f64 sbeta = beta[jvec];
            if (salfi == ZERO) {
                /* Real eigenvalue and -vector */
                f64 abmax = fmax(fabs(salfr), fabs(sbeta));
                if (fabs(salfr) > alfmax || fabs(sbeta) > betmax ||
                    abmax < ONE) {
                    f64 scale = ONE / fmax(abmax, safmin);
                    salfr = scale * salfr;
                    sbeta = scale * sbeta;
                }
                f64 scale = ONE / fmax(fabs(salfr) * bnorm,
                                  fmax(fabs(sbeta) * anorm, safmin));
                f64 acoef = scale * sbeta;
                f64 bcoefr = scale * salfr;
                cblas_dgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * jvec], 1, ZERO,
                            &work[n * jvec], 1);
                cblas_dgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * jvec], 1);
            } else {
                /* Complex conjugate pair */
                ilcplx = 1;
                if (jvec == n - 1) {
                    result[0] = TEN / ulp;
                    return;
                }
                f64 abmax = fmax(fabs(salfr) + fabs(salfi), fabs(sbeta));
                if (fabs(salfr) + fabs(salfi) > alfmax ||
                    fabs(sbeta) > betmax || abmax < ONE) {
                    f64 scale = ONE / fmax(abmax, safmin);
                    salfr = scale * salfr;
                    salfi = scale * salfi;
                    sbeta = scale * sbeta;
                }
                f64 scale = ONE / fmax((fabs(salfr) + fabs(salfi)) * bnorm,
                                  fmax(fabs(sbeta) * anorm, safmin));
                f64 acoef = scale * sbeta;
                f64 bcoefr = scale * salfr;
                f64 bcoefi = scale * salfi;
                if (left) {
                    bcoefi = -bcoefi;
                }

                cblas_dgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * jvec], 1, ZERO,
                            &work[n * jvec], 1);
                cblas_dgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * jvec], 1);
                cblas_dgemv(CblasColMajor, trans, n, n, bcoefi, B, ldb,
                            &E[lde * (jvec + 1)], 1, ONE,
                            &work[n * jvec], 1);

                cblas_dgemv(CblasColMajor, trans, n, n, acoef, A, lda,
                            &E[lde * (jvec + 1)], 1, ZERO,
                            &work[n * (jvec + 1)], 1);
                cblas_dgemv(CblasColMajor, trans, n, n, -bcoefi, B, ldb,
                            &E[lde * jvec], 1, ONE,
                            &work[n * (jvec + 1)], 1);
                cblas_dgemv(CblasColMajor, trans, n, n, -bcoefr, B, ldb,
                            &E[lde * (jvec + 1)], 1, ONE,
                            &work[n * (jvec + 1)], 1);
            }
        }
    }

    f64 errnrm = dlange("One", n, n, work, n, &work[n * n]) / enorm;

    result[0] = errnrm / ulp;

    /* Normalization of E */
    f64 enrmer = ZERO;
    ilcplx = 0;
    for (INT jvec = 0; jvec < n; jvec++) {
        if (ilcplx) {
            ilcplx = 0;
        } else {
            f64 temp1 = ZERO;
            if (alphai[jvec] == ZERO) {
                for (INT j = 0; j < n; j++) {
                    temp1 = fmax(temp1, fabs(E[j + lde * jvec]));
                }
                enrmer = fmax(enrmer, fabs(temp1 - ONE));
            } else {
                ilcplx = 1;
                for (INT j = 0; j < n; j++) {
                    temp1 = fmax(temp1, fabs(E[j + lde * jvec]) +
                                       fabs(E[j + lde * (jvec + 1)]));
                }
                enrmer = fmax(enrmer, fabs(temp1 - ONE));
            }
        }
    }

    result[1] = enrmer / ((f64)n * ulp);
}
