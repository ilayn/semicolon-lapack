/**
 * @file ztrsna.c
 * @brief ZTRSNA estimates condition numbers for eigenvalues and eigenvectors.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZTRSNA estimates reciprocal condition numbers for specified
 * eigenvalues and/or right eigenvectors of a complex upper triangular
 * matrix T (or of any matrix Q*T*Q**H with Q unitary).
 *
 * @param[in] job     Specifies whether condition numbers are required for
 *                    eigenvalues (S) or eigenvectors (SEP):
 *                    = 'E': for eigenvalues only (S);
 *                    = 'V': for eigenvectors only (SEP);
 *                    = 'B': for both eigenvalues and eigenvectors (S and SEP).
 * @param[in] howmny  = 'A': compute condition numbers for all eigenpairs;
 *                    = 'S': compute condition numbers for selected eigenpairs
 *                           specified by the array select.
 * @param[in] select  Integer array, dimension (n). If howmny = 'S', select
 *                    specifies the eigenpairs for which condition numbers are
 *                    required (nonzero = selected). If howmny = 'A', not referenced.
 * @param[in] n       The order of the matrix T. n >= 0.
 * @param[in] T       Complex array, dimension (ldt, n).
 *                    The upper triangular matrix T.
 * @param[in] ldt     The leading dimension of T. ldt >= max(1, n).
 * @param[in] VL      Complex array, dimension (ldvl, m).
 *                    If job = 'E' or 'B', left eigenvectors of T.
 *                    If job = 'V', VL is not referenced.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1;
 *                    if job = 'E' or 'B', ldvl >= n.
 * @param[in] VR      Complex array, dimension (ldvr, m).
 *                    If job = 'E' or 'B', right eigenvectors of T.
 *                    If job = 'V', VR is not referenced.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1;
 *                    if job = 'E' or 'B', ldvr >= n.
 * @param[out] S      Double precision array, dimension (mm).
 *                    If job = 'E' or 'B', the reciprocal condition numbers
 *                    of the selected eigenvalues.
 *                    If job = 'V', S is not referenced.
 * @param[out] sep    Double precision array, dimension (mm).
 *                    If job = 'V' or 'B', the estimated reciprocal condition
 *                    numbers of the selected eigenvectors.
 *                    If job = 'E', sep is not referenced.
 * @param[in] mm      The number of elements in the arrays S and/or sep. mm >= m.
 * @param[out] m      The number of elements of S and/or sep actually used.
 * @param[out] work   Complex array, dimension (ldwork, n+6).
 *                    If job = 'E', work is not referenced.
 * @param[in] ldwork  The leading dimension of work. ldwork >= 1;
 *                    if job = 'V' or 'B', ldwork >= n.
 * @param[out] rwork  Double precision array, dimension (n).
 *                    If job = 'E', rwork is not referenced.
 * @param[out] info
 *                    - = 0: successful exit
 *                    - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztrsna(const char* job, const char* howmny, const int* select,
            const int n, const c128* T, const int ldt,
            const c128* VL, const int ldvl,
            const c128* VR, const int ldvr,
            f64* S, f64* sep, const int mm, int* m,
            c128* work, const int ldwork,
            f64* rwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int wantbh, wants, wantsp, somcon;
    int i, ierr, ix, j, k, kase, ks;
    f64 eps, est, lnrm, rnrm, scale, smlnum, xnorm;
    c128 prod;
    char normin;
    int isave[3];
    c128 dummy[1];

    /* Decode and test the input parameters */
    wantbh = (job[0] == 'B' || job[0] == 'b');
    wants = (job[0] == 'E' || job[0] == 'e') || wantbh;
    wantsp = (job[0] == 'V' || job[0] == 'v') || wantbh;

    somcon = (howmny[0] == 'S' || howmny[0] == 's');

    /* Set M to the number of eigenpairs for which condition numbers are
     * to be computed. */
    if (somcon) {
        *m = 0;
        for (j = 0; j < n; j++) {
            if (select[j])
                (*m)++;
        }
    } else {
        *m = n;
    }

    *info = 0;
    if (!wants && !wantsp) {
        *info = -1;
    } else if (!(howmny[0] == 'A' || howmny[0] == 'a') && !somcon) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldvl < 1 || (wants && ldvl < n)) {
        *info = -8;
    } else if (ldvr < 1 || (wants && ldvr < n)) {
        *info = -10;
    } else if (mm < *m) {
        *info = -13;
    } else if (ldwork < 1 || (wantsp && ldwork < n)) {
        *info = -16;
    }
    if (*info != 0) {
        xerbla("ZTRSNA", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    if (n == 1) {
        if (somcon) {
            if (!select[0])
                return;
        }
        if (wants)
            S[0] = ONE;
        if (wantsp)
            sep[0] = cabs(T[0]);
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;

    ks = 0;
    for (k = 0; k < n; k++) {

        if (somcon) {
            if (!select[k])
                continue;
        }

        if (wants) {

            /* Compute the reciprocal condition number of the k-th
             * eigenvalue. */
            cblas_zdotc_sub(n, &VR[ks * ldvr], 1, &VL[ks * ldvl], 1, &prod);
            rnrm = cblas_dznrm2(n, &VR[ks * ldvr], 1);
            lnrm = cblas_dznrm2(n, &VL[ks * ldvl], 1);
            S[ks] = cabs(prod) / (rnrm * lnrm);
        }

        if (wantsp) {

            /* Estimate the reciprocal condition number of the k-th
             * eigenvector.
             *
             * Copy the matrix T to the array WORK and swap the k-th
             * diagonal element to the (1,1) position. */
            zlacpy("F", n, n, T, ldt, work, ldwork);
            ztrexc("N", n, work, ldwork, dummy, 1, k, 0, &ierr);

            /* Form  C = T22 - lambda*I in WORK(2:N,2:N). */
            for (i = 1; i < n; i++) {
                work[i + i * ldwork] = work[i + i * ldwork] - work[0];
            }

            /* Estimate a lower bound for the 1-norm of inv(C**H). The 1st
             * and (N+1)th columns of WORK are used to store work vectors. */
            sep[ks] = ZERO;
            est = ZERO;
            kase = 0;
            normin = 'N';
            for (;;) {
                zlacn2(n - 1, &work[n * ldwork], work, &est, &kase, isave);

                if (kase == 0)
                    break;

                if (kase == 1) {
                    /* Solve C**H*x = scale*b */
                    zlatrs("U", "C", "N", &normin, n - 1,
                           &work[1 + 1 * ldwork], ldwork,
                           work, &scale, rwork, &ierr);
                } else {
                    /* Solve C*x = scale*b */
                    zlatrs("U", "N", "N", &normin, n - 1,
                           &work[1 + 1 * ldwork], ldwork,
                           work, &scale, rwork, &ierr);
                }
                normin = 'Y';
                if (scale != ONE) {

                    /* Multiply by 1/SCALE if doing so will not cause
                     * overflow. */
                    ix = cblas_izamax(n - 1, work, 1);
                    xnorm = cabs1(work[ix]);
                    if (scale < xnorm * smlnum || scale == ZERO)
                        goto L40;
                    zdrscl(n, scale, work, 1);
                }
            }

            sep[ks] = ONE / (est > smlnum ? est : smlnum);
        }

L40:
        ks++;
    }
}
