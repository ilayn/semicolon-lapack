/**
 * @file dtrsna.c
 * @brief DTRSNA estimates condition numbers for eigenvalues and eigenvectors.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/**
 * DTRSNA estimates reciprocal condition numbers for specified
 * eigenvalues and/or right eigenvectors of a real upper
 * quasi-triangular matrix T (or of any matrix Q*T*Q**T with Q
 * orthogonal).
 *
 * T must be in Schur canonical form (as returned by DHSEQR), that is,
 * block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 * 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.
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
 * @param[in] T       The upper quasi-triangular matrix T, in Schur canonical form.
 *                    Dimension (ldt, n).
 * @param[in] ldt     The leading dimension of T. ldt >= max(1, n).
 * @param[in] VL      If job = 'E' or 'B', left eigenvectors of T.
 *                    Dimension (ldvl, m). If job = 'V', VL is not referenced.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1;
 *                    if job = 'E' or 'B', ldvl >= n.
 * @param[in] VR      If job = 'E' or 'B', right eigenvectors of T.
 *                    Dimension (ldvr, m). If job = 'V', VR is not referenced.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1;
 *                    if job = 'E' or 'B', ldvr >= n.
 * @param[out] S      Array, dimension (mm). If job = 'E' or 'B', the reciprocal
 *                    condition numbers of the selected eigenvalues.
 *                    If job = 'V', S is not referenced.
 * @param[out] sep    Array, dimension (mm). If job = 'V' or 'B', the estimated
 *                    reciprocal condition numbers of the selected eigenvectors.
 *                    If job = 'E', sep is not referenced.
 * @param[in] mm      The number of elements in the arrays S and/or sep. mm >= m.
 * @param[out] m      The number of elements of S and/or sep actually used.
 * @param[out] work   Workspace array, dimension (ldwork, n+6).
 *                    If job = 'E', work is not referenced.
 * @param[in] ldwork  The leading dimension of work. ldwork >= 1;
 *                    if job = 'V' or 'B', ldwork >= n.
 * @param[out] iwork  Integer array, dimension (2*(n-1)).
 *                    If job = 'E', iwork is not referenced.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dtrsna(const char* job, const char* howmny, const int* select,
            const int n, const f64* T, const int ldt,
            const f64* VL, const int ldvl,
            const f64* VR, const int ldvr,
            f64* S, f64* sep, const int mm, int* m,
            f64* work, const int ldwork,
            int* iwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    int wantbh, wants, wantsp, somcon;
    int i, ierr, ifst, ilst, j, k, kase, ks, n2, nn;
    int pair;
    f64 bignum, cond, cs, delta, dumm = 0.0, eps, est, lnrm;
    f64 mu, prod, prod1, prod2, rnrm, scale, smlnum, sn;
    int isave[3];
    f64 dummy[1];

    /* Decode and test the input parameters */
    wantbh = (job[0] == 'B' || job[0] == 'b');
    wants = (job[0] == 'E' || job[0] == 'e') || wantbh;
    wantsp = (job[0] == 'V' || job[0] == 'v') || wantbh;

    somcon = (howmny[0] == 'S' || howmny[0] == 's');

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
    } else {
        /* Set M to the number of eigenpairs for which condition numbers
         * are required, and test MM. */
        if (somcon) {
            *m = 0;
            pair = 0;
            for (k = 0; k < n; k++) {
                if (pair) {
                    pair = 0;
                } else {
                    if (k < n - 1) {
                        if (T[(k + 1) + k * ldt] == ZERO) {
                            if (select[k])
                                (*m)++;
                        } else {
                            pair = 1;
                            if (select[k] || select[k + 1])
                                *m += 2;
                        }
                    } else {
                        if (select[n - 1])
                            (*m)++;
                    }
                }
            }
        } else {
            *m = n;
        }

        if (mm < *m) {
            *info = -13;
        } else if (ldwork < 1 || (wantsp && ldwork < n)) {
            *info = -16;
        }
    }
    if (*info != 0) {
        xerbla("DTRSNA", -(*info));
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
            sep[0] = fabs(T[0]);
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    bignum = ONE / smlnum;

    ks = -1;  /* Will be incremented before first use */
    pair = 0;
    for (k = 0; k < n; k++) {

        /* Determine whether T(k,k) begins a 1-by-1 or 2-by-2 block. */
        if (pair) {
            pair = 0;
            continue;
        } else {
            if (k < n - 1)
                pair = (T[(k + 1) + k * ldt] != ZERO);
        }

        /* Determine whether condition numbers are required for the k-th
         * eigenpair. */
        if (somcon) {
            if (pair) {
                if (!select[k] && !select[k + 1])
                    continue;
            } else {
                if (!select[k])
                    continue;
            }
        }

        ks++;

        if (wants) {

            /* Compute the reciprocal condition number of the k-th
             * eigenvalue. */

            if (!pair) {

                /* Real eigenvalue. */
                prod = cblas_ddot(n, &VR[ks * ldvr], 1, &VL[ks * ldvl], 1);
                rnrm = cblas_dnrm2(n, &VR[ks * ldvr], 1);
                lnrm = cblas_dnrm2(n, &VL[ks * ldvl], 1);
                S[ks] = fabs(prod) / (rnrm * lnrm);

            } else {

                /* Complex eigenvalue. */
                prod1 = cblas_ddot(n, &VR[ks * ldvr], 1, &VL[ks * ldvl], 1);
                prod1 = prod1 + cblas_ddot(n, &VR[(ks + 1) * ldvr], 1,
                                           &VL[(ks + 1) * ldvl], 1);
                prod2 = cblas_ddot(n, &VL[ks * ldvl], 1, &VR[(ks + 1) * ldvr], 1);
                prod2 = prod2 - cblas_ddot(n, &VL[(ks + 1) * ldvl], 1,
                                           &VR[ks * ldvr], 1);
                rnrm = dlapy2(cblas_dnrm2(n, &VR[ks * ldvr], 1),
                              cblas_dnrm2(n, &VR[(ks + 1) * ldvr], 1));
                lnrm = dlapy2(cblas_dnrm2(n, &VL[ks * ldvl], 1),
                              cblas_dnrm2(n, &VL[(ks + 1) * ldvl], 1));
                cond = dlapy2(prod1, prod2) / (rnrm * lnrm);
                S[ks] = cond;
                S[ks + 1] = cond;
            }
        }

        if (wantsp) {

            /* Estimate the reciprocal condition number of the k-th
             * eigenvector.
             *
             * Copy the matrix T to the array WORK and swap the diagonal
             * block beginning at T(k,k) to the (1,1) position. */
            dlacpy("F", n, n, T, ldt, work, ldwork);
            ifst = k;
            ilst = 0;
            dtrexc("N", n, work, ldwork, dummy, 1, &ifst, &ilst,
                   &work[n * ldwork], &ierr);

            if (ierr == 1 || ierr == 2) {

                /* Could not swap because blocks not well separated */
                scale = ONE;
                est = bignum;

            } else {

                /* Reordering successful */

                if (work[1] == ZERO) {

                    /* Form C = T22 - lambda*I in WORK(2:N,2:N). */
                    for (i = 1; i < n; i++) {
                        work[i + i * ldwork] = work[i + i * ldwork] - work[0];
                    }
                    n2 = 1;
                    nn = n - 1;

                } else {

                    /* Triangularize the 2 by 2 block by unitary
                     * transformation U = [  cs   i*ss ]
                     *                    [ i*ss   cs  ].
                     * such that the (1,1) position of WORK is complex
                     * eigenvalue lambda with positive imaginary part. */
                    mu = sqrt(fabs(work[1 * ldwork])) * sqrt(fabs(work[1]));
                    delta = dlapy2(mu, work[1]);
                    cs = mu / delta;
                    sn = -work[1] / delta;

                    /* Form
                     * C**T = WORK(2:N,2:N) + i*[rwork(1) ..... rwork(n-1) ]
                     *                          [   mu                     ]
                     *                          [         ..               ]
                     *                          [             ..           ]
                     *                          [                  mu      ]
                     * where C**T is transpose of matrix C,
                     * and RWORK is stored starting in the N+1-st column of
                     * WORK. */
                    for (j = 2; j < n; j++) {
                        work[1 + j * ldwork] = cs * work[1 + j * ldwork];
                        work[j + j * ldwork] = work[j + j * ldwork] - work[0];
                    }
                    work[1 + 1 * ldwork] = ZERO;

                    work[n * ldwork] = TWO * mu;
                    for (i = 1; i < n - 1; i++) {
                        work[i + n * ldwork] = sn * work[(i + 1) * ldwork];
                    }
                    n2 = 2;
                    nn = 2 * (n - 1);
                }

                /* Estimate norm(inv(C**T)) */
                est = ZERO;
                kase = 0;
                for (;;) {
                    dlacn2(nn, &work[(n + 1) * ldwork], &work[(n + 3) * ldwork],
                           iwork, &est, &kase, isave);
                    if (kase == 0)
                        break;
                    if (kase == 1) {
                        if (n2 == 1) {
                            /* Real eigenvalue: solve C**T*x = scale*c. */
                            dlaqtr(1, 1, n - 1, &work[1 + ldwork],
                                   ldwork, dummy, dumm, &scale,
                                   &work[(n + 3) * ldwork], &work[(n + 5) * ldwork], &ierr);
                        } else {
                            /* Complex eigenvalue: solve
                             * C**T*(p+iq) = scale*(c+id) in real arithmetic. */
                            dlaqtr(1, 0, n - 1, &work[1 + ldwork],
                                   ldwork, &work[n * ldwork], mu, &scale,
                                   &work[(n + 3) * ldwork], &work[(n + 5) * ldwork], &ierr);
                        }
                    } else {
                        if (n2 == 1) {
                            /* Real eigenvalue: solve C*x = scale*c. */
                            dlaqtr(0, 1, n - 1, &work[1 + ldwork],
                                   ldwork, dummy, dumm, &scale,
                                   &work[(n + 3) * ldwork], &work[(n + 5) * ldwork], &ierr);
                        } else {
                            /* Complex eigenvalue: solve
                             * C*(p+iq) = scale*(c+id) in real arithmetic. */
                            dlaqtr(0, 0, n - 1, &work[1 + ldwork],
                                   ldwork, &work[n * ldwork], mu, &scale,
                                   &work[(n + 3) * ldwork], &work[(n + 5) * ldwork], &ierr);
                        }
                    }
                }
            }

            sep[ks] = scale / fmax(est, smlnum);
            if (pair)
                sep[ks + 1] = sep[ks];
        }

        if (pair)
            ks++;
    }
}
