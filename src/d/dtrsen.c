/**
 * @file dtrsen.c
 * @brief DTRSEN reorders the real Schur factorization and computes condition numbers.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <stddef.h>

/**
 * DTRSEN reorders the real Schur factorization of a real matrix
 * A = Q*T*Q**T, so that a selected cluster of eigenvalues appears in
 * the leading diagonal blocks of the upper quasi-triangular matrix T,
 * and the leading columns of Q form an orthonormal basis of the
 * corresponding right invariant subspace.
 *
 * Optionally the routine computes the reciprocal condition numbers of
 * the cluster of eigenvalues and/or the invariant subspace.
 *
 * T must be in Schur canonical form (as returned by DHSEQR), that is,
 * block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 * 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.
 *
 * @param[in] job    Specifies whether condition numbers are required:
 *                   = 'N': none;
 *                   = 'E': for eigenvalues only (S);
 *                   = 'V': for invariant subspace only (SEP);
 *                   = 'B': for both eigenvalues and invariant subspace.
 * @param[in] compq  = 'V': update the matrix Q of Schur vectors;
 *                   = 'N': do not update Q.
 * @param[in] select Array of length n. SELECT specifies the eigenvalues in
 *                   the selected cluster. To select a real eigenvalue w(j),
 *                   SELECT(j) must be nonzero. To select a complex conjugate
 *                   pair, either SELECT(j) or SELECT(j+1) or both must be
 *                   nonzero.
 * @param[in] n      The order of the matrix T. n >= 0.
 * @param[in,out] T  On entry, the upper quasi-triangular matrix T, in Schur
 *                   canonical form. On exit, T is overwritten by the reordered
 *                   matrix. Dimension (ldt, n).
 * @param[in] ldt    The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] Q  On entry, if compq = 'V', the matrix Q of Schur vectors.
 *                   On exit, if compq = 'V', Q has been postmultiplied by
 *                   the orthogonal transformation matrix. Dimension (ldq, n).
 *                   If compq = 'N', Q is not referenced.
 * @param[in] ldq    The leading dimension of Q. ldq >= 1, and if compq = 'V',
 *                   ldq >= n.
 * @param[out] wr    Array of length n. Real parts of the reordered eigenvalues.
 * @param[out] wi    Array of length n. Imaginary parts of the reordered eigenvalues.
 * @param[out] m     The dimension of the specified invariant subspace.
 * @param[out] s     If job = 'E' or 'B', a lower bound on the reciprocal
 *                   condition number for the selected cluster of eigenvalues.
 *                   If job = 'N' or 'V', s is not referenced.
 * @param[out] sep   If job = 'V' or 'B', the estimated reciprocal condition
 *                   number of the specified invariant subspace.
 *                   If job = 'N' or 'E', sep is not referenced.
 * @param[out] work  Workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork  The dimension of work.
 *                   If job = 'N', lwork >= max(1, n);
 *                   if job = 'E', lwork >= max(1, m*(n-m));
 *                   if job = 'V' or 'B', lwork >= max(1, 2*m*(n-m)).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] iwork Integer workspace array, dimension (max(1, liwork)).
 *                   On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in] liwork The dimension of iwork.
 *                   If job = 'N' or 'E', liwork >= 1;
 *                   if job = 'V' or 'B', liwork >= max(1, m*(n-m)).
 *                   If liwork = -1, a workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: reordering of T failed because some eigenvalues are
 *                           too close to separate; T may have been partially
 *                           reordered, and wr and wi contain the eigenvalues in
 *                           the same order as in T; s and sep are set to zero.
 */
void dtrsen(const char* job, const char* compq, const int* select,
            const int n, f64* T, const int ldt,
            f64* Q, const int ldq,
            f64* wr, f64* wi, int* m,
            f64* s, f64* sep,
            f64* work, const int lwork,
            int* iwork, const int liwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int wantbh, wants, wantsp, wantq, lquery;
    int ierr, k, kase, kk, ks, lwmin, liwmin, n1, n2, nn;
    int pair, swap;
    f64 est, rnorm, scale;
    int isave[3];

    /* Decode and test the input parameters */
    wantbh = (job[0] == 'B' || job[0] == 'b');
    wants = (job[0] == 'E' || job[0] == 'e') || wantbh;
    wantsp = (job[0] == 'V' || job[0] == 'v') || wantbh;
    wantq = (compq[0] == 'V' || compq[0] == 'v');

    *info = 0;
    lquery = (lwork == -1) || (liwork == -1);

    if (!(job[0] == 'N' || job[0] == 'n') && !wants && !wantsp) {
        *info = -1;
    } else if (!(compq[0] == 'N' || compq[0] == 'n') && !wantq) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -8;
    }

    if (*info == 0) {
        /*
         * Set M to the dimension of the specified invariant subspace,
         * and test LWORK and LIWORK.
         */
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

        n1 = *m;
        n2 = n - *m;
        nn = n1 * n2;

        if (wantsp) {
            lwmin = (1 > 2 * nn) ? 1 : 2 * nn;
            liwmin = (1 > nn) ? 1 : nn;
        } else if (job[0] == 'N' || job[0] == 'n') {
            lwmin = (1 > n) ? 1 : n;
            liwmin = 1;
        } else if (job[0] == 'E' || job[0] == 'e') {
            lwmin = (1 > nn) ? 1 : nn;
            liwmin = 1;
        } else {
            /* Default case - shouldn't reach here if validation is correct */
            lwmin = 1;
            liwmin = 1;
        }

        if (lwork < lwmin && !lquery) {
            *info = -15;
        } else if (liwork < liwmin && !lquery) {
            *info = -17;
        }
    }

    if (*info == 0) {
        work[0] = (f64)lwmin;
        iwork[0] = liwmin;
    }

    if (*info != 0) {
        xerbla("DTRSEN", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible. */
    if (*m == n || *m == 0) {
        if (wants)
            *s = ONE;
        if (wantsp)
            *sep = dlange("1", n, n, T, ldt, work);
        goto L40;
    }

    /* Collect the selected blocks at the top-left corner of T. */
    ks = 0;
    pair = 0;
    for (k = 0; k < n; k++) {
        if (pair) {
            pair = 0;
        } else {
            swap = select[k];
            if (k < n - 1) {
                if (T[(k + 1) + k * ldt] != ZERO) {
                    pair = 1;
                    swap = swap || select[k + 1];
                }
            }
            if (swap) {
                ks++;

                /* Swap the K-th block to position KS.
                 * Note: dtrexc uses 1-based indices for ifst and ilst. */
                ierr = 0;
                kk = k + 1;  /* Convert to 1-based */
                if (k + 1 != ks) {
                    dtrexc(compq, n, T, ldt, Q, ldq, &kk, &ks, work, &ierr);
                }
                if (ierr == 1 || ierr == 2) {
                    /* Blocks too close to swap: exit. */
                    *info = 1;
                    if (wants)
                        *s = ZERO;
                    if (wantsp)
                        *sep = ZERO;
                    goto L40;
                }
                if (pair)
                    ks++;
            }
        }
    }

    if (wants) {
        /*
         * Solve Sylvester equation for R:
         *   T11*R - R*T22 = scale*T12
         */
        dlacpy("F", n1, n2, &T[n1 * ldt], ldt, work, n1);
        dtrsyl("N", "N", -1, n1, n2, T, ldt, &T[n1 + n1 * ldt],
               ldt, work, n1, &scale, &ierr);

        /*
         * Estimate the reciprocal of the condition number of the cluster
         * of eigenvalues.
         */
        rnorm = dlange("F", n1, n2, work, n1, NULL);
        if (rnorm == ZERO) {
            *s = ONE;
        } else {
            *s = scale / (sqrt(scale * scale / rnorm + rnorm) * sqrt(rnorm));
        }
    }

    if (wantsp) {
        /* Estimate sep(T11, T22). */
        est = ZERO;
        kase = 0;
        for (;;) {
            dlacn2(nn, &work[nn], work, iwork, &est, &kase, isave);
            if (kase == 0)
                break;
            if (kase == 1) {
                /* Solve  T11*R - R*T22 = scale*X. */
                dtrsyl("N", "N", -1, n1, n2, T, ldt,
                       &T[n1 + n1 * ldt], ldt, work, n1, &scale, &ierr);
            } else {
                /* Solve T11**T*R - R*T22**T = scale*X. */
                dtrsyl("T", "T", -1, n1, n2, T, ldt,
                       &T[n1 + n1 * ldt], ldt, work, n1, &scale, &ierr);
            }
        }

        *sep = scale / est;
    }

L40:
    /* Store the output eigenvalues in WR and WI. */
    for (k = 0; k < n; k++) {
        wr[k] = T[k + k * ldt];
        wi[k] = ZERO;
    }
    for (k = 0; k < n - 1; k++) {
        if (T[(k + 1) + k * ldt] != ZERO) {
            wi[k] = sqrt(fabs(T[k + (k + 1) * ldt])) *
                    sqrt(fabs(T[(k + 1) + k * ldt]));
            wi[k + 1] = -wi[k];
        }
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
