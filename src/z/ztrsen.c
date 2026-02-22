/**
 * @file ztrsen.c
 * @brief ZTRSEN reorders the Schur factorization and computes condition numbers.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>

/**
 * ZTRSEN reorders the Schur factorization of a complex matrix
 * A = Q*T*Q**H, so that a selected cluster of eigenvalues appears in
 * the leading positions on the diagonal of the upper triangular matrix
 * T, and the leading columns of Q form an orthonormal basis of the
 * corresponding right invariant subspace.
 *
 * Optionally the routine computes the reciprocal condition numbers of
 * the cluster of eigenvalues and/or the invariant subspace.
 *
 * @param[in] job    Specifies whether condition numbers are required:
 *                   = 'N': none;
 *                   = 'E': for eigenvalues only (S);
 *                   = 'V': for invariant subspace only (SEP);
 *                   = 'B': for both eigenvalues and invariant subspace.
 * @param[in] compq  = 'V': update the matrix Q of Schur vectors;
 *                   = 'N': do not update Q.
 * @param[in] select Array of length n. SELECT specifies the eigenvalues in
 *                   the selected cluster. To select the j-th eigenvalue,
 *                   SELECT(j) must be set to nonzero.
 * @param[in] n      The order of the matrix T. n >= 0.
 * @param[in,out] T  Complex array, dimension (ldt, n).
 *                   On entry, the upper triangular matrix T.
 *                   On exit, T is overwritten by the reordered matrix T,
 *                   with the selected eigenvalues as the leading diagonal
 *                   elements.
 * @param[in] ldt    The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] Q  Complex array, dimension (ldq, n).
 *                   On entry, if compq = 'V', the matrix Q of Schur vectors.
 *                   On exit, if compq = 'V', Q has been postmultiplied by
 *                   the unitary transformation matrix which reorders T.
 *                   If compq = 'N', Q is not referenced.
 * @param[in] ldq    The leading dimension of Q. ldq >= 1, and if
 *                   compq = 'V', ldq >= n.
 * @param[out] W     Complex array, dimension (n).
 *                   The reordered eigenvalues of T, in the same order as
 *                   they appear on the diagonal of T.
 * @param[out] m     The dimension of the specified invariant subspace.
 *                   0 <= m <= n.
 * @param[out] s     If job = 'E' or 'B', a lower bound on the reciprocal
 *                   condition number for the selected cluster of eigenvalues.
 *                   If job = 'N' or 'V', s is not referenced.
 * @param[out] sep   If job = 'V' or 'B', the estimated reciprocal condition
 *                   number of the specified invariant subspace.
 *                   If job = 'N' or 'E', sep is not referenced.
 * @param[out] work  Complex workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork  The dimension of work.
 *                   If job = 'N', lwork >= 1;
 *                   if job = 'E', lwork >= max(1, m*(n-m));
 *                   if job = 'V' or 'B', lwork >= max(1, 2*m*(n-m)).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 */
void ztrsen(const char* job, const char* compq, const INT* select,
            const INT n, c128* T, const INT ldt,
            c128* Q, const INT ldq,
            c128* W, INT* m, f64* s, f64* sep,
            c128* work, const INT lwork, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT wantbh, wants, wantsp, wantq, lquery;
    INT ierr, k, kase, ks, lwmin = 0, n1, n2, nn;
    f64 est, rnorm, scale;
    INT isave[3];
    f64 rwork[1];

    /* Decode and test the input parameters. */
    wantbh = (job[0] == 'B' || job[0] == 'b');
    wants = (job[0] == 'E' || job[0] == 'e') || wantbh;
    wantsp = (job[0] == 'V' || job[0] == 'v') || wantbh;
    wantq = (compq[0] == 'V' || compq[0] == 'v');

    /* Set M to the number of selected eigenvalues. */
    *m = 0;
    for (k = 0; k < n; k++) {
        if (select[k])
            (*m)++;
    }

    n1 = *m;
    n2 = n - *m;
    nn = n1 * n2;

    *info = 0;
    lquery = (lwork == -1);

    if (wantsp) {
        lwmin = (1 > 2 * nn) ? 1 : 2 * nn;
    } else if (job[0] == 'N' || job[0] == 'n') {
        lwmin = 1;
    } else if (job[0] == 'E' || job[0] == 'e') {
        lwmin = (1 > nn) ? 1 : nn;
    }

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
    } else if (lwork < lwmin && !lquery) {
        *info = -14;
    }

    if (*info == 0) {
        work[0] = CMPLX((f64)lwmin, 0.0);
    }

    if (*info != 0) {
        xerbla("ZTRSEN", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (*m == n || *m == 0) {
        if (wants)
            *s = ONE;
        if (wantsp)
            *sep = zlange("1", n, n, T, ldt, rwork);
        goto L40;
    }

    /* Collect the selected eigenvalues at the top left corner of T. */
    ks = 0;
    for (k = 0; k < n; k++) {
        if (select[k]) {

            /* Swap the K-th eigenvalue to position KS. */
            if (k != ks) {
                INT ifst = k;
                INT ilst = ks;
                ztrexc(compq, n, T, ldt, Q, ldq, ifst, ilst, &ierr);
            }
            ks++;
        }
    }

    if (wants) {
        /*
         * Solve the Sylvester equation for R:
         *
         *   T11*R - R*T22 = scale*T12
         */
        zlacpy("F", n1, n2, &T[n1 * ldt], ldt, work, n1);
        ztrsyl("N", "N", -1, n1, n2, T, ldt, &T[n1 + n1 * ldt],
               ldt, work, n1, &scale, &ierr);

        /*
         * Estimate the reciprocal of the condition number of the cluster
         * of eigenvalues.
         */
        rnorm = zlange("F", n1, n2, work, n1, rwork);
        if (rnorm == ZERO) {
            *s = ONE;
        } else {
            *s = scale / (sqrt(scale * scale / rnorm + rnorm) *
                 sqrt(rnorm));
        }
    }

    if (wantsp) {
        /* Estimate sep(T11, T22). */
        est = ZERO;
        kase = 0;
        for (;;) {
            zlacn2(nn, &work[nn], work, &est, &kase, isave);
            if (kase == 0)
                break;
            if (kase == 1) {
                /* Solve T11*R - R*T22 = scale*X. */
                ztrsyl("N", "N", -1, n1, n2, T, ldt,
                       &T[n1 + n1 * ldt], ldt, work, n1, &scale,
                       &ierr);
            } else {
                /* Solve T11**H*R - R*T22**H = scale*X. */
                ztrsyl("C", "C", -1, n1, n2, T, ldt,
                       &T[n1 + n1 * ldt], ldt, work, n1, &scale,
                       &ierr);
            }
        }

        *sep = scale / est;
    }

L40:
    /* Copy reordered eigenvalues to W. */
    for (k = 0; k < n; k++) {
        W[k] = T[k + k * ldt];
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
}
