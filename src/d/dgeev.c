/**
 * @file dgeev.c
 * @brief DGEEV computes the eigenvalues and, optionally, eigenvectors
 *        of a general real matrix.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * DGEEV computes for an N-by-N real nonsymmetric matrix A, the
 * eigenvalues and, optionally, the left and/or right eigenvectors.
 *
 * The right eigenvector v(j) of A satisfies
 *                  A * v(j) = lambda(j) * v(j)
 * where lambda(j) is its eigenvalue.
 * The left eigenvector u(j) of A satisfies
 *               u(j)**H * A = lambda(j) * u(j)**H
 * where u(j)**H denotes the conjugate-transpose of u(j).
 *
 * The computed eigenvectors are normalized to have Euclidean norm
 * equal to 1 and largest component real.
 *
 * @param[in] jobvl  = 'N': left eigenvectors of A are not computed;
 *                   = 'V': left eigenvectors of A are computed.
 * @param[in] jobvr  = 'N': right eigenvectors of A are not computed;
 *                   = 'V': right eigenvectors of A are computed.
 * @param[in] n      The order of the matrix A. n >= 0.
 * @param[in,out] A  On entry, the N-by-N matrix A.
 *                   On exit, A has been overwritten. Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, n).
 * @param[out] wr    Array, dimension (n). Real parts of eigenvalues.
 * @param[out] wi    Array, dimension (n). Imaginary parts of eigenvalues.
 *                   Complex conjugate pairs appear consecutively with
 *                   the eigenvalue having positive imaginary part first.
 * @param[out] VL    If jobvl = 'V', the left eigenvectors are stored one
 *                   after another in the columns of VL. Dimension (ldvl, n).
 *                   If jobvl = 'N', VL is not referenced.
 * @param[in] ldvl   The leading dimension of VL. ldvl >= 1;
 *                   if jobvl = 'V', ldvl >= n.
 * @param[out] VR    If jobvr = 'V', the right eigenvectors are stored one
 *                   after another in the columns of VR. Dimension (ldvr, n).
 *                   If jobvr = 'N', VR is not referenced.
 * @param[in] ldvr   The leading dimension of VR. ldvr >= 1;
 *                   if jobvr = 'V', ldvr >= n.
 * @param[out] work  Workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1, 3*n), and
 *                   if jobvl = 'V' or jobvr = 'V', lwork >= 4*n.
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the QR algorithm failed to compute all
 *                           eigenvalues, and no eigenvectors have been computed;
 *                           elements i:n-1 of wr and wi contain eigenvalues which
 *                           have converged.
 */
void dgeev(const char* jobvl, const char* jobvr, const int n,
           f64* A, const int lda,
           f64* wr, f64* wi,
           f64* VL, const int ldvl,
           f64* VR, const int ldvr,
           f64* work, const int lwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int lquery, scalea, wantvl, wantvr;
    char side[2];
    int hswork, i, ibal, ierr, ihi, ilo, itau, iwrk, k;
    int lwork_trevc, maxwrk, minwrk, nout;
    f64 anrm, bignum, cs, cscale, eps, r, scl, smlnum, sn;
    int select[1];  /* Dummy select array for dtrevc3 */
    f64 dum[1];
    int nb_gehrd, nb_orghr;

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    wantvl = (jobvl[0] == 'V' || jobvl[0] == 'v');
    wantvr = (jobvr[0] == 'V' || jobvr[0] == 'v');

    if (!wantvl && !(jobvl[0] == 'N' || jobvl[0] == 'n')) {
        *info = -1;
    } else if (!wantvr && !(jobvr[0] == 'N' || jobvr[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldvl < 1 || (wantvl && ldvl < n)) {
        *info = -9;
    } else if (ldvr < 1 || (wantvr && ldvr < n)) {
        *info = -11;
    }

    /* Compute workspace */
    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            nb_gehrd = lapack_get_nb("GEHRD");
            nb_orghr = lapack_get_nb("ORGHR");
            if (nb_orghr == 1) nb_orghr = 32;  /* Default for ORGHR from ilaenv */

            maxwrk = 2 * n + n * nb_gehrd;
            if (wantvl) {
                minwrk = 4 * n;
                maxwrk = maxwrk > (2 * n + (n - 1) * nb_orghr) ?
                         maxwrk : (2 * n + (n - 1) * nb_orghr);
                /* Query DHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
                dhseqr("S", "V", n, 0, n - 1, A, lda, wr, wi, VL, ldvl,
                       work, -1, &ierr);
                hswork = (int)work[0];
                maxwrk = maxwrk > (n + 1) ? maxwrk : (n + 1);
                maxwrk = maxwrk > (n + hswork) ? maxwrk : (n + hswork);
                /* Query DTREVC3 for workspace */
                dtrevc3("L", "B", select, n, A, lda, VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, &ierr);
                lwork_trevc = (int)work[0];
                maxwrk = maxwrk > (n + lwork_trevc) ? maxwrk : (n + lwork_trevc);
                maxwrk = maxwrk > (4 * n) ? maxwrk : (4 * n);
            } else if (wantvr) {
                minwrk = 4 * n;
                maxwrk = maxwrk > (2 * n + (n - 1) * nb_orghr) ?
                         maxwrk : (2 * n + (n - 1) * nb_orghr);
                /* Query DHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
                dhseqr("S", "V", n, 0, n - 1, A, lda, wr, wi, VR, ldvr,
                       work, -1, &ierr);
                hswork = (int)work[0];
                maxwrk = maxwrk > (n + 1) ? maxwrk : (n + 1);
                maxwrk = maxwrk > (n + hswork) ? maxwrk : (n + hswork);
                /* Query DTREVC3 for workspace */
                dtrevc3("R", "B", select, n, A, lda, VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, &ierr);
                lwork_trevc = (int)work[0];
                maxwrk = maxwrk > (n + lwork_trevc) ? maxwrk : (n + lwork_trevc);
                maxwrk = maxwrk > (4 * n) ? maxwrk : (4 * n);
            } else {
                minwrk = 3 * n;
                /* Query DHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
                dhseqr("E", "N", n, 0, n - 1, A, lda, wr, wi, VR, ldvr,
                       work, -1, &ierr);
                hswork = (int)work[0];
                maxwrk = maxwrk > (n + 1) ? maxwrk : (n + 1);
                maxwrk = maxwrk > (n + hswork) ? maxwrk : (n + hswork);
            }
            maxwrk = maxwrk > minwrk ? maxwrk : minwrk;
        }
        work[0] = (f64)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("DGEEV", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = dlamch("S");
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = dlange("M", n, n, A, lda, dum);
    scalea = 0;
    if (anrm > ZERO && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea)
        dlascl("G", 0, 0, anrm, cscale, n, n, A, lda, &ierr);

    /* Balance the matrix (Workspace: need N) */
    ibal = 0;  /* 0-based index into work */
    dgebal("B", n, A, lda, &ilo, &ihi, &work[ibal], &ierr);

    /* Reduce to upper Hessenberg form (Workspace: need 3*N, prefer 2*N+N*NB) */
    itau = ibal + n;
    iwrk = itau + n;
    dgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    if (wantvl) {
        /* Want left eigenvectors - Copy Householder vectors to VL */
        side[0] = 'L';
        side[1] = '\0';
        dlacpy("L", n, n, A, lda, VL, ldvl);

        /* Generate orthogonal matrix in VL (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB) */
        dorghr(n, ilo, ihi, VL, ldvl, &work[itau], &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VL */
        iwrk = itau;
        dhseqr("S", "V", n, ilo, ihi, A, lda, wr, wi, VL, ldvl,
               &work[iwrk], lwork - iwrk, info);

        if (wantvr) {
            /* Want left and right eigenvectors - Copy Schur vectors to VR */
            side[0] = 'B';
            dlacpy("F", n, n, VL, ldvl, VR, ldvr);
        }

    } else if (wantvr) {
        /* Want right eigenvectors - Copy Householder vectors to VR */
        side[0] = 'R';
        side[1] = '\0';
        dlacpy("L", n, n, A, lda, VR, ldvr);

        /* Generate orthogonal matrix in VR */
        dorghr(n, ilo, ihi, VR, ldvr, &work[itau], &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VR */
        iwrk = itau;
        dhseqr("S", "V", n, ilo, ihi, A, lda, wr, wi, VR, ldvr,
               &work[iwrk], lwork - iwrk, info);

    } else {
        /* Compute eigenvalues only */
        iwrk = itau;
        dhseqr("E", "N", n, ilo, ihi, A, lda, wr, wi, VR, ldvr,
               &work[iwrk], lwork - iwrk, info);
    }

    /* If INFO != 0 from DHSEQR, then quit */
    if (*info != 0)
        goto L50;

    if (wantvl || wantvr) {
        /* Compute left and/or right eigenvectors */
        dtrevc3(side, "B", select, n, A, lda, VL, ldvl, VR, ldvr,
                n, &nout, &work[iwrk], lwork - iwrk, &ierr);
    }

    if (wantvl) {
        /* Undo balancing of left eigenvectors (Workspace: need N) */
        dgebak("B", "L", n, ilo, ihi, &work[ibal], n, VL, ldvl, &ierr);

        /* Normalize left eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            if (wi[i] == ZERO) {
                scl = ONE / cblas_dnrm2(n, &VL[i * ldvl], 1);
                cblas_dscal(n, scl, &VL[i * ldvl], 1);
            } else if (wi[i] > ZERO) {
                scl = ONE / dlapy2(cblas_dnrm2(n, &VL[i * ldvl], 1),
                                   cblas_dnrm2(n, &VL[(i + 1) * ldvl], 1));
                cblas_dscal(n, scl, &VL[i * ldvl], 1);
                cblas_dscal(n, scl, &VL[(i + 1) * ldvl], 1);
                for (k = 0; k < n; k++) {
                    work[iwrk + k] = VL[k + i * ldvl] * VL[k + i * ldvl] +
                                     VL[k + (i + 1) * ldvl] * VL[k + (i + 1) * ldvl];
                }
                k = (int)cblas_idamax(n, &work[iwrk], 1);
                dlartg(VL[k + i * ldvl], VL[k + (i + 1) * ldvl], &cs, &sn, &r);
                cblas_drot(n, &VL[i * ldvl], 1, &VL[(i + 1) * ldvl], 1, cs, sn);
                VL[k + (i + 1) * ldvl] = ZERO;
            }
        }
    }

    if (wantvr) {
        /* Undo balancing of right eigenvectors (Workspace: need N) */
        dgebak("B", "R", n, ilo, ihi, &work[ibal], n, VR, ldvr, &ierr);

        /* Normalize right eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            if (wi[i] == ZERO) {
                scl = ONE / cblas_dnrm2(n, &VR[i * ldvr], 1);
                cblas_dscal(n, scl, &VR[i * ldvr], 1);
            } else if (wi[i] > ZERO) {
                scl = ONE / dlapy2(cblas_dnrm2(n, &VR[i * ldvr], 1),
                                   cblas_dnrm2(n, &VR[(i + 1) * ldvr], 1));
                cblas_dscal(n, scl, &VR[i * ldvr], 1);
                cblas_dscal(n, scl, &VR[(i + 1) * ldvr], 1);
                for (k = 0; k < n; k++) {
                    work[iwrk + k] = VR[k + i * ldvr] * VR[k + i * ldvr] +
                                     VR[k + (i + 1) * ldvr] * VR[k + (i + 1) * ldvr];
                }
                k = (int)cblas_idamax(n, &work[iwrk], 1);
                dlartg(VR[k + i * ldvr], VR[k + (i + 1) * ldvr], &cs, &sn, &r);
                cblas_drot(n, &VR[i * ldvr], 1, &VR[(i + 1) * ldvr], 1, cs, sn);
                VR[k + (i + 1) * ldvr] = ZERO;
            }
        }
    }

    /* Undo scaling if necessary */
L50:
    if (scalea) {
        dlascl("G", 0, 0, cscale, anrm, n - *info, 1, &wr[*info],
               (n - *info) > 1 ? (n - *info) : 1, &ierr);
        dlascl("G", 0, 0, cscale, anrm, n - *info, 1, &wi[*info],
               (n - *info) > 1 ? (n - *info) : 1, &ierr);
        if (*info > 0) {
            dlascl("G", 0, 0, cscale, anrm, ilo - 1, 1, wr, n, &ierr);
            dlascl("G", 0, 0, cscale, anrm, ilo - 1, 1, wi, n, &ierr);
        }
    }

    work[0] = (f64)maxwrk;
}
