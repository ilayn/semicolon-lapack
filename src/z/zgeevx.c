/**
 * @file zgeevx.c
 * @brief ZGEEVX computes eigenvalues, eigenvectors, and condition numbers.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZGEEVX computes for an N-by-N complex nonsymmetric matrix A, the
 * eigenvalues and, optionally, the left and/or right eigenvectors.
 *
 * Optionally also, it computes a balancing transformation to improve
 * the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
 * SCALE, and ABNRM), reciprocal condition numbers for the eigenvalues
 * (RCONDE), and reciprocal condition numbers for the right
 * eigenvectors (RCONDV).
 *
 * The right eigenvector v(j) of A satisfies
 *                  A * v(j) = lambda(j) * v(j)
 * where lambda(j) is its eigenvalue.
 * The left eigenvector u(j) of A satisfies
 *               u(j)**H * A = lambda(j) * u(j)**H
 * where u(j)**H denotes the conjugate transpose of u(j).
 *
 * The computed eigenvectors are normalized to have Euclidean norm
 * equal to 1 and largest component real.
 *
 * Balancing a matrix means permuting the rows and columns to make it
 * more nearly upper triangular, and applying a diagonal similarity
 * transformation D * A * D**(-1), where D is a diagonal matrix, to
 * make its rows and columns closer in norm and the condition numbers
 * of its eigenvalues and eigenvectors smaller.
 *
 * @param[in] balanc  Indicates how the input matrix should be diagonally scaled
 *                    and/or permuted to improve the conditioning:
 *                    = 'N': Do not diagonally scale or permute;
 *                    = 'P': Perform permutations to make the matrix more nearly
 *                           upper triangular. Do not diagonally scale;
 *                    = 'S': Diagonally scale the matrix, i.e. replace A by
 *                           D*A*D**(-1), where D is a diagonal matrix;
 *                    = 'B': Both diagonally scale and permute A.
 * @param[in] jobvl   = 'N': left eigenvectors of A are not computed;
 *                    = 'V': left eigenvectors of A are computed.
 *                    If sense = 'E' or 'B', jobvl must = 'V'.
 * @param[in] jobvr   = 'N': right eigenvectors of A are not computed;
 *                    = 'V': right eigenvectors of A are computed.
 *                    If sense = 'E' or 'B', jobvr must = 'V'.
 * @param[in] sense   Determines which reciprocal condition numbers are computed:
 *                    = 'N': None are computed;
 *                    = 'E': Computed for eigenvalues only;
 *                    = 'V': Computed for right eigenvectors only;
 *                    = 'B': Computed for eigenvalues and right eigenvectors.
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in,out] A   On entry, the N-by-N matrix A.
 *                    On exit, A has been overwritten. If jobvl = 'V' or
 *                    jobvr = 'V', A contains the Schur form of the balanced
 *                    version of the input matrix A.
 *                    Dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1, n).
 * @param[out] W      Complex array, dimension (n). Contains the computed
 *                    eigenvalues.
 * @param[out] VL     If jobvl = 'V', the left eigenvectors u(j) are stored one
 *                    after another in the columns of VL. Dimension (ldvl, n).
 *                    If jobvl = 'N', VL is not referenced.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1;
 *                    if jobvl = 'V', ldvl >= n.
 * @param[out] VR     If jobvr = 'V', the right eigenvectors v(j) are stored one
 *                    after another in the columns of VR. Dimension (ldvr, n).
 *                    If jobvr = 'N', VR is not referenced.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1;
 *                    if jobvr = 'V', ldvr >= n.
 * @param[out] ilo    Integer value determined when A was balanced.
 * @param[out] ihi    Integer value determined when A was balanced.
 *                    The balanced A(i,j) = 0 if I > J and
 *                    J = 0,...,ilo-2 or I = ihi,...,n-1.
 * @param[out] scale  Array, dimension (n). Details of the permutations and
 *                    scaling factors applied when balancing A.
 * @param[out] abnrm  The one-norm of the balanced matrix.
 * @param[out] rconde Array, dimension (n). Reciprocal condition numbers of
 *                    the eigenvalues.
 * @param[out] rcondv Array, dimension (n). Reciprocal condition numbers of
 *                    the right eigenvectors.
 * @param[out] work   Complex workspace array, dimension (max(1, lwork)).
 *                    On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork   The dimension of work. If sense = 'N' or 'E',
 *                    lwork >= max(1, 2*n), and if sense = 'V' or 'B',
 *                    lwork >= n*n+2*n.
 *                    If lwork = -1, a workspace query is assumed.
 * @param[out] rwork  Double precision array, dimension (2*n).
 * @param[out] info
 *                    - = 0: successful exit
 *                    - < 0: if info = -i, the i-th argument had an illegal value
 *                    - > 0: if info = i, the QR algorithm failed to compute all
 *                      eigenvalues, and no eigenvectors or condition numbers
 *                      have been computed; elements 0:ilo-2 and i:n-1 of W
 *                      contain eigenvalues which have converged.
 */
void zgeevx(const char* balanc, const char* jobvl, const char* jobvr,
            const char* sense, const int n, c128* A, const int lda,
            c128* W,
            c128* VL, const int ldvl,
            c128* VR, const int ldvr,
            int* ilo, int* ihi, f64* scale, f64* abnrm,
            f64* rconde, f64* rcondv,
            c128* work, const int lwork,
            f64* rwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int lquery, scalea, wantvl, wantvr, wntsnb, wntsne, wntsnn, wntsnv;
    int hswork, i, icond, ierr, itau, iwrk, k;
    int lwork_trevc, maxwrk, minwrk, nout;
    f64 anrm, bignum, cscale, eps, scl, smlnum;
    c128 tmp;
    int select[1];
    f64 dum[1];
    const char* side;
    char job_hseqr;
    int nb_gehrd, nb_unghr;

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    wantvl = (jobvl[0] == 'V' || jobvl[0] == 'v');
    wantvr = (jobvr[0] == 'V' || jobvr[0] == 'v');
    wntsnn = (sense[0] == 'N' || sense[0] == 'n');
    wntsne = (sense[0] == 'E' || sense[0] == 'e');
    wntsnv = (sense[0] == 'V' || sense[0] == 'v');
    wntsnb = (sense[0] == 'B' || sense[0] == 'b');

    if (!(balanc[0] == 'N' || balanc[0] == 'n' ||
          balanc[0] == 'S' || balanc[0] == 's' ||
          balanc[0] == 'P' || balanc[0] == 'p' ||
          balanc[0] == 'B' || balanc[0] == 'b')) {
        *info = -1;
    } else if (!wantvl && !(jobvl[0] == 'N' || jobvl[0] == 'n')) {
        *info = -2;
    } else if (!wantvr && !(jobvr[0] == 'N' || jobvr[0] == 'n')) {
        *info = -3;
    } else if (!(wntsnn || wntsne || wntsnb || wntsnv) ||
               ((wntsne || wntsnb) && !(wantvl && wantvr))) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldvl < 1 || (wantvl && ldvl < n)) {
        *info = -10;
    } else if (ldvr < 1 || (wantvr && ldvr < n)) {
        *info = -12;
    }

    /* Compute workspace */
    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            nb_gehrd = lapack_get_nb("GEHRD");
            nb_unghr = lapack_get_nb("ORGHR");
            if (nb_unghr == 1) nb_unghr = 32;

            maxwrk = n + n * nb_gehrd;

            if (wantvl) {
                ztrevc3("L", "B", select, n, A, lda,
                        VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, rwork, -1, &ierr);
                lwork_trevc = (int)creal(work[0]);
                maxwrk = maxwrk > lwork_trevc ? maxwrk : lwork_trevc;
                zhseqr("S", "V", n, 0, n - 1, A, lda, W, VL, ldvl,
                       work, -1, info);
            } else if (wantvr) {
                ztrevc3("R", "B", select, n, A, lda,
                        VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, rwork, -1, &ierr);
                lwork_trevc = (int)creal(work[0]);
                maxwrk = maxwrk > lwork_trevc ? maxwrk : lwork_trevc;
                zhseqr("S", "V", n, 0, n - 1, A, lda, W, VR, ldvr,
                       work, -1, info);
            } else {
                if (wntsnn) {
                    zhseqr("E", "N", n, 0, n - 1, A, lda, W, VR, ldvr,
                           work, -1, info);
                } else {
                    zhseqr("S", "N", n, 0, n - 1, A, lda, W, VR, ldvr,
                           work, -1, info);
                }
            }
            hswork = (int)creal(work[0]);

            if (!wantvl && !wantvr) {
                minwrk = 2 * n;
                if (!(wntsnn || wntsne))
                    minwrk = minwrk > (n * n + 2 * n) ?
                             minwrk : (n * n + 2 * n);
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
                if (!(wntsnn || wntsne))
                    maxwrk = maxwrk > (n * n + 2 * n) ?
                             maxwrk : (n * n + 2 * n);
            } else {
                minwrk = 2 * n;
                if (!(wntsnn || wntsne))
                    minwrk = minwrk > (n * n + 2 * n) ?
                             minwrk : (n * n + 2 * n);
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
                maxwrk = maxwrk > (n + (n - 1) * nb_unghr) ?
                         maxwrk : (n + (n - 1) * nb_unghr);
                if (!(wntsnn || wntsne))
                    maxwrk = maxwrk > (n * n + 2 * n) ?
                             maxwrk : (n * n + 2 * n);
                maxwrk = maxwrk > (2 * n) ? maxwrk : (2 * n);
            }
            maxwrk = maxwrk > minwrk ? maxwrk : minwrk;
        }
        work[0] = CMPLX((f64)maxwrk, 0.0);

        if (lwork < minwrk && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        xerbla("ZGEEVX", -(*info));
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
    icond = 0;
    anrm = zlange("M", n, n, A, lda, dum);
    scalea = 0;
    if (anrm > ZERO && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea)
        zlascl("G", 0, 0, anrm, cscale, n, n, A, lda, &ierr);

    /* Balance the matrix and compute ABNRM */
    zgebal(balanc, n, A, lda, ilo, ihi, scale, &ierr);
    *abnrm = zlange("1", n, n, A, lda, dum);
    if (scalea) {
        dum[0] = *abnrm;
        dlascl("G", 0, 0, cscale, anrm, 1, 1, dum, 1, &ierr);
        *abnrm = dum[0];
    }

    /* Reduce to upper Hessenberg form */
    itau = 0;
    iwrk = itau + n;
    zgehrd(n, *ilo, *ihi, A, lda, &work[itau], &work[iwrk],
           lwork - iwrk, &ierr);

    if (wantvl) {
        /* Want left eigenvectors - Copy Householder vectors to VL */
        side = "L";
        zlacpy("L", n, n, A, lda, VL, ldvl);

        /* Generate unitary matrix in VL */
        zunghr(n, *ilo, *ihi, VL, ldvl, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VL */
        iwrk = itau;
        zhseqr("S", "V", n, *ilo, *ihi, A, lda, W, VL, ldvl,
               &work[iwrk], lwork - iwrk, info);

        if (wantvr) {
            /* Want left and right eigenvectors - Copy Schur vectors to VR */
            side = "B";
            zlacpy("F", n, n, VL, ldvl, VR, ldvr);
        }

    } else if (wantvr) {
        /* Want right eigenvectors - Copy Householder vectors to VR */
        side = "R";
        zlacpy("L", n, n, A, lda, VR, ldvr);

        /* Generate unitary matrix in VR */
        zunghr(n, *ilo, *ihi, VR, ldvr, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VR */
        iwrk = itau;
        zhseqr("S", "V", n, *ilo, *ihi, A, lda, W, VR, ldvr,
               &work[iwrk], lwork - iwrk, info);

    } else {
        /* Compute eigenvalues only - If condition numbers desired, compute Schur form */
        if (wntsnn) {
            job_hseqr = 'E';
        } else {
            job_hseqr = 'S';
        }

        iwrk = itau;
        {
            char job_str[2] = {job_hseqr, '\0'};
            zhseqr(job_str, "N", n, *ilo, *ihi, A, lda, W, VR, ldvr,
                   &work[iwrk], lwork - iwrk, info);
        }
    }

    /* If INFO != 0 from ZHSEQR, then quit */
    if (*info != 0)
        goto L50;

    if (wantvl || wantvr) {
        /* Compute left and/or right eigenvectors */
        ztrevc3(side, "B", select, n, A, lda, VL, ldvl, VR, ldvr,
                n, &nout, &work[iwrk], lwork - iwrk,
                rwork, n, &ierr);
    }

    /* Compute condition numbers if desired */
    if (!wntsnn) {
        ztrsna(sense, "A", select, n, A, lda, VL, ldvl, VR, ldvr,
               rconde, rcondv, n, &nout, &work[iwrk], n, rwork,
               &icond);
    }

    if (wantvl) {
        /* Undo balancing of left eigenvectors */
        zgebak(balanc, "L", n, *ilo, *ihi, scale, n, VL, ldvl, &ierr);

        /* Normalize left eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            scl = ONE / cblas_dznrm2(n, &VL[i * ldvl], 1);
            cblas_zdscal(n, scl, &VL[i * ldvl], 1);
            for (k = 0; k < n; k++) {
                rwork[k] = creal(VL[k + i * ldvl]) *
                           creal(VL[k + i * ldvl]) +
                           cimag(VL[k + i * ldvl]) *
                           cimag(VL[k + i * ldvl]);
            }
            k = (int)cblas_idamax(n, rwork, 1);
            tmp = conj(VL[k + i * ldvl]) / sqrt(rwork[k]);
            cblas_zscal(n, &tmp, &VL[i * ldvl], 1);
            VL[k + i * ldvl] = CMPLX(creal(VL[k + i * ldvl]), ZERO);
        }
    }

    if (wantvr) {
        /* Undo balancing of right eigenvectors */
        zgebak(balanc, "R", n, *ilo, *ihi, scale, n, VR, ldvr, &ierr);

        /* Normalize right eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            scl = ONE / cblas_dznrm2(n, &VR[i * ldvr], 1);
            cblas_zdscal(n, scl, &VR[i * ldvr], 1);
            for (k = 0; k < n; k++) {
                rwork[k] = creal(VR[k + i * ldvr]) *
                           creal(VR[k + i * ldvr]) +
                           cimag(VR[k + i * ldvr]) *
                           cimag(VR[k + i * ldvr]);
            }
            k = (int)cblas_idamax(n, rwork, 1);
            tmp = conj(VR[k + i * ldvr]) / sqrt(rwork[k]);
            cblas_zscal(n, &tmp, &VR[i * ldvr], 1);
            VR[k + i * ldvr] = CMPLX(creal(VR[k + i * ldvr]), ZERO);
        }
    }

    /* Undo scaling if necessary */
L50:
    if (scalea) {
        zlascl("G", 0, 0, cscale, anrm, n - *info, 1, &W[*info],
               (n - *info) > 1 ? (n - *info) : 1, &ierr);
        if (*info == 0) {
            if ((wntsnv || wntsnb) && icond == 0)
                dlascl("G", 0, 0, cscale, anrm, n, 1, rcondv, n, &ierr);
        } else {
            zlascl("G", 0, 0, cscale, anrm, *ilo, 1, W, n, &ierr);
        }
    }

    work[0] = CMPLX((f64)maxwrk, 0.0);
}
