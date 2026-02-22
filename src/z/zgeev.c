/**
 * @file zgeev.c
 * @brief ZGEEV computes the eigenvalues and, optionally, eigenvectors
 *        of a general complex matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

/**
 * ZGEEV computes for an N-by-N complex nonsymmetric matrix A, the
 * eigenvalues and, optionally, the left and/or right eigenvectors.
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
 * @param[in] jobvl  = 'N': left eigenvectors of A are not computed;
 *                   = 'V': left eigenvectors of A are computed.
 * @param[in] jobvr  = 'N': right eigenvectors of A are not computed;
 *                   = 'V': right eigenvectors of A are computed.
 * @param[in] n      The order of the matrix A. n >= 0.
 * @param[in,out] A  On entry, the N-by-N matrix A.
 *                   On exit, A has been overwritten. Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, n).
 * @param[out] W     Complex array, dimension (n). Contains the computed
 *                   eigenvalues.
 * @param[out] VL    If jobvl = 'V', the left eigenvectors u(j) are stored
 *                   one after another in the columns of VL, in the same
 *                   order as their eigenvalues.
 *                   If jobvl = 'N', VL is not referenced.
 *                   Dimension (ldvl, n).
 * @param[in] ldvl   The leading dimension of VL. ldvl >= 1;
 *                   if jobvl = 'V', ldvl >= n.
 * @param[out] VR    If jobvr = 'V', the right eigenvectors v(j) are stored
 *                   one after another in the columns of VR, in the same
 *                   order as their eigenvalues.
 *                   If jobvr = 'N', VR is not referenced.
 *                   Dimension (ldvr, n).
 * @param[in] ldvr   The leading dimension of VR. ldvr >= 1;
 *                   if jobvr = 'V', ldvr >= n.
 * @param[out] work  Complex workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1, 2*n).
 *                   For good performance, lwork must generally be larger.
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] rwork Double precision array, dimension (2*n).
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 *                   - > 0: if info = i, the QR algorithm failed to compute all
 *                     eigenvalues, and no eigenvectors have been computed;
 *                     elements i:n-1 of W contain eigenvalues which have
 *                     converged.
 */
void zgeev(const char* jobvl, const char* jobvr, const INT n,
           c128* A, const INT lda,
           c128* W,
           c128* VL, const INT ldvl,
           c128* VR, const INT ldvr,
           c128* work, const INT lwork,
           f64* rwork, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT lquery, scalea, wantvl, wantvr;
    char side[2];
    INT hswork, i, ibal, ierr, ihi, ilo, irwork, itau, iwrk, k;
    INT lwork_trevc, maxwrk, minwrk, nout;
    f64 anrm, bignum, cscale, eps, scl, smlnum;
    c128 tmp;
    INT select[1];
    f64 dum[1];
    INT nb_gehrd, nb_unghr;

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
        *info = -8;
    } else if (ldvr < 1 || (wantvr && ldvr < n)) {
        *info = -10;
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
            minwrk = 2 * n;

            if (wantvl) {
                maxwrk = maxwrk > (n + (n - 1) * nb_unghr) ?
                         maxwrk : (n + (n - 1) * nb_unghr);
                ztrevc3("L", "B", select, n, A, lda,
                        VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, rwork, -1, &ierr);
                lwork_trevc = (INT)creal(work[0]);
                maxwrk = maxwrk > (n + lwork_trevc) ?
                         maxwrk : (n + lwork_trevc);
                zhseqr("S", "V", n, 0, n - 1, A, lda, W, VL, ldvl,
                       work, -1, info);
            } else if (wantvr) {
                maxwrk = maxwrk > (n + (n - 1) * nb_unghr) ?
                         maxwrk : (n + (n - 1) * nb_unghr);
                ztrevc3("R", "B", select, n, A, lda,
                        VL, ldvl, VR, ldvr,
                        n, &nout, work, -1, rwork, -1, &ierr);
                lwork_trevc = (INT)creal(work[0]);
                maxwrk = maxwrk > (n + lwork_trevc) ?
                         maxwrk : (n + lwork_trevc);
                zhseqr("S", "V", n, 0, n - 1, A, lda, W, VR, ldvr,
                       work, -1, info);
            } else {
                zhseqr("E", "N", n, 0, n - 1, A, lda, W, VR, ldvr,
                       work, -1, info);
            }
            hswork = (INT)creal(work[0]);
            maxwrk = maxwrk > hswork ? maxwrk : hswork;
            maxwrk = maxwrk > minwrk ? maxwrk : minwrk;
        }
        work[0] = CMPLX((f64)maxwrk, 0.0);

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZGEEV", -(*info));
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

    /* Balance the matrix */
    ibal = 0;
    zgebal("B", n, A, lda, &ilo, &ihi, &rwork[ibal], &ierr);

    /* Reduce to upper Hessenberg form */
    itau = 0;
    iwrk = itau + n;
    zgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk],
           lwork - iwrk, &ierr);

    if (wantvl) {
        /* Want left eigenvectors - Copy Householder vectors to VL */
        side[0] = 'L';
        side[1] = '\0';
        zlacpy("L", n, n, A, lda, VL, ldvl);

        /* Generate unitary matrix in VL */
        zunghr(n, ilo, ihi, VL, ldvl, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VL */
        iwrk = itau;
        zhseqr("S", "V", n, ilo, ihi, A, lda, W, VL, ldvl,
               &work[iwrk], lwork - iwrk, info);

        if (wantvr) {
            /* Want left and right eigenvectors - Copy Schur vectors to VR */
            side[0] = 'B';
            zlacpy("F", n, n, VL, ldvl, VR, ldvr);
        }

    } else if (wantvr) {
        /* Want right eigenvectors - Copy Householder vectors to VR */
        side[0] = 'R';
        side[1] = '\0';
        zlacpy("L", n, n, A, lda, VR, ldvr);

        /* Generate unitary matrix in VR */
        zunghr(n, ilo, ihi, VR, ldvr, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);

        /* Perform QR iteration, accumulating Schur vectors in VR */
        iwrk = itau;
        zhseqr("S", "V", n, ilo, ihi, A, lda, W, VR, ldvr,
               &work[iwrk], lwork - iwrk, info);

    } else {
        /* Compute eigenvalues only */
        iwrk = itau;
        zhseqr("E", "N", n, ilo, ihi, A, lda, W, VR, ldvr,
               &work[iwrk], lwork - iwrk, info);
    }

    /* If INFO != 0 from ZHSEQR, then quit */
    if (*info != 0)
        goto L50;

    if (wantvl || wantvr) {
        /* Compute left and/or right eigenvectors */
        irwork = ibal + n;
        ztrevc3(side, "B", select, n, A, lda, VL, ldvl, VR, ldvr,
                n, &nout, &work[iwrk], lwork - iwrk,
                &rwork[irwork], n, &ierr);
    }

    if (wantvl) {
        /* Undo balancing of left eigenvectors */
        zgebak("B", "L", n, ilo, ihi, &rwork[ibal], n, VL, ldvl, &ierr);

        /* Normalize left eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            scl = ONE / cblas_dznrm2(n, &VL[i * ldvl], 1);
            cblas_zdscal(n, scl, &VL[i * ldvl], 1);
            for (k = 0; k < n; k++) {
                rwork[irwork + k] = creal(VL[k + i * ldvl]) *
                                    creal(VL[k + i * ldvl]) +
                                    cimag(VL[k + i * ldvl]) *
                                    cimag(VL[k + i * ldvl]);
            }
            k = (INT)cblas_idamax(n, &rwork[irwork], 1);
            tmp = conj(VL[k + i * ldvl]) / sqrt(rwork[irwork + k]);
            cblas_zscal(n, &tmp, &VL[i * ldvl], 1);
            VL[k + i * ldvl] = CMPLX(creal(VL[k + i * ldvl]), ZERO);
        }
    }

    if (wantvr) {
        /* Undo balancing of right eigenvectors */
        zgebak("B", "R", n, ilo, ihi, &rwork[ibal], n, VR, ldvr, &ierr);

        /* Normalize right eigenvectors and make largest component real */
        for (i = 0; i < n; i++) {
            scl = ONE / cblas_dznrm2(n, &VR[i * ldvr], 1);
            cblas_zdscal(n, scl, &VR[i * ldvr], 1);
            for (k = 0; k < n; k++) {
                rwork[irwork + k] = creal(VR[k + i * ldvr]) *
                                    creal(VR[k + i * ldvr]) +
                                    cimag(VR[k + i * ldvr]) *
                                    cimag(VR[k + i * ldvr]);
            }
            k = (INT)cblas_idamax(n, &rwork[irwork], 1);
            tmp = conj(VR[k + i * ldvr]) / sqrt(rwork[irwork + k]);
            cblas_zscal(n, &tmp, &VR[i * ldvr], 1);
            VR[k + i * ldvr] = CMPLX(creal(VR[k + i * ldvr]), ZERO);
        }
    }

    /* Undo scaling if necessary */
L50:
    if (scalea) {
        zlascl("G", 0, 0, cscale, anrm, n - *info, 1, &W[*info],
               (n - *info) > 1 ? (n - *info) : 1, &ierr);
        if (*info > 0) {
            zlascl("G", 0, 0, cscale, anrm, ilo, 1, W, n, &ierr);
        }
    }

    work[0] = CMPLX((f64)maxwrk, 0.0);
}
