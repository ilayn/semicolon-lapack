/**
 * @file zggev3.c
 * @brief ZGGEV3 computes the eigenvalues and, optionally, the left and/or
 *        right eigenvectors for GE matrices (blocked algorithm).
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZGGEV3 computes for a pair of N-by-N complex nonsymmetric matrices
 * (A,B), the generalized eigenvalues, and optionally, the left and/or
 * right generalized eigenvectors.
 *
 * A generalized eigenvalue for a pair of matrices (A,B) is a scalar
 * lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
 * singular. It is usually represented as the pair (alpha,beta), as
 * there is a reasonable interpretation for beta=0, and even for both
 * being zero.
 *
 * The right generalized eigenvector v(j) corresponding to the
 * generalized eigenvalue lambda(j) of (A,B) satisfies
 *
 *              A * v(j) = lambda(j) * B * v(j).
 *
 * The left generalized eigenvector u(j) corresponding to the
 * generalized eigenvalues lambda(j) of (A,B) satisfies
 *
 *              u(j)**H * A = lambda(j) * u(j)**H * B
 *
 * where u(j)**H is the conjugate-transpose of u(j).
 *
 * @param[in] jobvl  = 'N': do not compute the left generalized eigenvectors;
 *                   = 'V': compute the left generalized eigenvectors.
 * @param[in] jobvr  = 'N': do not compute the right generalized eigenvectors;
 *                   = 'V': compute the right generalized eigenvectors.
 * @param[in] n      The order of the matrices A, B, VL, and VR. n >= 0.
 * @param[in,out] A  On entry, the matrix A in the pair (A,B).
 *                   On exit, A has been overwritten.
 * @param[in] lda    The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B  On entry, the matrix B in the pair (A,B).
 *                   On exit, B has been overwritten.
 * @param[in] ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out] alpha Complex array, dimension (n).
 * @param[out] beta  Complex array, dimension (n).
 *                   On exit, ALPHA(j)/BETA(j), j=1,...,N, will be the
 *                   generalized eigenvalues.
 * @param[out] VL    If jobvl = 'V', the left eigenvectors u(j) are stored
 *                   one after another in the columns of VL.
 * @param[in] ldvl   The leading dimension of VL. ldvl >= 1, and
 *                   if jobvl = 'V', ldvl >= n.
 * @param[out] VR    If jobvr = 'V', the right eigenvectors v(j) are stored
 *                   one after another in the columns of VR.
 * @param[in] ldvr   The leading dimension of VR. ldvr >= 1, and
 *                   if jobvr = 'V', ldvr >= n.
 * @param[out] work  Complex workspace array, dimension (max(1,lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1,2*n).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] rwork Double precision array, dimension (8*n).
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 *                   - = 1,...,n: the QZ iteration failed
 *                   - > n: other errors
 */
void zggev3(const char* jobvl, const char* jobvr, const int n,
            double complex* A, const int lda,
            double complex* B, const int ldb,
            double complex* alpha, double complex* beta,
            double complex* VL, const int ldvl,
            double complex* VR, const int ldvr,
            double complex* work, const int lwork,
            double* rwork, int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double complex CZERO = CMPLX(0.0, 0.0);
    const double complex CONE = CMPLX(1.0, 0.0);

    int ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
    int icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo;
    int in, iright, irows, irwrk, itau, iwrk, jc, jr, lwkopt, lwkmin;
    double anrm, anrmto = 0.0, bignum, bnrm, bnrmto = 0.0, eps, smlnum, temp;
    double complex x;
    int ldumma[1];

    if (jobvl[0] == 'N' || jobvl[0] == 'n') {
        ijobvl = 1;
        ilvl = 0;
    } else if (jobvl[0] == 'V' || jobvl[0] == 'v') {
        ijobvl = 2;
        ilvl = 1;
    } else {
        ijobvl = -1;
        ilvl = 0;
    }

    if (jobvr[0] == 'N' || jobvr[0] == 'n') {
        ijobvr = 1;
        ilvr = 0;
    } else if (jobvr[0] == 'V' || jobvr[0] == 'v') {
        ijobvr = 2;
        ilvr = 1;
    } else {
        ijobvr = -1;
        ilvr = 0;
    }
    ilv = ilvl || ilvr;

    *info = 0;
    lquery = (lwork == -1);
    lwkmin = 1 > 2 * n ? 1 : 2 * n;
    if (ijobvl <= 0) {
        *info = -1;
    } else if (ijobvr <= 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldvl < 1 || (ilvl && ldvl < n)) {
        *info = -11;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        *info = -13;
    } else if (lwork < lwkmin && !lquery) {
        *info = -15;
    }

    if (*info == 0) {
        zgeqrf(n, n, B, ldb, work, work, -1, &ierr);
        lwkopt = lwkmin > (n + (int)creal(work[0])) ?
                 lwkmin : (n + (int)creal(work[0]));
        zunmqr("L", "C", n, n, n, B, ldb, work, A, lda, work,
               -1, &ierr);
        lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                 lwkopt : (n + (int)creal(work[0]));
        if (ilvl) {
            zungqr(n, n, n, VL, ldvl, work, work, -1, &ierr);
            lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                     lwkopt : (n + (int)creal(work[0]));
        }
        if (ilv) {
            zgghd3(jobvl, jobvr, n, 1, n, A, lda, B, ldb, VL,
                   ldvl, VR, ldvr, work, -1, &ierr);
            lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                     lwkopt : (n + (int)creal(work[0]));
            zlaqz0("S", jobvl, jobvr, n, 1, n, A, lda, B, ldb,
                   alpha, beta, VL, ldvl, VR, ldvr, work, -1,
                   rwork, 0, &ierr);
            lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                     lwkopt : (n + (int)creal(work[0]));
        } else {
            zgghd3(jobvl, jobvr, n, 1, n, A, lda, B, ldb, VL,
                   ldvl, VR, ldvr, work, -1, &ierr);
            lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                     lwkopt : (n + (int)creal(work[0]));
            zlaqz0("E", jobvl, jobvr, n, 1, n, A, lda, B, ldb,
                   alpha, beta, VL, ldvl, VR, ldvr, work, -1,
                   rwork, 0, &ierr);
            lwkopt = lwkopt > (n + (int)creal(work[0])) ?
                     lwkopt : (n + (int)creal(work[0]));
        }
        if (n == 0) {
            work[0] = CONE;
        } else {
            work[0] = CMPLX((double)lwkopt, 0.0);
        }
    }

    if (*info != 0) {
        xerbla("ZGGEV3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    eps = dlamch("P");
    smlnum = dlamch("S");
    bignum = ONE / smlnum;
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    anrm = zlange("M", n, n, A, lda, rwork);
    ilascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = 1;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = 1;
    }
    if (ilascl)
        zlascl("G", 0, 0, anrm, anrmto, n, n, A, lda, &ierr);

    bnrm = zlange("M", n, n, B, ldb, rwork);
    ilbscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        bnrmto = smlnum;
        ilbscl = 1;
    } else if (bnrm > bignum) {
        bnrmto = bignum;
        ilbscl = 1;
    }
    if (ilbscl)
        zlascl("G", 0, 0, bnrm, bnrmto, n, n, B, ldb, &ierr);

    ileft = 0;
    iright = n;
    irwrk = iright + n;
    zggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &rwork[ileft],
           &rwork[iright], &rwork[irwrk], &ierr);

    irows = ihi + 1 - ilo;
    if (ilv) {
        icols = n + 1 - ilo;
    } else {
        icols = irows;
    }
    itau = 0;
    iwrk = itau + irows;
    zgeqrf(irows, icols, &B[(ilo - 1) + (ilo - 1) * ldb], ldb,
           &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    zunmqr("L", "C", irows, icols, irows,
           &B[(ilo - 1) + (ilo - 1) * ldb], ldb, &work[itau],
           &A[(ilo - 1) + (ilo - 1) * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    if (ilvl) {
        zlaset("Full", n, n, CZERO, CONE, VL, ldvl);
        if (irows > 1) {
            zlacpy("L", irows - 1, irows - 1,
                   &B[ilo + (ilo - 1) * ldb], ldb,
                   &VL[ilo + (ilo - 1) * ldvl], ldvl);
        }
        zungqr(irows, irows, irows, &VL[(ilo - 1) + (ilo - 1) * ldvl],
               ldvl, &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvr)
        zlaset("Full", n, n, CZERO, CONE, VR, ldvr);

    if (ilv) {
        zgghd3(jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb, VL,
               ldvl, VR, ldvr, &work[iwrk], lwork - iwrk, &ierr);
    } else {
        zgghd3("N", "N", irows, 1, irows, &A[(ilo - 1) + (ilo - 1) * lda],
               lda, &B[(ilo - 1) + (ilo - 1) * ldb], ldb, VL, ldvl, VR,
               ldvr, &work[iwrk], lwork - iwrk, &ierr);
    }

    iwrk = itau;
    const char* chtemp = ilv ? "S" : "E";
    zlaqz0(chtemp, jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb,
           alpha, beta, VL, ldvl, VR, ldvr, &work[iwrk],
           lwork - iwrk, &rwork[irwrk], 0, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L70;
    }

    if (ilv) {
        const char* side;
        if (ilvl) {
            if (ilvr) {
                side = "B";
            } else {
                side = "L";
            }
        } else {
            side = "R";
        }

        ztgevc(side, "B", ldumma, n, A, lda, B, ldb, VL, ldvl,
               VR, ldvr, n, &in, &work[iwrk], &rwork[irwrk],
               &ierr);
        if (ierr != 0) {
            *info = n + 2;
            goto L70;
        }

        if (ilvl) {
            zggbak("P", "L", n, ilo, ihi, &rwork[ileft],
                   &rwork[iright], n, VL, ldvl, &ierr);
            for (jc = 0; jc < n; jc++) {
                temp = ZERO;
                for (jr = 0; jr < n; jr++) {
                    x = VL[jr + jc * ldvl];
                    temp = temp > (fabs(creal(x)) + fabs(cimag(x))) ?
                           temp : (fabs(creal(x)) + fabs(cimag(x)));
                }
                if (temp < smlnum)
                    continue;
                temp = ONE / temp;
                for (jr = 0; jr < n; jr++) {
                    VL[jr + jc * ldvl] = VL[jr + jc * ldvl] * temp;
                }
            }
        }
        if (ilvr) {
            zggbak("P", "R", n, ilo, ihi, &rwork[ileft],
                   &rwork[iright], n, VR, ldvr, &ierr);
            for (jc = 0; jc < n; jc++) {
                temp = ZERO;
                for (jr = 0; jr < n; jr++) {
                    x = VR[jr + jc * ldvr];
                    temp = temp > (fabs(creal(x)) + fabs(cimag(x))) ?
                           temp : (fabs(creal(x)) + fabs(cimag(x)));
                }
                if (temp < smlnum)
                    continue;
                temp = ONE / temp;
                for (jr = 0; jr < n; jr++) {
                    VR[jr + jc * ldvr] = VR[jr + jc * ldvr] * temp;
                }
            }
        }
    }

L70:
    if (ilascl)
        zlascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);

    if (ilbscl)
        zlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

    work[0] = CMPLX((double)lwkopt, 0.0);
    return;
}
