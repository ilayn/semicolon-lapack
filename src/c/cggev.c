/**
 * @file cggev.c
 * @brief CGGEV computes the eigenvalues and, optionally, the left and/or
 *        right eigenvectors for GE matrices.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CGGEV computes for a pair of N-by-N complex nonsymmetric matrices
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
 * @param[out] VL     If jobvl = 'V', the left eigenvectors.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1, and
 *                   if jobvl = 'V', ldvl >= n.
 * @param[out] VR     If jobvr = 'V', the right eigenvectors.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1, and
 *                   if jobvr = 'V', ldvr >= n.
 * @param[out] work   Complex workspace array, dimension (max(1,lwork)).
 *                   On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork   The dimension of work. lwork >= max(1,2*n).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] rwork  Single precision array, dimension (8*n).
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 *                   - = 1,...,n: the QZ iteration failed
 *                   - > n: other errors
 */
void cggev(const char* jobvl, const char* jobvr, const int n,
           c64* A, const int lda,
           c64* B, const int ldb,
           c64* alpha, c64* beta,
           c64* VL, const int ldvl,
           c64* VR, const int ldvr,
           c64* work, const int lwork,
           f32* rwork, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
    int icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo;
    int in, iright, irows, irwrk, itau, iwrk, jc, jr;
    int lwkmin, lwkopt;
    f32 anrm, anrmto = 0.0f, bignum, bnrm, bnrmto = 0.0f, eps,
           smlnum, temp;
    int ldumma[1];
    int nb_geqrf, nb_unmqr, nb_ungqr;

    /* Decode the input arguments */
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

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
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
    }

    /* Compute workspace */
    if (*info == 0) {
        lwkmin = 1 > 2 * n ? 1 : 2 * n;
        nb_geqrf = lapack_get_nb("GEQRF");
        nb_unmqr = lapack_get_nb("ORMQR");
        nb_ungqr = lapack_get_nb("ORGQR");
        lwkopt = 1 > n + n * nb_geqrf ? 1 : n + n * nb_geqrf;
        lwkopt = lwkopt > n + n * nb_unmqr ? lwkopt : n + n * nb_unmqr;
        if (ilvl) {
            lwkopt = lwkopt > n + n * nb_ungqr ? lwkopt : n + n * nb_ungqr;
        }
        work[0] = CMPLXF((f32)lwkopt, 0.0f);

        if (lwork < lwkmin && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("CGGEV", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Get machine constants */
    eps = slamch("P");
    smlnum = slamch("S");
    smlnum = sqrtf(smlnum) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = clange("M", n, n, A, lda, rwork);
    ilascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = 1;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = 1;
    }
    if (ilascl)
        clascl("G", 0, 0, anrm, anrmto, n, n, A, lda, &ierr);

    /* Scale B if max element outside range [SMLNUM,BIGNUM] */
    bnrm = clange("M", n, n, B, ldb, rwork);
    ilbscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        bnrmto = smlnum;
        ilbscl = 1;
    } else if (bnrm > bignum) {
        bnrmto = bignum;
        ilbscl = 1;
    }
    if (ilbscl)
        clascl("G", 0, 0, bnrm, bnrmto, n, n, B, ldb, &ierr);

    /* Permute the matrices A, B to isolate eigenvalues if possible */
    ileft = 0;
    iright = n;
    irwrk = iright + n;
    cggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &rwork[ileft],
           &rwork[iright], &rwork[irwrk], &ierr);

    /* Reduce B to triangular form (QR decomposition of B) */
    irows = ihi + 1 - ilo;
    if (ilv) {
        icols = n - ilo;
    } else {
        icols = irows;
    }
    itau = 0;
    iwrk = itau + irows;
    cgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb,
           &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    /* Apply the unitary transformation to matrix A */
    cunmqr("L", "C", irows, icols, irows,
           &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda,
           &work[iwrk], lwork - iwrk, &ierr);

    /* Initialize VL */
    if (ilvl) {
        claset("Full", n, n, CZERO, CONE, VL, ldvl);
        if (irows > 1) {
            clacpy("L", irows - 1, irows - 1,
                   &B[(ilo + 1) + ilo * ldb], ldb,
                   &VL[(ilo + 1) + ilo * ldvl], ldvl);
        }
        cungqr(irows, irows, irows,
               &VL[ilo + ilo * ldvl], ldvl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    /* Initialize VR */
    if (ilvr)
        claset("Full", n, n, CZERO, CONE, VR, ldvr);

    /* Reduce to generalized Hessenberg form */
    if (ilv) {
        cgghrd(jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb, VL,
               ldvl, VR, ldvr, &ierr);
    } else {
        cgghrd("N", "N", irows, 0, irows - 1, &A[ilo + ilo * lda],
               lda, &B[ilo + ilo * ldb], ldb, VL, ldvl,
               VR, ldvr, &ierr);
    }

    /* Perform QZ algorithm (Compute eigenvalues, and optionally, the
       Schur form and Schur vectors) */
    iwrk = itau;
    const char* chtemp = ilv ? "S" : "E";
    chgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb,
           alpha, beta, VL, ldvl, VR, ldvr,
           &work[iwrk], lwork - iwrk, &rwork[irwrk], &ierr);
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

    /* Compute Eigenvectors */
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

        ctgevc(side, "B", ldumma, n, A, lda, B, ldb, VL, ldvl,
               VR, ldvr, n, &in, &work[iwrk], &rwork[irwrk], &ierr);
        if (ierr != 0) {
            *info = n + 2;
            goto L70;
        }

        /* Undo balancing on VL and VR and normalization */
        if (ilvl) {
            cggbak("P", "L", n, ilo, ihi, &rwork[ileft],
                   &rwork[iright], n, VL, ldvl, &ierr);
            for (jc = 0; jc < n; jc++) {
                temp = ZERO;
                for (jr = 0; jr < n; jr++) {
                    temp = temp > cabs1f(VL[jr + jc * ldvl]) ?
                           temp : cabs1f(VL[jr + jc * ldvl]);
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
            cggbak("P", "R", n, ilo, ihi, &rwork[ileft],
                   &rwork[iright], n, VR, ldvr, &ierr);
            for (jc = 0; jc < n; jc++) {
                temp = ZERO;
                for (jr = 0; jr < n; jr++) {
                    temp = temp > cabs1f(VR[jr + jc * ldvr]) ?
                           temp : cabs1f(VR[jr + jc * ldvr]);
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

    /* Undo scaling if necessary */
L70:
    if (ilascl)
        clascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);

    if (ilbscl)
        clascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
    return;
}
