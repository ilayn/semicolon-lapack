/**
 * @file sggev3.c
 * @brief SGGEV3 computes the eigenvalues and, optionally, the left and/or
 *        right eigenvectors for GE matrices (blocked algorithm).
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * SGGEV3 computes for a pair of N-by-N real nonsymmetric matrices (A,B)
 * the generalized eigenvalues, and optionally, the left and/or right
 * generalized eigenvectors.
 *
 * A generalized eigenvalue for a pair of matrices (A,B) is a scalar
 * lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
 * singular. It is usually represented as the pair (alpha,beta), as
 * there is a reasonable interpretation for beta=0, and even for both
 * being zero.
 *
 * The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
 * of (A,B) satisfies
 *                  A * v(j) = lambda(j) * B * v(j).
 *
 * The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
 * of (A,B) satisfies
 *                  u(j)**H * A  = lambda(j) * u(j)**H * B .
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
 * @param[out] alphar Real parts of generalized eigenvalues.
 * @param[out] alphai Imaginary parts of generalized eigenvalues.
 * @param[out] beta   Beta values of generalized eigenvalues.
 * @param[out] VL     If jobvl = 'V', the left eigenvectors.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1, and
 *                   if jobvl = 'V', ldvl >= n.
 * @param[out] VR     If jobvr = 'V', the right eigenvectors.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1, and
 *                   if jobvr = 'V', ldvr >= n.
 * @param[out] work   Workspace array, dimension (max(1,lwork)).
 * @param[in] lwork   The dimension of work. lwork >= max(1,8*n).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1,...,n: the QZ iteration failed
 *                         - > n: other errors
 */
void sggev3(const char* jobvl, const char* jobvr, const int n,
            float* const restrict A, const int lda,
            float* const restrict B, const int ldb,
            float* const restrict alphar, float* const restrict alphai,
            float* const restrict beta,
            float* const restrict VL, const int ldvl,
            float* const restrict VR, const int ldvr,
            float* const restrict work, const int lwork, int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
    int icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo;
    int in, iright, irows, itau, iwrk, jc, jr, lwkopt, lwkmin;
    float anrm, anrmto = 0.0f, bignum, bnrm, bnrmto = 0.0f, eps, smlnum, temp;
    int ldumma[1];

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
    lwkmin = 1 > 8 * n ? 1 : 8 * n;
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
        *info = -12;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        *info = -14;
    } else if (lwork < lwkmin && !lquery) {
        *info = -16;
    }

    /* Compute workspace */
    if (*info == 0) {
        sgeqrf(n, n, B, ldb, NULL, work, -1, &ierr);
        lwkopt = lwkmin > (3 * n + (int)work[0]) ? lwkmin : (3 * n + (int)work[0]);
        sormqr("L", "T", n, n, n, B, ldb, NULL, A, lda, work, -1, &ierr);
        lwkopt = lwkopt > (3 * n + (int)work[0]) ? lwkopt : (3 * n + (int)work[0]);
        if (ilvl) {
            sorgqr(n, n, n, VL, ldvl, NULL, work, -1, &ierr);
            lwkopt = lwkopt > (3 * n + (int)work[0]) ? lwkopt : (3 * n + (int)work[0]);
        }
        if (ilv) {
            sgghd3(jobvl, jobvr, n, 0, n - 1, A, lda, B, ldb, VL, ldvl,
                   VR, ldvr, work, -1, &ierr);
            lwkopt = lwkopt > (3 * n + (int)work[0]) ? lwkopt : (3 * n + (int)work[0]);
            slaqz0("S", jobvl, jobvr, n, 0, n - 1, A, lda, B, ldb,
                   alphar, alphai, beta, VL, ldvl, VR, ldvr,
                   work, -1, 0, &ierr);
            lwkopt = lwkopt > (2 * n + (int)work[0]) ? lwkopt : (2 * n + (int)work[0]);
        } else {
            sgghd3("N", "N", n, 0, n - 1, A, lda, B, ldb, VL, ldvl,
                   VR, ldvr, work, -1, &ierr);
            lwkopt = lwkopt > (3 * n + (int)work[0]) ? lwkopt : (3 * n + (int)work[0]);
            slaqz0("E", jobvl, jobvr, n, 0, n - 1, A, lda, B, ldb,
                   alphar, alphai, beta, VL, ldvl, VR, ldvr,
                   work, -1, 0, &ierr);
            lwkopt = lwkopt > (2 * n + (int)work[0]) ? lwkopt : (2 * n + (int)work[0]);
        }
        if (n == 0) {
            work[0] = 1;
        } else {
            work[0] = (float)lwkopt;
        }
    }

    if (*info != 0) {
        xerbla("SGGEV3", -(*info));
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
    bignum = ONE / smlnum;
    smlnum = sqrtf(smlnum) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = slange("M", n, n, A, lda, work);
    ilascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = 1;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = 1;
    }
    if (ilascl)
        slascl("G", 0, 0, anrm, anrmto, n, n, A, lda, &ierr);

    /* Scale B if max element outside range [SMLNUM,BIGNUM] */
    bnrm = slange("M", n, n, B, ldb, work);
    ilbscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        bnrmto = smlnum;
        ilbscl = 1;
    } else if (bnrm > bignum) {
        bnrmto = bignum;
        ilbscl = 1;
    }
    if (ilbscl)
        slascl("G", 0, 0, bnrm, bnrmto, n, n, B, ldb, &ierr);

    /* Permute the matrices A, B to isolate eigenvalues if possible */
    ileft = 0;
    iright = n;
    iwrk = iright + n;
    sggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &work[ileft],
           &work[iright], &work[iwrk], &ierr);

    /* Reduce B to triangular form (QR decomposition of B) */
    irows = ihi + 1 - ilo;
    if (ilv) {
        icols = n - ilo;
    } else {
        icols = irows;
    }
    itau = iwrk;
    iwrk = itau + irows;
    sgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb, &work[itau],
           &work[iwrk], lwork - iwrk, &ierr);

    /* Apply the orthogonal transformation to matrix A */
    sormqr("L", "T", irows, icols, irows, &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    /* Initialize VL */
    if (ilvl) {
        slaset("Full", n, n, ZERO, ONE, VL, ldvl);
        if (irows > 1) {
            slacpy("L", irows - 1, irows - 1, &B[(ilo + 1) + ilo * ldb], ldb,
                   &VL[(ilo + 1) + ilo * ldvl], ldvl);
        }
        sorgqr(irows, irows, irows, &VL[ilo + ilo * ldvl], ldvl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    /* Initialize VR */
    if (ilvr)
        slaset("Full", n, n, ZERO, ONE, VR, ldvr);

    /* Reduce to generalized Hessenberg form */
    if (ilv) {
        /* Eigenvectors requested -- work on whole matrix. */
        sgghd3(jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb, VL,
               ldvl, VR, ldvr, &work[iwrk], lwork - iwrk, &ierr);
    } else {
        sgghd3("N", "N", irows, 0, irows - 1, &A[ilo + ilo * lda], lda,
               &B[ilo + ilo * ldb], ldb, VL, ldvl, VR, ldvr,
               &work[iwrk], lwork - iwrk, &ierr);
    }

    /* Perform QZ algorithm (Compute eigenvalues, and optionally, the
       Schur forms and Schur vectors) */
    iwrk = itau;
    const char* chtemp = ilv ? "S" : "E";
    slaqz0(chtemp, jobvl, jobvr, n, ilo, ihi, A, lda, B, ldb,
           alphar, alphai, beta, VL, ldvl, VR, ldvr,
           &work[iwrk], lwork - iwrk, 0, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L110;
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
        stgevc(side, "B", ldumma, n, A, lda, B, ldb, VL, ldvl,
               VR, ldvr, n, &in, &work[iwrk], &ierr);
        if (ierr != 0) {
            *info = n + 2;
            goto L110;
        }

        /* Undo balancing on VL and VR and normalization */
        if (ilvl) {
            sggbak("P", "L", n, ilo, ihi, &work[ileft],
                   &work[iright], n, VL, ldvl, &ierr);
            for (jc = 0; jc < n; jc++) {
                if (alphai[jc] < ZERO)
                    continue;
                temp = ZERO;
                if (alphai[jc] == ZERO) {
                    for (jr = 0; jr < n; jr++) {
                        temp = temp > fabsf(VL[jr + jc * ldvl]) ?
                               temp : fabsf(VL[jr + jc * ldvl]);
                    }
                } else {
                    for (jr = 0; jr < n; jr++) {
                        temp = temp > (fabsf(VL[jr + jc * ldvl]) +
                                       fabsf(VL[jr + (jc + 1) * ldvl])) ?
                               temp : (fabsf(VL[jr + jc * ldvl]) +
                                       fabsf(VL[jr + (jc + 1) * ldvl]));
                    }
                }
                if (temp < smlnum)
                    continue;
                temp = ONE / temp;
                if (alphai[jc] == ZERO) {
                    for (jr = 0; jr < n; jr++) {
                        VL[jr + jc * ldvl] = VL[jr + jc * ldvl] * temp;
                    }
                } else {
                    for (jr = 0; jr < n; jr++) {
                        VL[jr + jc * ldvl] = VL[jr + jc * ldvl] * temp;
                        VL[jr + (jc + 1) * ldvl] = VL[jr + (jc + 1) * ldvl] * temp;
                    }
                }
            }
        }
        if (ilvr) {
            sggbak("P", "R", n, ilo, ihi, &work[ileft],
                   &work[iright], n, VR, ldvr, &ierr);
            for (jc = 0; jc < n; jc++) {
                if (alphai[jc] < ZERO)
                    continue;
                temp = ZERO;
                if (alphai[jc] == ZERO) {
                    for (jr = 0; jr < n; jr++) {
                        temp = temp > fabsf(VR[jr + jc * ldvr]) ?
                               temp : fabsf(VR[jr + jc * ldvr]);
                    }
                } else {
                    for (jr = 0; jr < n; jr++) {
                        temp = temp > (fabsf(VR[jr + jc * ldvr]) +
                                       fabsf(VR[jr + (jc + 1) * ldvr])) ?
                               temp : (fabsf(VR[jr + jc * ldvr]) +
                                       fabsf(VR[jr + (jc + 1) * ldvr]));
                    }
                }
                if (temp < smlnum)
                    continue;
                temp = ONE / temp;
                if (alphai[jc] == ZERO) {
                    for (jr = 0; jr < n; jr++) {
                        VR[jr + jc * ldvr] = VR[jr + jc * ldvr] * temp;
                    }
                } else {
                    for (jr = 0; jr < n; jr++) {
                        VR[jr + jc * ldvr] = VR[jr + jc * ldvr] * temp;
                        VR[jr + (jc + 1) * ldvr] = VR[jr + (jc + 1) * ldvr] * temp;
                    }
                }
            }
        }

        /* End of eigenvector calculation */
    }

    /* Undo scaling if necessary */
L110:
    if (ilascl) {
        slascl("G", 0, 0, anrmto, anrm, n, 1, alphar, n, &ierr);
        slascl("G", 0, 0, anrmto, anrm, n, 1, alphai, n, &ierr);
    }

    if (ilbscl) {
        slascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
    }

    work[0] = (float)lwkopt;
    return;
}
