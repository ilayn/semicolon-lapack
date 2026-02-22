/**
 * @file dgges3.c
 * @brief DGGES3 computes the eigenvalues, the Schur form, and, optionally,
 *        the matrix of Schur vectors for GE matrices (blocked algorithm).
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include "semicolon_cblas.h"

/**
 * DGGES3 computes for a pair of N-by-N real nonsymmetric matrices (A,B),
 * the generalized eigenvalues, the generalized real Schur form (S,T),
 * optionally, the left and/or right matrices of Schur vectors (VSL and
 * VSR). This gives the generalized Schur factorization
 *
 *          (A,B) = ( (VSL)*S*(VSR)**T, (VSL)*T*(VSR)**T )
 *
 * Optionally, it also orders the eigenvalues so that a selected cluster
 * of eigenvalues appears in the leading diagonal blocks of the upper
 * quasi-triangular matrix S and the upper triangular matrix T.
 *
 * @param[in] jobvsl  = 'N': do not compute the left Schur vectors;
 *                    = 'V': compute the left Schur vectors.
 * @param[in] jobvsr  = 'N': do not compute the right Schur vectors;
 *                    = 'V': compute the right Schur vectors.
 * @param[in] sort    = 'N': Eigenvalues are not ordered;
 *                    = 'S': Eigenvalues are ordered (see selctg).
 * @param[in] selctg  Selection function. If sort = 'S', selctg is used to
 *                    select eigenvalues to sort to the top left of the
 *                    Schur form. If sort = 'N', selctg is not referenced.
 * @param[in] n       The order of the matrices A, B, VSL, and VSR. n >= 0.
 * @param[in,out] A   On entry, the first of the pair of matrices.
 *                    On exit, A has been overwritten by its Schur form S.
 * @param[in] lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B   On entry, the second of the pair of matrices.
 *                    On exit, B has been overwritten by its Schur form T.
 * @param[in] ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out] sdim   If sort = 'N', sdim = 0. If sort = 'S', sdim = number
 *                    of eigenvalues for which selctg is true.
 * @param[out] alphar Real parts of generalized eigenvalues.
 * @param[out] alphai Imaginary parts of generalized eigenvalues.
 * @param[out] beta   Beta values of generalized eigenvalues.
 * @param[out] VSL    If jobvsl = 'V', the left Schur vectors.
 * @param[in] ldvsl   The leading dimension of VSL.
 * @param[out] VSR    If jobvsr = 'V', the right Schur vectors.
 * @param[in] ldvsr   The leading dimension of VSR.
 * @param[out] work   Workspace array, dimension (max(1,lwork)).
 * @param[in] lwork   The dimension of work.
 * @param[out] bwork  Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: errors from QZ iteration or reordering
 */
void dgges3(const char* jobvsl, const char* jobvsr, const char* sort,
            dselect3_t selctg, const INT n,
            f64* restrict A, const INT lda,
            f64* restrict B, const INT ldb,
            INT* sdim,
            f64* restrict alphar, f64* restrict alphai,
            f64* restrict beta,
            f64* restrict VSL, const INT ldvsl,
            f64* restrict VSR, const INT ldvsr,
            f64* restrict work, const INT lwork,
            INT* restrict bwork, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, lst2sl, wantst;
    INT i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo;
    INT ip, iright, irows, itau, iwrk, lwkopt, lwkmin;
    f64 anrm, anrmto = 0.0, bignum, bnrm, bnrmto = 0.0, eps;
    f64 pvsl, pvsr, safmax, safmin, smlnum;
    INT idum[1];
    f64 dif[2];

    /* Decode the input arguments */
    if (jobvsl[0] == 'N' || jobvsl[0] == 'n') {
        ijobvl = 1;
        ilvsl = 0;
    } else if (jobvsl[0] == 'V' || jobvsl[0] == 'v') {
        ijobvl = 2;
        ilvsl = 1;
    } else {
        ijobvl = -1;
        ilvsl = 0;
    }

    if (jobvsr[0] == 'N' || jobvsr[0] == 'n') {
        ijobvr = 1;
        ilvsr = 0;
    } else if (jobvsr[0] == 'V' || jobvsr[0] == 'v') {
        ijobvr = 2;
        ilvsr = 1;
    } else {
        ijobvr = -1;
        ilvsr = 0;
    }

    wantst = (sort[0] == 'S' || sort[0] == 's');

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    if (n == 0) {
        lwkmin = 1;
    } else {
        lwkmin = 6 * n + 16;
    }

    if (ijobvl <= 0) {
        *info = -1;
    } else if (ijobvr <= 0) {
        *info = -2;
    } else if (!wantst && !(sort[0] == 'N' || sort[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldvsl < 1 || (ilvsl && ldvsl < n)) {
        *info = -15;
    } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
        *info = -17;
    } else if (lwork < lwkmin && !lquery) {
        *info = -19;
    }

    /* Compute workspace */
    if (*info == 0) {
        dgeqrf(n, n, B, ldb, NULL, work, -1, &ierr);
        lwkopt = lwkmin > (3 * n + (INT)work[0]) ? lwkmin : (3 * n + (INT)work[0]);
        dormqr("L", "T", n, n, n, B, ldb, NULL, A, lda, work, -1, &ierr);
        lwkopt = lwkopt > (3 * n + (INT)work[0]) ? lwkopt : (3 * n + (INT)work[0]);
        if (ilvsl) {
            dorgqr(n, n, n, VSL, ldvsl, NULL, work, -1, &ierr);
            lwkopt = lwkopt > (3 * n + (INT)work[0]) ? lwkopt : (3 * n + (INT)work[0]);
        }
        dgghd3(jobvsl, jobvsr, n, 0, n - 1, A, lda, B, ldb, VSL, ldvsl,
               VSR, ldvsr, work, -1, &ierr);
        lwkopt = lwkopt > (3 * n + (INT)work[0]) ? lwkopt : (3 * n + (INT)work[0]);
        dlaqz0("S", jobvsl, jobvsr, n, 0, n - 1, A, lda, B, ldb,
               alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr,
               work, -1, 0, &ierr);
        lwkopt = lwkopt > (2 * n + (INT)work[0]) ? lwkopt : (2 * n + (INT)work[0]);
        if (wantst) {
            dtgsen(0, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
                   alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr,
                   sdim, &pvsl, &pvsr, dif, work, -1, idum, 1, &ierr);
            lwkopt = lwkopt > (2 * n + (INT)work[0]) ? lwkopt : (2 * n + (INT)work[0]);
        }
        if (n == 0) {
            work[0] = 1;
        } else {
            work[0] = (f64)lwkopt;
        }
    }

    if (*info != 0) {
        xerbla("DGGES3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        *sdim = 0;
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    safmin = dlamch("S");
    safmax = ONE / safmin;
    smlnum = sqrt(safmin) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = dlange("M", n, n, A, lda, work);
    ilascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = 1;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = 1;
    }
    if (ilascl)
        dlascl("G", 0, 0, anrm, anrmto, n, n, A, lda, &ierr);

    /* Scale B if max element outside range [SMLNUM,BIGNUM] */
    bnrm = dlange("M", n, n, B, ldb, work);
    ilbscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        bnrmto = smlnum;
        ilbscl = 1;
    } else if (bnrm > bignum) {
        bnrmto = bignum;
        ilbscl = 1;
    }
    if (ilbscl)
        dlascl("G", 0, 0, bnrm, bnrmto, n, n, B, ldb, &ierr);

    /* Permute the matrix to make it more nearly triangular */
    ileft = 0;
    iright = n;
    iwrk = iright + n;
    dggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &work[ileft],
           &work[iright], &work[iwrk], &ierr);

    /* Reduce B to triangular form (QR decomposition of B) */
    irows = ihi + 1 - ilo;
    icols = n - ilo;
    itau = iwrk;
    iwrk = itau + irows;
    dgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb, &work[itau],
           &work[iwrk], lwork - iwrk, &ierr);

    /* Apply the orthogonal transformation to matrix A */
    dormqr("L", "T", irows, icols, irows, &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    /* Initialize VSL */
    if (ilvsl) {
        dlaset("Full", n, n, ZERO, ONE, VSL, ldvsl);
        if (irows > 1) {
            dlacpy("L", irows - 1, irows - 1, &B[(ilo + 1) + ilo * ldb], ldb,
                   &VSL[(ilo + 1) + ilo * ldvsl], ldvsl);
        }
        dorgqr(irows, irows, irows, &VSL[ilo + ilo * ldvsl], ldvsl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    /* Initialize VSR */
    if (ilvsr)
        dlaset("Full", n, n, ZERO, ONE, VSR, ldvsr);

    /* Reduce to generalized Hessenberg form */
    dgghd3(jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb, VSL,
           ldvsl, VSR, ldvsr, &work[iwrk], lwork - iwrk, &ierr);

    /* Perform QZ algorithm, computing Schur vectors if desired */
    iwrk = itau;
    dlaqz0("S", jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb,
           alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr,
           &work[iwrk], lwork - iwrk, 0, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L50;
    }

    /* Sort eigenvalues ALPHA/BETA if desired */
    *sdim = 0;
    if (wantst) {

        /* Undo scaling on eigenvalues before SELCTGing */
        if (ilascl) {
            dlascl("G", 0, 0, anrmto, anrm, n, 1, alphar, n, &ierr);
            dlascl("G", 0, 0, anrmto, anrm, n, 1, alphai, n, &ierr);
        }
        if (ilbscl)
            dlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

        /* Select eigenvalues */
        for (i = 0; i < n; i++) {
            bwork[i] = selctg(&alphar[i], &alphai[i], &beta[i]);
        }

        dtgsen(0, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
               alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr, sdim, &pvsl,
               &pvsr, dif, &work[iwrk], lwork - iwrk, idum, 1, &ierr);
        if (ierr == 1)
            *info = n + 3;
    }

    /* Apply back-permutation to VSL and VSR */
    if (ilvsl)
        dggbak("P", "L", n, ilo, ihi, &work[ileft],
               &work[iright], n, VSL, ldvsl, &ierr);

    if (ilvsr)
        dggbak("P", "R", n, ilo, ihi, &work[ileft],
               &work[iright], n, VSR, ldvsr, &ierr);

    /* Check if unscaling would cause over/underflow, if so, rescale */
    if (ilascl) {
        for (i = 0; i < n; i++) {
            if (alphai[i] != ZERO) {
                if ((alphar[i] / safmax) > (anrmto / anrm) ||
                    (safmin / alphar[i]) > (anrm / anrmto)) {
                    work[0] = fabs(A[i + i * lda] / alphar[i]);
                    beta[i] = beta[i] * work[0];
                    alphar[i] = alphar[i] * work[0];
                    alphai[i] = alphai[i] * work[0];
                } else if ((alphai[i] / safmax) > (anrmto / anrm) ||
                           (safmin / alphai[i]) > (anrm / anrmto)) {
                    work[0] = fabs(A[i + (i + 1) * lda] / alphai[i]);
                    beta[i] = beta[i] * work[0];
                    alphar[i] = alphar[i] * work[0];
                    alphai[i] = alphai[i] * work[0];
                }
            }
        }
    }

    if (ilbscl) {
        for (i = 0; i < n; i++) {
            if (alphai[i] != ZERO) {
                if ((beta[i] / safmax) > (bnrmto / bnrm) ||
                    (safmin / beta[i]) > (bnrm / bnrmto)) {
                    work[0] = fabs(B[i + i * ldb] / beta[i]);
                    beta[i] = beta[i] * work[0];
                    alphar[i] = alphar[i] * work[0];
                    alphai[i] = alphai[i] * work[0];
                }
            }
        }
    }

    /* Undo scaling */
    if (ilascl) {
        dlascl("H", 0, 0, anrmto, anrm, n, n, A, lda, &ierr);
        dlascl("G", 0, 0, anrmto, anrm, n, 1, alphar, n, &ierr);
        dlascl("G", 0, 0, anrmto, anrm, n, 1, alphai, n, &ierr);
    }

    if (ilbscl) {
        dlascl("U", 0, 0, bnrmto, bnrm, n, n, B, ldb, &ierr);
        dlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
    }

    if (wantst) {

        /* Check if reordering is correct */
        lastsl = 1;
        lst2sl = 1;
        *sdim = 0;
        ip = 0;
        for (i = 0; i < n; i++) {
            cursl = selctg(&alphar[i], &alphai[i], &beta[i]);
            if (alphai[i] == ZERO) {
                if (cursl)
                    (*sdim)++;
                ip = 0;
                if (cursl && !lastsl)
                    *info = n + 2;
            } else {
                if (ip == 1) {
                    /* Last eigenvalue of conjugate pair */
                    cursl = cursl || lastsl;
                    lastsl = cursl;
                    if (cursl)
                        *sdim = *sdim + 2;
                    ip = -1;
                    if (cursl && !lst2sl)
                        *info = n + 2;
                } else {
                    /* First eigenvalue of conjugate pair */
                    ip = 1;
                }
            }
            lst2sl = lastsl;
            lastsl = cursl;
        }
    }

L50:
    work[0] = (f64)lwkopt;

    return;
}
