/**
 * @file cgges3.c
 * @brief CGGES3 computes the eigenvalues, the Schur form, and, optionally,
 *        the matrix of Schur vectors for GE matrices (blocked algorithm).
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CGGES3 computes for a pair of N-by-N complex nonsymmetric matrices
 * (A,B), the generalized eigenvalues, the generalized complex Schur
 * form (S, T), and optionally left and/or right Schur vectors (VSL
 * and VSR). This gives the generalized Schur factorization
 *
 *      (A,B) = ( (VSL)*S*(VSR)**H, (VSL)*T*(VSR)**H )
 *
 * where (VSR)**H is the conjugate-transpose of VSR.
 *
 * Optionally, it also orders the eigenvalues so that a selected cluster
 * of eigenvalues appears in the leading diagonal blocks of the upper
 * triangular matrix S and the upper triangular matrix T. The leading
 * columns of VSL and VSR then form an unitary basis for the
 * corresponding left and right eigenspaces (deflating subspaces).
 *
 * (If only the generalized eigenvalues are needed, use the driver
 * CGGEV instead, which is faster.)
 *
 * A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
 * or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
 * usually represented as the pair (alpha,beta), as there is a
 * reasonable interpretation for beta=0, and even for both being zero.
 *
 * A pair of matrices (S,T) is in generalized complex Schur form if S
 * and T are upper triangular and, in addition, the diagonal elements
 * of T are non-negative real numbers.
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
 *                    On exit, A has been overwritten by its generalized
 *                    Schur form S.
 * @param[in] lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B   On entry, the second of the pair of matrices.
 *                    On exit, B has been overwritten by its generalized
 *                    Schur form T.
 * @param[in] ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out] sdim   If sort = 'N', sdim = 0. If sort = 'S', sdim = number
 *                    of eigenvalues for which selctg is true.
 * @param[out] alpha  Complex array, dimension (n).
 * @param[out] beta   Complex array, dimension (n).
 *                    On exit, ALPHA(j)/BETA(j), j=1,...,N, will be the
 *                    generalized eigenvalues.
 * @param[out] VSL    If jobvsl = 'V', the left Schur vectors.
 * @param[in] ldvsl   The leading dimension of VSL.
 * @param[out] VSR    If jobvsr = 'V', the right Schur vectors.
 * @param[in] ldvsr   The leading dimension of VSR.
 * @param[out] work   Complex workspace array, dimension (max(1,lwork)).
 *                    On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork   The dimension of work. lwork >= max(1,2*n).
 *                    If lwork = -1, a workspace query is assumed.
 * @param[out] rwork  Single precision array, dimension (8*n).
 * @param[out] bwork  Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info
 *                    - = 0: successful exit
 *                    - < 0: if info = -i, the i-th argument had an illegal value
 *                    - = 1,...,n: the QZ iteration failed
 *                    - = n+1: other than QZ iteration failed in CLAQZ0
 *                    - = n+2: after reordering, roundoff changed values of
 *                             some complex eigenvalues so that leading
 *                             eigenvalues in the Generalized Schur form no
 *                             longer satisfy SELCTG=.TRUE.
 *                    - = n+3: reordering failed in CTGSEN
 */
void cgges3(const char* jobvsl, const char* jobvsr, const char* sort,
            cselect2_t selctg, const int n,
            c64* A, const int lda,
            c64* B, const int ldb,
            int* sdim,
            c64* alpha, c64* beta,
            c64* VSL, const int ldvsl,
            c64* VSR, const int ldvsr,
            c64* work, const int lwork,
            f32* rwork, int* bwork, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, wantst;
    int i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo;
    int iright, irows, irwrk, itau, iwrk, lwkopt, lwkmin;
    f32 anrm, anrmto = 0.0f, bignum, bnrm, bnrmto = 0.0f, eps;
    f32 pvsl, pvsr, smlnum;
    int idum[1];
    f32 dif[2];

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

    *info = 0;
    lquery = (lwork == -1);
    lwkmin = 1 > 2 * n ? 1 : 2 * n;

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
        *info = -14;
    } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
        *info = -16;
    } else if (lwork < lwkmin && !lquery) {
        *info = -18;
    }

    if (*info == 0) {
        cgeqrf(n, n, B, ldb, NULL, work, -1, &ierr);
        lwkopt = lwkmin > (n + (int)crealf(work[0])) ?
                 lwkmin : (n + (int)crealf(work[0]));
        cunmqr("L", "C", n, n, n, B, ldb, NULL, A, lda, work,
               -1, &ierr);
        lwkopt = lwkopt > (n + (int)crealf(work[0])) ?
                 lwkopt : (n + (int)crealf(work[0]));
        if (ilvsl) {
            cungqr(n, n, n, VSL, ldvsl, NULL, work, -1, &ierr);
            lwkopt = lwkopt > (n + (int)crealf(work[0])) ?
                     lwkopt : (n + (int)crealf(work[0]));
        }
        cgghd3(jobvsl, jobvsr, n, 0, n - 1, A, lda, B, ldb, VSL,
               ldvsl, VSR, ldvsr, work, -1, &ierr);
        lwkopt = lwkopt > (n + (int)crealf(work[0])) ?
                 lwkopt : (n + (int)crealf(work[0]));
        claqz0("S", jobvsl, jobvsr, n, 0, n - 1, A, lda, B, ldb,
               alpha, beta, VSL, ldvsl, VSR, ldvsr, work, -1,
               rwork, 0, &ierr);
        lwkopt = lwkopt > (int)crealf(work[0]) ?
                 lwkopt : (int)crealf(work[0]);
        if (wantst) {
            ctgsen(0, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
                   alpha, beta, VSL, ldvsl, VSR, ldvsr, sdim,
                   &pvsl, &pvsr, dif, work, -1, idum, 1, &ierr);
            lwkopt = lwkopt > (int)crealf(work[0]) ?
                     lwkopt : (int)crealf(work[0]);
        }
        if (n == 0) {
            work[0] = CONE;
        } else {
            work[0] = CMPLXF((f32)lwkopt, 0.0f);
        }
    }

    if (*info != 0) {
        xerbla("CGGES3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        *sdim = 0;
        return;
    }

    eps = slamch("P");
    smlnum = slamch("S");
    smlnum = sqrtf(smlnum) / eps;
    bignum = ONE / smlnum;

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

    ileft = 0;
    iright = n;
    irwrk = iright + n;
    cggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &rwork[ileft],
           &rwork[iright], &rwork[irwrk], &ierr);

    irows = ihi + 1 - ilo;
    icols = n - ilo;
    itau = 0;
    iwrk = itau + irows;
    cgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb,
           &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    cunmqr("L", "C", irows, icols, irows,
           &B[ilo + ilo * ldb], ldb, &work[itau],
           &A[ilo + ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    if (ilvsl) {
        claset("Full", n, n, CZERO, CONE, VSL, ldvsl);
        if (irows > 1) {
            clacpy("L", irows - 1, irows - 1,
                   &B[(ilo + 1) + ilo * ldb], ldb,
                   &VSL[(ilo + 1) + ilo * ldvsl], ldvsl);
        }
        cungqr(irows, irows, irows, &VSL[ilo + ilo * ldvsl],
               ldvsl, &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvsr)
        claset("Full", n, n, CZERO, CONE, VSR, ldvsr);

    cgghd3(jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb, VSL,
           ldvsl, VSR, ldvsr, &work[iwrk], lwork - iwrk, &ierr);

    *sdim = 0;

    iwrk = itau;
    claqz0("S", jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb,
           alpha, beta, VSL, ldvsl, VSR, ldvsr, &work[iwrk],
           lwork - iwrk, &rwork[irwrk], 0, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L30;
    }

    if (wantst) {

        if (ilascl)
            clascl("G", 0, 0, anrm, anrmto, n, 1, alpha, n, &ierr);
        if (ilbscl)
            clascl("G", 0, 0, bnrm, bnrmto, n, 1, beta, n, &ierr);

        for (i = 0; i < n; i++) {
            bwork[i] = selctg(&alpha[i], &beta[i]);
        }

        ctgsen(0, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
               alpha, beta, VSL, ldvsl, VSR, ldvsr, sdim, &pvsl,
               &pvsr, dif, &work[iwrk], lwork - iwrk, idum, 1, &ierr);
        if (ierr == 1)
            *info = n + 3;
    }

    if (ilvsl)
        cggbak("P", "L", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSL, ldvsl, &ierr);
    if (ilvsr)
        cggbak("P", "R", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSR, ldvsr, &ierr);

    if (ilascl) {
        clascl("U", 0, 0, anrmto, anrm, n, n, A, lda, &ierr);
        clascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);
    }

    if (ilbscl) {
        clascl("U", 0, 0, bnrmto, bnrm, n, n, B, ldb, &ierr);
        clascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
    }

    if (wantst) {

        lastsl = 1;
        *sdim = 0;
        for (i = 0; i < n; i++) {
            cursl = selctg(&alpha[i], &beta[i]);
            if (cursl)
                (*sdim)++;
            if (cursl && !lastsl)
                *info = n + 2;
            lastsl = cursl;
        }
    }

L30:
    work[0] = CMPLXF((f32)lwkopt, 0.0f);

    return;
}
