/**
 * @file cgges.c
 * @brief CGGES computes the eigenvalues, the Schur form, and, optionally,
 *        the matrix of Schur vectors for GE matrices.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CGGES computes for a pair of N-by-N complex nonsymmetric matrices
 * (A,B), the generalized eigenvalues, the generalized complex Schur
 * form (S, T), and optionally left and/or right Schur vectors (VSL
 * and VSR). This gives the generalized Schur factorization
 *
 *         (A,B) = ( (VSL)*S*(VSR)**H, (VSL)*T*(VSR)**H )
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
 * @param[in] selctg  Selection function of two complex arguments.
 *                    If sort = 'S', selctg is used to select eigenvalues
 *                    to sort to the top left of the Schur form.
 *                    An eigenvalue ALPHA(j)/BETA(j) is selected if
 *                    SELCTG(ALPHA(j),BETA(j)) is true.
 *                    If sort = 'N', selctg is not referenced.
 * @param[in] n       The order of the matrices A, B, VSL, and VSR. n >= 0.
 * @param[in,out] A   On entry, the first of the pair of matrices.
 *                    On exit, A has been overwritten by its generalized Schur
 *                    form S. Dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B   On entry, the second of the pair of matrices.
 *                    On exit, B has been overwritten by its generalized Schur
 *                    form T. Dimension (ldb, n).
 * @param[in] ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out] sdim   If sort = 'N', sdim = 0.
 *                    If sort = 'S', sdim = number of eigenvalues (after sorting)
 *                    for which selctg is true.
 * @param[out] alpha  Complex array, dimension (n).
 * @param[out] beta   Complex array, dimension (n).
 *                    On exit, ALPHA(j)/BETA(j), j=1,...,N, will be the
 *                    generalized eigenvalues. BETA(j) will be non-negative real.
 * @param[out] VSL    If jobvsl = 'V', the left Schur vectors.
 *                    Dimension (ldvsl, n).
 * @param[in] ldvsl   The leading dimension of VSL. ldvsl >= 1, and
 *                    if jobvsl = 'V', ldvsl >= n.
 * @param[out] VSR    If jobvsr = 'V', the right Schur vectors.
 *                    Dimension (ldvsr, n).
 * @param[in] ldvsr   The leading dimension of VSR. ldvsr >= 1, and
 *                    if jobvsr = 'V', ldvsr >= n.
 * @param[out] work   Complex workspace array, dimension (max(1,lwork)).
 *                    On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork   The dimension of work. lwork >= max(1,2*n).
 *                    If lwork = -1, a workspace query is assumed.
 * @param[out] rwork  Single precision array, dimension (8*n).
 * @param[out] bwork  Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info
 *                    - = 0: successful exit
 *                    - < 0: if info = -i, the i-th argument had an illegal value
 *                    - = 1,...,n: the QZ iteration failed.  (A,B) are not in Schur
 *                      form, but ALPHA(j) and BETA(j) should be correct for
 *                      j=INFO+1,...,N.
 *                    - = n+1: other than QZ iteration failed in CHGEQZ
 *                    - = n+2: after reordering, roundoff changed values of
 *                      some complex eigenvalues so that leading eigenvalues
 *                      in the Generalized Schur form no longer satisfy
 *                      SELCTG=.TRUE.  This could also be caused due to scaling.
 *                    - = n+3: reordering failed in CTGSEN.
 */
void cgges(const char* jobvsl, const char* jobvsr, const char* sort,
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
    int iright, irows, irwrk, itau, iwrk, lwkmin, lwkopt;
    f32 anrm, anrmto = 0.0f, bignum, bnrm, bnrmto = 0.0f, eps,
           pvsl, pvsr, smlnum;
    int idum[1];
    f32 dif[2];
    int nb_geqrf, nb_unmqr, nb_ungqr;

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
    }

    /* Compute workspace */
    if (*info == 0) {
        lwkmin = 1 > 2 * n ? 1 : 2 * n;
        nb_geqrf = lapack_get_nb("GEQRF");
        nb_unmqr = lapack_get_nb("ORMQR");
        nb_ungqr = lapack_get_nb("ORGQR");
        lwkopt = 1 > n + n * nb_geqrf ? 1 : n + n * nb_geqrf;
        lwkopt = lwkopt > n + n * nb_unmqr ? lwkopt : n + n * nb_unmqr;
        if (ilvsl) {
            lwkopt = lwkopt > n + n * nb_ungqr ? lwkopt : n + n * nb_ungqr;
        }
        work[0] = CMPLXF((f32)lwkopt, 0.0f);

        if (lwork < lwkmin && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("CGGES", -(*info));
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

    /* Permute the matrix to make it more nearly triangular
       (Real Workspace: need 6*N) */
    ileft = 0;
    iright = n;
    irwrk = iright + n;
    cggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &rwork[ileft],
           &rwork[iright], &rwork[irwrk], &ierr);

    /* Reduce B to triangular form (QR decomposition of B)
       (Complex Workspace: need N, prefer N*NB) */
    irows = ihi + 1 - ilo;
    icols = n - ilo;
    itau = 0;
    iwrk = itau + irows;
    cgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb,
           &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    /* Apply the unitary transformation to matrix A
       (Complex Workspace: need N, prefer N*NB) */
    cunmqr("L", "C", irows, icols, irows,
           &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda,
           &work[iwrk], lwork - iwrk, &ierr);

    /* Initialize VSL
       (Complex Workspace: need N, prefer N*NB) */
    if (ilvsl) {
        claset("Full", n, n, CZERO, CONE, VSL, ldvsl);
        if (irows > 1) {
            clacpy("L", irows - 1, irows - 1,
                   &B[(ilo + 1) + ilo * ldb], ldb,
                   &VSL[(ilo + 1) + ilo * ldvsl], ldvsl);
        }
        cungqr(irows, irows, irows,
               &VSL[ilo + ilo * ldvsl], ldvsl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    /* Initialize VSR */
    if (ilvsr)
        claset("Full", n, n, CZERO, CONE, VSR, ldvsr);

    /* Reduce to generalized Hessenberg form
       (Workspace: none needed) */
    cgghrd(jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb, VSL,
           ldvsl, VSR, ldvsr, &ierr);

    *sdim = 0;

    /* Perform QZ algorithm, computing Schur vectors if desired
       (Complex Workspace: need N)
       (Real Workspace: need N) */
    iwrk = itau;
    chgeqz("S", jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb,
           alpha, beta, VSL, ldvsl, VSR, ldvsr,
           &work[iwrk], lwork - iwrk, &rwork[irwrk], &ierr);
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

    /* Sort eigenvalues ALPHA/BETA if desired
       (Workspace: none needed) */
    if (wantst) {

        /* Undo scaling on eigenvalues before selecting */
        if (ilascl)
            clascl("G", 0, 0, anrm, anrmto, n, 1, alpha, n, &ierr);
        if (ilbscl)
            clascl("G", 0, 0, bnrm, bnrmto, n, 1, beta, n, &ierr);

        /* Select eigenvalues */
        for (i = 0; i < n; i++) {
            bwork[i] = selctg(&alpha[i], &beta[i]);
        }

        ctgsen(0, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
               alpha, beta, VSL, ldvsl, VSR, ldvsr, sdim, &pvsl,
               &pvsr, dif, &work[iwrk], lwork - iwrk, idum, 1, &ierr);
        if (ierr == 1)
            *info = n + 3;
    }

    /* Apply back-permutation to VSL and VSR
       (Workspace: none needed) */
    if (ilvsl)
        cggbak("P", "L", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSL, ldvsl, &ierr);
    if (ilvsr)
        cggbak("P", "R", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSR, ldvsr, &ierr);

    /* Undo scaling */
    if (ilascl) {
        clascl("U", 0, 0, anrmto, anrm, n, n, A, lda, &ierr);
        clascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);
    }

    if (ilbscl) {
        clascl("U", 0, 0, bnrmto, bnrm, n, n, B, ldb, &ierr);
        clascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
    }

    if (wantst) {

        /* Check if reordering is correct */
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
