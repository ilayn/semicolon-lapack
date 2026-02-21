/**
 * @file zggesx.c
 * @brief ZGGESX computes the eigenvalues, the Schur form, and, optionally,
 *        the matrix of Schur vectors for GE matrices with extended options.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZGGESX computes for a pair of N-by-N complex nonsymmetric matrices
 * (A,B), the generalized eigenvalues, the complex Schur form (S,T),
 * and, optionally, the left and/or right matrices of Schur vectors (VSL
 * and VSR). This gives the generalized Schur factorization
 *
 *      (A,B) = ( (VSL) S (VSR)**H, (VSL) T (VSR)**H )
 *
 * where (VSR)**H is the conjugate-transpose of VSR.
 *
 * Optionally, it also orders the eigenvalues so that a selected cluster
 * of eigenvalues appears in the leading diagonal blocks of the upper
 * triangular matrix S and the upper triangular matrix T; computes
 * a reciprocal condition number for the average of the selected
 * eigenvalues (RCONDE); and computes a reciprocal condition number for
 * the right and left deflating subspaces corresponding to the selected
 * eigenvalues (RCONDV).
 *
 * @param[in]     jobvsl  = 'N': do not compute the left Schur vectors;
 *                         = 'V': compute the left Schur vectors.
 * @param[in]     jobvsr  = 'N': do not compute the right Schur vectors;
 *                         = 'V': compute the right Schur vectors.
 * @param[in]     sort    = 'N': Eigenvalues are not ordered;
 *                         = 'S': Eigenvalues are ordered (see selctg).
 * @param[in]     selctg  Selection function for eigenvalue ordering.
 * @param[in]     sense   Determines which reciprocal condition numbers are computed.
 *                         = 'N': None are computed;
 *                         = 'E': Computed for average of selected eigenvalues only;
 *                         = 'V': Computed for selected deflating subspaces only;
 *                         = 'B': Computed for both.
 * @param[in]     n       The order of the matrices A, B, VSL, and VSR. n >= 0.
 * @param[in,out] A       On entry, the first of the pair of matrices.
 *                        On exit, A has been overwritten by its generalized Schur
 *                        form S.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       On entry, the second of the pair of matrices.
 *                        On exit, B has been overwritten by its generalized Schur
 *                        form T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    sdim    Number of eigenvalues for which selctg is true.
 * @param[out]    alpha   Complex array, dimension (n).
 * @param[out]    beta    Complex array, dimension (n).
 * @param[out]    VSL     If jobvsl = 'V', the left Schur vectors.
 * @param[in]     ldvsl   The leading dimension of VSL.
 * @param[out]    VSR     If jobvsr = 'V', the right Schur vectors.
 * @param[in]     ldvsr   The leading dimension of VSR.
 * @param[out]    rconde  Reciprocal condition numbers for eigenvalues (dimension 2).
 * @param[out]    rcondv  Reciprocal condition numbers for subspaces (dimension 2).
 * @param[out]    work    Complex workspace array, dimension (max(1,lwork)).
 * @param[in]     lwork   The dimension of work.
 * @param[out]    rwork   Double precision array, dimension (8*n).
 * @param[out]    iwork   Integer workspace array.
 * @param[in]     liwork  The dimension of iwork.
 * @param[out]    bwork   Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: errors from QZ iteration or reordering
 */
void zggesx(const char* jobvsl, const char* jobvsr, const char* sort,
            zselect2_t selctg, const char* sense, const int n,
            c128* A, const int lda,
            c128* B, const int ldb,
            int* sdim,
            c128* alpha, c128* beta,
            c128* VSL, const int ldvsl,
            c128* VSR, const int ldvsr,
            f64* rconde, f64* rcondv,
            c128* work, const int lwork,
            f64* rwork,
            int* iwork, const int liwork,
            int* bwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    int cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery;
    int wantsb, wantse, wantsn, wantst, wantsv;
    int i, icols, ierr, ihi, ijob, ijobvl, ijobvr;
    int ileft, ilo, iright, irows, irwrk, itau, iwrk;
    int liwmin, lwrk, maxwrk, minwrk;
    f64 anrm, anrmto = 0.0, bignum, bnrm, bnrmto = 0.0, eps;
    f64 pl, pr, smlnum;
    f64 dif[2];
    int nb_geqrf, nb_unmqr, nb_ungqr;

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
    wantsn = (sense[0] == 'N' || sense[0] == 'n');
    wantse = (sense[0] == 'E' || sense[0] == 'e');
    wantsv = (sense[0] == 'V' || sense[0] == 'v');
    wantsb = (sense[0] == 'B' || sense[0] == 'b');
    lquery = (lwork == -1 || liwork == -1);

    if (wantsn) {
        ijob = 0;
    } else if (wantse) {
        ijob = 1;
    } else if (wantsv) {
        ijob = 2;
    } else if (wantsb) {
        ijob = 4;
    } else {
        ijob = 0;
    }

    *info = 0;
    if (ijobvl <= 0) {
        *info = -1;
    } else if (ijobvr <= 0) {
        *info = -2;
    } else if (!wantst && !(sort[0] == 'N' || sort[0] == 'n')) {
        *info = -3;
    } else if (!(wantsn || wantse || wantsv || wantsb) ||
               (!wantst && !wantsn)) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldvsl < 1 || (ilvsl && ldvsl < n)) {
        *info = -15;
    } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
        *info = -17;
    }

    if (*info == 0) {
        if (n > 0) {
            minwrk = 2 * n;
            nb_geqrf = lapack_get_nb("GEQRF");
            nb_unmqr = lapack_get_nb("ORMQR");
            nb_ungqr = lapack_get_nb("ORGQR");
            maxwrk = n * (1 + nb_geqrf);
            maxwrk = maxwrk > (n * (1 + nb_unmqr)) ?
                     maxwrk : (n * (1 + nb_unmqr));
            if (ilvsl) {
                maxwrk = maxwrk > (n * (1 + nb_ungqr)) ?
                         maxwrk : (n * (1 + nb_ungqr));
            }
            lwrk = maxwrk;
            if (ijob >= 1)
                lwrk = lwrk > (n * n / 2) ? lwrk : (n * n / 2);
        } else {
            minwrk = 1;
            maxwrk = 1;
            lwrk = 1;
        }
        work[0] = CMPLX((f64)lwrk, 0.0);
        if (wantsn || n == 0) {
            liwmin = 1;
        } else {
            liwmin = n + 2;
        }
        iwork[0] = liwmin;

        if (lwork < minwrk && !lquery) {
            *info = -21;
        } else if (liwork < liwmin && !lquery) {
            *info = -24;
        }
    }

    if (*info != 0) {
        xerbla("ZGGESX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        *sdim = 0;
        return;
    }

    eps = dlamch("P");
    smlnum = dlamch("S");
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
    icols = n - ilo;
    itau = 0;
    iwrk = itau + irows;
    zgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb,
           &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    zunmqr("L", "C", irows, icols, irows,
           &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda,
           &work[iwrk], lwork - iwrk, &ierr);

    if (ilvsl) {
        zlaset("Full", n, n, CZERO, CONE, VSL, ldvsl);
        if (irows > 1) {
            zlacpy("L", irows - 1, irows - 1,
                   &B[(ilo + 1) + ilo * ldb], ldb,
                   &VSL[(ilo + 1) + ilo * ldvsl], ldvsl);
        }
        zungqr(irows, irows, irows,
               &VSL[ilo + ilo * ldvsl], ldvsl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvsr)
        zlaset("Full", n, n, CZERO, CONE, VSR, ldvsr);

    zgghrd(jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb, VSL,
           ldvsl, VSR, ldvsr, &ierr);

    *sdim = 0;

    iwrk = itau;
    zhgeqz("S", jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb,
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
        goto L40;
    }

    if (wantst) {

        if (ilascl)
            zlascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);
        if (ilbscl)
            zlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

        for (i = 0; i < n; i++) {
            bwork[i] = selctg(&alpha[i], &beta[i]);
        }

        ztgsen(ijob, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
               alpha, beta, VSL, ldvsl, VSR, ldvsr,
               sdim, &pl, &pr, dif, &work[iwrk], lwork - iwrk,
               iwork, liwork, &ierr);

        if (ijob >= 1) {
            maxwrk = maxwrk > (2 * (*sdim) * (n - (*sdim))) ?
                     maxwrk : (2 * (*sdim) * (n - (*sdim)));
        }
        if (ierr == -21) {

            *info = -21;
        } else {
            if (ijob == 1 || ijob == 4) {
                rconde[0] = pl;
                rconde[1] = pr;
            }
            if (ijob == 2 || ijob == 4) {
                rcondv[0] = dif[0];
                rcondv[1] = dif[1];
            }
            if (ierr == 1)
                *info = n + 3;
        }

    }

    if (ilvsl)
        zggbak("P", "L", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSL, ldvsl, &ierr);

    if (ilvsr)
        zggbak("P", "R", n, ilo, ihi, &rwork[ileft],
               &rwork[iright], n, VSR, ldvsr, &ierr);

    if (ilascl) {
        zlascl("U", 0, 0, anrmto, anrm, n, n, A, lda, &ierr);
        zlascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);
    }

    if (ilbscl) {
        zlascl("U", 0, 0, bnrmto, bnrm, n, n, B, ldb, &ierr);
        zlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
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

L40:
    work[0] = CMPLX((f64)maxwrk, 0.0);
    iwork[0] = liwmin;

    return;
}
