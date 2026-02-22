/**
 * @file dggesx.c
 * @brief DGGESX computes the eigenvalues, the Schur form, and, optionally,
 *        the matrix of Schur vectors for GE matrices with extended options.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * DGGESX computes for a pair of N-by-N real nonsymmetric matrices (A,B),
 * the generalized eigenvalues, the real Schur form (S,T), and,
 * optionally, the left and/or right matrices of Schur vectors (VSL and
 * VSR). This gives the generalized Schur factorization
 *
 *      (A,B) = ( (VSL) S (VSR)**T, (VSL) T (VSR)**T )
 *
 * Optionally, it also orders the eigenvalues so that a selected cluster
 * of eigenvalues appears in the leading diagonal blocks of the upper
 * quasi-triangular matrix S and the upper triangular matrix T; computes
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
 *                        On exit, A has been overwritten by its Schur form S.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       On entry, the second of the pair of matrices.
 *                        On exit, B has been overwritten by its Schur form T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    sdim    Number of eigenvalues for which selctg is true.
 * @param[out]    alphar  Real parts of generalized eigenvalues.
 * @param[out]    alphai  Imaginary parts of generalized eigenvalues.
 * @param[out]    beta    Beta values of generalized eigenvalues.
 * @param[out]    VSL     If jobvsl = 'V', the left Schur vectors.
 * @param[in]     ldvsl   The leading dimension of VSL.
 * @param[out]    VSR     If jobvsr = 'V', the right Schur vectors.
 * @param[in]     ldvsr   The leading dimension of VSR.
 * @param[out]    rconde  Reciprocal condition numbers for eigenvalues (dimension 2).
 * @param[out]    rcondv  Reciprocal condition numbers for subspaces (dimension 2).
 * @param[out]    work    Workspace array.
 * @param[in]     lwork   The dimension of work.
 * @param[out]    iwork   Integer workspace array.
 * @param[in]     liwork  The dimension of iwork.
 * @param[out]    bwork   Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: errors from QZ iteration or reordering
 */
void dggesx(const char* jobvsl, const char* jobvsr, const char* sort,
            dselect3_t selctg, const char* sense, const INT n,
            f64* restrict A, const INT lda,
            f64* restrict B, const INT ldb,
            INT* sdim,
            f64* restrict alphar, f64* restrict alphai,
            f64* restrict beta,
            f64* restrict VSL, const INT ldvsl,
            f64* restrict VSR, const INT ldvsr,
            f64* restrict rconde, f64* restrict rcondv,
            f64* restrict work, const INT lwork,
            INT* restrict iwork, const INT liwork,
            INT* restrict bwork, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, lst2sl;
    INT wantsb, wantse, wantsn, wantst, wantsv;
    INT i, icols, ierr, ihi, ijob, ijobvl, ijobvr;
    INT ileft, ilo, ip, iright, irows, itau, iwrk;
    INT liwmin, lwrk, maxwrk, minwrk;
    f64 anrm, anrmto = 0.0, bignum, bnrm, bnrmto = 0.0, eps;
    f64 pl, pr, safmax, safmin, smlnum;
    f64 dif[2];
    INT nb_geqrf, nb_ormqr, nb_orgqr;

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
        *info = -16;
    } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
        *info = -18;
    }

    if (*info == 0) {
        if (n > 0) {
            minwrk = 8 * n > 6 * n + 16 ? 8 * n : 6 * n + 16;
            nb_geqrf = lapack_get_nb("GEQRF");
            nb_ormqr = lapack_get_nb("ORMQR");
            nb_orgqr = lapack_get_nb("ORGQR");
            maxwrk = minwrk - n + n * nb_geqrf;
            maxwrk = maxwrk > (minwrk - n + n * nb_ormqr) ?
                     maxwrk : (minwrk - n + n * nb_ormqr);
            if (ilvsl) {
                maxwrk = maxwrk > (minwrk - n + n * nb_orgqr) ?
                         maxwrk : (minwrk - n + n * nb_orgqr);
            }
            lwrk = maxwrk;
            if (ijob >= 1) {
                lwrk = lwrk > (n * n / 2) ? lwrk : (n * n / 2);
            }
        } else {
            minwrk = 1;
            maxwrk = 1;
            lwrk = 1;
        }
        work[0] = (f64)lwrk;
        if (wantsn || n == 0) {
            liwmin = 1;
        } else {
            liwmin = n + 6;
        }
        iwork[0] = liwmin;

        if (lwork < minwrk && !lquery) {
            *info = -22;
        } else if (liwork < liwmin && !lquery) {
            *info = -24;
        }
    }

    if (*info != 0) {
        xerbla("DGGESX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        *sdim = 0;
        return;
    }

    eps = dlamch("P");
    safmin = dlamch("S");
    safmax = ONE / safmin;
    smlnum = sqrt(safmin) / eps;
    bignum = ONE / smlnum;

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

    ileft = 0;
    iright = n;
    iwrk = iright + n;
    dggbal("P", n, A, lda, B, ldb, &ilo, &ihi, &work[ileft],
           &work[iright], &work[iwrk], &ierr);

    irows = ihi + 1 - ilo;
    icols = n - ilo;
    itau = iwrk;
    iwrk = itau + irows;
    dgeqrf(irows, icols, &B[ilo + ilo * ldb], ldb, &work[itau],
           &work[iwrk], lwork - iwrk, &ierr);

    dormqr("L", "T", irows, icols, irows, &B[ilo + ilo * ldb], ldb,
           &work[itau], &A[ilo + ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    if (ilvsl) {
        dlaset("Full", n, n, ZERO, ONE, VSL, ldvsl);
        if (irows > 1) {
            dlacpy("L", irows - 1, irows - 1, &B[(ilo + 1) + ilo * ldb], ldb,
                   &VSL[(ilo + 1) + ilo * ldvsl], ldvsl);
        }
        dorgqr(irows, irows, irows, &VSL[ilo + ilo * ldvsl], ldvsl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvsr)
        dlaset("Full", n, n, ZERO, ONE, VSR, ldvsr);

    dgghrd(jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb, VSL, ldvsl, VSR, ldvsr, &ierr);

    *sdim = 0;

    iwrk = itau;
    dhgeqz("S", jobvsl, jobvsr, n, ilo, ihi, A, lda, B, ldb,
           alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr,
           &work[iwrk], lwork - iwrk, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L60;
    }

    if (wantst) {

        if (ilascl) {
            dlascl("G", 0, 0, anrmto, anrm, n, 1, alphar, n, &ierr);
            dlascl("G", 0, 0, anrmto, anrm, n, 1, alphai, n, &ierr);
        }
        if (ilbscl)
            dlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

        for (i = 0; i < n; i++) {
            bwork[i] = selctg(&alphar[i], &alphai[i], &beta[i]);
        }

        dtgsen(ijob, ilvsl, ilvsr, bwork, n, A, lda, B, ldb,
               alphar, alphai, beta, VSL, ldvsl, VSR, ldvsr,
               sdim, &pl, &pr, dif, &work[iwrk], lwork - iwrk,
               iwork, liwork, &ierr);

        if (ijob >= 1) {
            maxwrk = maxwrk > (2 * (*sdim) * (n - (*sdim))) ?
                     maxwrk : (2 * (*sdim) * (n - (*sdim)));
        }
        if (ierr == -22) {
            *info = -22;
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
        dggbak("P", "L", n, ilo, ihi, &work[ileft],
               &work[iright], n, VSL, ldvsl, &ierr);

    if (ilvsr)
        dggbak("P", "R", n, ilo, ihi, &work[ileft],
               &work[iright], n, VSR, ldvsr, &ierr);

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
                    cursl = cursl || lastsl;
                    lastsl = cursl;
                    if (cursl)
                        *sdim = *sdim + 2;
                    ip = -1;
                    if (cursl && !lst2sl)
                        *info = n + 2;
                } else {
                    ip = 1;
                }
            }
            lst2sl = lastsl;
            lastsl = cursl;
        }
    }

L60:
    work[0] = (f64)maxwrk;
    iwork[0] = liwmin;

    return;
}
