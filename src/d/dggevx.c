/**
 * @file dggevx.c
 * @brief DGGEVX computes the eigenvalues and, optionally, the left and/or
 *        right eigenvectors for GE matrices with extended options.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * DGGEVX computes for a pair of N-by-N real nonsymmetric matrices (A,B)
 * the generalized eigenvalues, and optionally, the left and/or right
 * generalized eigenvectors.
 *
 * Optionally also, it computes a balancing transformation to improve
 * the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
 * LSCALE, RSCALE, ABNRM, and BBNRM), reciprocal condition numbers for
 * the eigenvalues (RCONDE), and reciprocal condition numbers for the
 * right eigenvectors (RCONDV).
 *
 * @param[in]     balanc  Specifies the balance option to be performed.
 *                        = 'N': do not diagonally scale or permute;
 *                        = 'P': permute only;
 *                        = 'S': scale only;
 *                        = 'B': both permute and scale.
 * @param[in]     jobvl   = 'N': do not compute the left generalized eigenvectors;
 *                        = 'V': compute the left generalized eigenvectors.
 * @param[in]     jobvr   = 'N': do not compute the right generalized eigenvectors;
 *                        = 'V': compute the right generalized eigenvectors.
 * @param[in]     sense   Determines which reciprocal condition numbers are computed.
 *                        = 'N': none are computed;
 *                        = 'E': computed for eigenvalues only;
 *                        = 'V': computed for eigenvectors only;
 *                        = 'B': computed for eigenvalues and eigenvectors.
 * @param[in]     n       The order of the matrices A, B, VL, and VR. n >= 0.
 * @param[in,out] A       On entry, the matrix A in the pair (A,B).
 *                        On exit, A has been overwritten.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       On entry, the matrix B in the pair (A,B).
 *                        On exit, B has been overwritten.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    alphar  Real parts of generalized eigenvalues.
 * @param[out]    alphai  Imaginary parts of generalized eigenvalues.
 * @param[out]    beta    Beta values of generalized eigenvalues.
 * @param[out]    VL      If jobvl = 'V', the left eigenvectors.
 * @param[in]     ldvl    The leading dimension of VL.
 * @param[out]    VR      If jobvr = 'V', the right eigenvectors.
 * @param[in]     ldvr    The leading dimension of VR.
 * @param[out]    ilo     See ihi.
 * @param[out]    ihi     ILO and IHI are integer values such that on exit
 *                        A(i,j) = 0 and B(i,j) = 0 if i > j and
 *                        j = 0,...,ILO-1 or i = IHI+1,...,N-1.
 *                        If balanc = 'N' or 'S', ILO = 0 and IHI = N-1.
 * @param[out]    lscale  Array of dimension (n). Details of the permutations and
 *                        scaling factors applied to the left side of A and B.
 *                        If PL(j) is the index of the row interchanged with row j,
 *                        and DL(j) is the scaling factor applied to row j, then
 *                          lscale[j] = PL(j)    for j = 0,...,ILO-1
 *                                    = DL(j)    for j = ILO,...,IHI
 *                                    = PL(j)    for j = IHI+1,...,N-1.
 *                        The order in which the interchanges are made is N-1 to IHI+1,
 *                        then 0 to ILO-1.
 * @param[out]    rscale  Array of dimension (n). Details of the permutations and
 *                        scaling factors applied to the right side of A and B.
 *                        If PR(j) is the index of the column interchanged with column j,
 *                        and DR(j) is the scaling factor applied to column j, then
 *                          rscale[j] = PR(j)    for j = 0,...,ILO-1
 *                                    = DR(j)    for j = ILO,...,IHI
 *                                    = PR(j)    for j = IHI+1,...,N-1.
 *                        The order in which the interchanges are made is N-1 to IHI+1,
 *                        then 0 to ILO-1.
 * @param[out]    abnrm   The one-norm of the balanced matrix A.
 * @param[out]    bbnrm   The one-norm of the balanced matrix B.
 * @param[out]    rconde  Reciprocal condition numbers for eigenvalues.
 * @param[out]    rcondv  Reciprocal condition numbers for eigenvectors.
 * @param[out]    work    Workspace array.
 * @param[in]     lwork   The dimension of work.
 * @param[out]    iwork   Integer workspace array.
 * @param[out]    bwork   Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: errors from QZ iteration or eigenvector computation
 */
void dggevx(const char* balanc, const char* jobvl, const char* jobvr,
            const char* sense, const int n,
            f64* restrict A, const int lda,
            f64* restrict B, const int ldb,
            f64* restrict alphar, f64* restrict alphai,
            f64* restrict beta,
            f64* restrict VL, const int ldvl,
            f64* restrict VR, const int ldvr,
            int* ilo, int* ihi,
            f64* restrict lscale, f64* restrict rscale,
            f64* abnrm, f64* bbnrm,
            f64* restrict rconde, f64* restrict rcondv,
            f64* restrict work, const int lwork,
            int* restrict iwork, int* restrict bwork,
            int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int ilascl, ilbscl, ilv, ilvl, ilvr, lquery, noscl;
    int pair, wantsb, wantse, wantsn, wantsv;
    int i, icols, ierr, ijobvl, ijobvr, in, irows;
    int itau, iwrk, iwrk1, j, jc, jr, m, maxwrk, minwrk, mm;
    f64 anrm, anrmto = 0.0, bignum, bnrm, bnrmto = 0.0, eps, smlnum, temp;
    int ldumma[1];
    int nb_geqrf, nb_ormqr, nb_orgqr;

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

    noscl = (balanc[0] == 'N' || balanc[0] == 'n' ||
             balanc[0] == 'P' || balanc[0] == 'p');
    wantsn = (sense[0] == 'N' || sense[0] == 'n');
    wantse = (sense[0] == 'E' || sense[0] == 'e');
    wantsv = (sense[0] == 'V' || sense[0] == 'v');
    wantsb = (sense[0] == 'B' || sense[0] == 'b');

    *info = 0;
    lquery = (lwork == -1);
    if (!(balanc[0] == 'N' || balanc[0] == 'n' ||
          balanc[0] == 'S' || balanc[0] == 's' ||
          balanc[0] == 'P' || balanc[0] == 'p' ||
          balanc[0] == 'B' || balanc[0] == 'b')) {
        *info = -1;
    } else if (ijobvl <= 0) {
        *info = -2;
    } else if (ijobvr <= 0) {
        *info = -3;
    } else if (!(wantsn || wantse || wantsb || wantsv)) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldvl < 1 || (ilvl && ldvl < n)) {
        *info = -14;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        *info = -16;
    }

    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            if (noscl && !ilv) {
                minwrk = 2 * n;
            } else {
                minwrk = 6 * n;
            }
            if (wantse || wantsb) {
                minwrk = 10 * n;
            }
            if (wantsv || wantsb) {
                int tmp = 2 * n * (n + 4) + 16;
                minwrk = minwrk > tmp ? minwrk : tmp;
            }
            maxwrk = minwrk;
            nb_geqrf = lapack_get_nb("GEQRF");
            nb_ormqr = lapack_get_nb("ORMQR");
            nb_orgqr = lapack_get_nb("ORGQR");
            maxwrk = maxwrk > (n + n * nb_geqrf) ? maxwrk : (n + n * nb_geqrf);
            maxwrk = maxwrk > (n + n * nb_ormqr) ? maxwrk : (n + n * nb_ormqr);
            if (ilvl) {
                maxwrk = maxwrk > (n + n * nb_orgqr) ? maxwrk : (n + n * nb_orgqr);
            }
        }
        work[0] = (f64)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -26;
        }
    }

    if (*info != 0) {
        xerbla("DGGEVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    eps = dlamch("P");
    smlnum = dlamch("S");
    smlnum = sqrt(smlnum) / eps;
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

    dggbal(balanc, n, A, lda, B, ldb, ilo, ihi, lscale, rscale, work, &ierr);

    *abnrm = dlange("1", n, n, A, lda, work);
    if (ilascl) {
        work[0] = *abnrm;
        dlascl("G", 0, 0, anrmto, anrm, 1, 1, work, 1, &ierr);
        *abnrm = work[0];
    }

    *bbnrm = dlange("1", n, n, B, ldb, work);
    if (ilbscl) {
        work[0] = *bbnrm;
        dlascl("G", 0, 0, bnrmto, bnrm, 1, 1, work, 1, &ierr);
        *bbnrm = work[0];
    }

    irows = *ihi + 1 - *ilo;
    if (ilv || !wantsn) {
        icols = n - *ilo;
    } else {
        icols = irows;
    }
    itau = 0;
    iwrk = itau + irows;
    dgeqrf(irows, icols, &B[*ilo + *ilo * ldb], ldb, &work[itau],
           &work[iwrk], lwork - iwrk, &ierr);

    dormqr("L", "T", irows, icols, irows, &B[*ilo + *ilo * ldb], ldb,
           &work[itau], &A[*ilo + *ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    if (ilvl) {
        dlaset("Full", n, n, ZERO, ONE, VL, ldvl);
        if (irows > 1) {
            dlacpy("L", irows - 1, irows - 1, &B[(*ilo + 1) + *ilo * ldb], ldb,
                   &VL[(*ilo + 1) + *ilo * ldvl], ldvl);
        }
        dorgqr(irows, irows, irows, &VL[*ilo + *ilo * ldvl], ldvl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvr)
        dlaset("Full", n, n, ZERO, ONE, VR, ldvr);

    if (ilv || !wantsn) {
        dgghrd(jobvl, jobvr, n, *ilo, *ihi, A, lda, B, ldb, VL, ldvl, VR, ldvr, &ierr);
    } else {
        dgghrd("N", "N", irows, 0, irows - 1, &A[*ilo + *ilo * lda], lda,
               &B[*ilo + *ilo * ldb], ldb, VL, ldvl, VR, ldvr, &ierr);
    }

    const char* chtemp = (ilv || !wantsn) ? "S" : "E";
    dhgeqz(chtemp, jobvl, jobvr, n, *ilo, *ihi, A, lda, B, ldb,
           alphar, alphai, beta, VL, ldvl, VR, ldvr, work, lwork, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L130;
    }

    if (ilv || !wantsn) {
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

            dtgevc(side, "B", ldumma, n, A, lda, B, ldb, VL, ldvl,
                   VR, ldvr, n, &in, work, &ierr);
            if (ierr != 0) {
                *info = n + 2;
                goto L130;
            }
        }

        if (!wantsn) {

            pair = 0;
            for (i = 0; i < n; i++) {

                if (pair) {
                    pair = 0;
                    continue;
                }
                mm = 1;
                if (i < n - 1) {
                    if (A[(i + 1) + i * lda] != ZERO) {
                        pair = 1;
                        mm = 2;
                    }
                }

                for (j = 0; j < n; j++) {
                    bwork[j] = 0;
                }
                if (mm == 1) {
                    bwork[i] = 1;
                } else if (mm == 2) {
                    bwork[i] = 1;
                    bwork[i + 1] = 1;
                }

                iwrk = mm * n;
                iwrk1 = iwrk + mm * n;

                if (wantse || wantsb) {
                    dtgevc("B", "S", bwork, n, A, lda, B, ldb,
                           work, n, &work[iwrk], n, mm, &m,
                           &work[iwrk1], &ierr);
                    if (ierr != 0) {
                        *info = n + 2;
                        goto L130;
                    }
                }

                dtgsna(sense, "S", bwork, n, A, lda, B, ldb,
                       work, n, &work[iwrk], n, &rconde[i],
                       &rcondv[i], mm, &m, &work[iwrk1],
                       lwork - iwrk1, iwork, &ierr);
            }
        }
    }

    if (ilvl) {
        dggbak(balanc, "L", n, *ilo, *ihi, lscale, rscale, n, VL, ldvl, &ierr);

        for (jc = 0; jc < n; jc++) {
            if (alphai[jc] < ZERO)
                continue;
            temp = ZERO;
            if (alphai[jc] == ZERO) {
                for (jr = 0; jr < n; jr++) {
                    temp = temp > fabs(VL[jr + jc * ldvl]) ?
                           temp : fabs(VL[jr + jc * ldvl]);
                }
            } else {
                for (jr = 0; jr < n; jr++) {
                    temp = temp > (fabs(VL[jr + jc * ldvl]) +
                                   fabs(VL[jr + (jc + 1) * ldvl])) ?
                           temp : (fabs(VL[jr + jc * ldvl]) +
                                   fabs(VL[jr + (jc + 1) * ldvl]));
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
        dggbak(balanc, "R", n, *ilo, *ihi, lscale, rscale, n, VR, ldvr, &ierr);

        for (jc = 0; jc < n; jc++) {
            if (alphai[jc] < ZERO)
                continue;
            temp = ZERO;
            if (alphai[jc] == ZERO) {
                for (jr = 0; jr < n; jr++) {
                    temp = temp > fabs(VR[jr + jc * ldvr]) ?
                           temp : fabs(VR[jr + jc * ldvr]);
                }
            } else {
                for (jr = 0; jr < n; jr++) {
                    temp = temp > (fabs(VR[jr + jc * ldvr]) +
                                   fabs(VR[jr + (jc + 1) * ldvr])) ?
                           temp : (fabs(VR[jr + jc * ldvr]) +
                                   fabs(VR[jr + (jc + 1) * ldvr]));
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

L130:
    if (ilascl) {
        dlascl("G", 0, 0, anrmto, anrm, n, 1, alphar, n, &ierr);
        dlascl("G", 0, 0, anrmto, anrm, n, 1, alphai, n, &ierr);
    }

    if (ilbscl) {
        dlascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);
    }

    work[0] = (f64)maxwrk;
    return;
}
