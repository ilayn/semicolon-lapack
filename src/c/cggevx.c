/**
 * @file cggevx.c
 * @brief CGGEVX computes the eigenvalues and, optionally, the left and/or
 *        right eigenvectors for GE matrices with extended options.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CGGEVX computes for a pair of N-by-N complex nonsymmetric matrices (A,B)
 * the generalized eigenvalues, and optionally, the left and/or right
 * generalized eigenvectors.
 *
 * Optionally, it also computes a balancing transformation to improve
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
 * @param[out]    alpha   Complex array, dimension (n).
 * @param[out]    beta    Complex array, dimension (n).
 *                        On exit, ALPHA(j)/BETA(j), j=0,...,n-1, will be the
 *                        generalized eigenvalues.
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
 * @param[out]    work    Complex workspace array.
 * @param[in]     lwork   The dimension of work.
 * @param[out]    rwork   Single precision workspace array.
 * @param[out]    iwork   Integer workspace array.
 * @param[out]    bwork   Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: errors from QZ iteration or eigenvector computation
 */
void cggevx(const char* balanc, const char* jobvl, const char* jobvr,
            const char* sense, const int n,
            c64* restrict A, const int lda,
            c64* restrict B, const int ldb,
            c64* restrict alpha,
            c64* restrict beta,
            c64* restrict VL, const int ldvl,
            c64* restrict VR, const int ldvr,
            int* ilo, int* ihi,
            f32* restrict lscale, f32* restrict rscale,
            f32* abnrm, f32* bbnrm,
            f32* restrict rconde, f32* restrict rcondv,
            c64* restrict work, const int lwork,
            f32* restrict rwork,
            int* restrict iwork, int* restrict bwork,
            int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
    int wantsb, wantse, wantsn, wantsv;
    int i, icols, ierr, ijobvl, ijobvr, in, irows;
    int itau, iwrk, iwrk1, j, jc, jr, m, maxwrk, minwrk;
    f32 anrm, anrmto = 0.0f, bignum, bnrm, bnrmto = 0.0f, eps, smlnum, temp;
    int ldumma[1];
    int nb_geqrf, nb_unmqr, nb_ungqr;

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
        *info = -13;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        *info = -15;
    }

    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            minwrk = 2 * n;
            if (wantse) {
                minwrk = 4 * n;
            } else if (wantsv || wantsb) {
                minwrk = 2 * n * (n + 1);
            }
            maxwrk = minwrk;
            nb_geqrf = lapack_get_nb("GEQRF");
            nb_unmqr = lapack_get_nb("ORMQR");
            nb_ungqr = lapack_get_nb("ORGQR");
            maxwrk = maxwrk > (n + n * nb_geqrf) ? maxwrk : (n + n * nb_geqrf);
            maxwrk = maxwrk > (n + n * nb_unmqr) ? maxwrk : (n + n * nb_unmqr);
            if (ilvl) {
                maxwrk = maxwrk > (n + n * nb_ungqr) ? maxwrk : (n + n * nb_ungqr);
            }
        }
        work[0] = CMPLXF((f32)maxwrk, 0.0f);

        if (lwork < minwrk && !lquery) {
            *info = -25;
        }
    }

    if (*info != 0) {
        xerbla("CGGEVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    /* Get machine constants */
    eps = slamch("P");
    smlnum = slamch("S");
    bignum = ONE / smlnum;
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

    /* Permute and/or balance the matrix pair (A,B) */
    cggbal(balanc, n, A, lda, B, ldb, ilo, ihi, lscale, rscale, rwork, &ierr);

    /* Compute ABNRM and BBNRM */
    *abnrm = clange("1", n, n, A, lda, rwork);
    if (ilascl) {
        rwork[0] = *abnrm;
        slascl("G", 0, 0, anrmto, anrm, 1, 1, rwork, 1, &ierr);
        *abnrm = rwork[0];
    }

    *bbnrm = clange("1", n, n, B, ldb, rwork);
    if (ilbscl) {
        rwork[0] = *bbnrm;
        slascl("G", 0, 0, bnrmto, bnrm, 1, 1, rwork, 1, &ierr);
        *bbnrm = rwork[0];
    }

    /* Reduce B to triangular form (QR decomposition of B) */
    irows = *ihi + 1 - *ilo;
    if (ilv || !wantsn) {
        icols = n - *ilo;
    } else {
        icols = irows;
    }
    itau = 0;
    iwrk = itau + irows;
    cgeqrf(irows, icols, &B[*ilo + *ilo * ldb], ldb, &work[itau],
           &work[iwrk], lwork - iwrk, &ierr);

    /* Apply the unitary transformation to A */
    cunmqr("L", "C", irows, icols, irows, &B[*ilo + *ilo * ldb], ldb,
           &work[itau], &A[*ilo + *ilo * lda], lda, &work[iwrk],
           lwork - iwrk, &ierr);

    /* Initialize VL and/or VR */
    if (ilvl) {
        claset("Full", n, n, CZERO, CONE, VL, ldvl);
        if (irows > 1) {
            clacpy("L", irows - 1, irows - 1, &B[(*ilo + 1) + *ilo * ldb], ldb,
                   &VL[(*ilo + 1) + *ilo * ldvl], ldvl);
        }
        cungqr(irows, irows, irows, &VL[*ilo + *ilo * ldvl], ldvl,
               &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    if (ilvr)
        claset("Full", n, n, CZERO, CONE, VR, ldvr);

    /* Reduce to generalized Hessenberg form */
    if (ilv || !wantsn) {
        cgghrd(jobvl, jobvr, n, *ilo, *ihi, A, lda, B, ldb, VL, ldvl, VR, ldvr, &ierr);
    } else {
        cgghrd("N", "N", irows, 0, irows - 1, &A[*ilo + *ilo * lda], lda,
               &B[*ilo + *ilo * ldb], ldb, VL, ldvl, VR, ldvr, &ierr);
    }

    /* Perform QZ algorithm (Compute eigenvalues, and optionally, the
       Schur forms and Schur vectors) */
    iwrk = itau;
    const char* chtemp = (ilv || !wantsn) ? "S" : "E";

    chgeqz(chtemp, jobvl, jobvr, n, *ilo, *ihi, A, lda, B, ldb,
           alpha, beta, VL, ldvl, VR, ldvr,
           &work[iwrk], lwork - iwrk, rwork, &ierr);
    if (ierr != 0) {
        if (ierr > 0 && ierr <= n) {
            *info = ierr;
        } else if (ierr > n && ierr <= 2 * n) {
            *info = ierr - n;
        } else {
            *info = n + 1;
        }
        goto L90;
    }

    /* Compute Eigenvectors and estimate condition numbers if desired */
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

            ctgevc(side, "B", ldumma, n, A, lda, B, ldb, VL, ldvl,
                   VR, ldvr, n, &in, &work[iwrk], rwork, &ierr);
            if (ierr != 0) {
                *info = n + 2;
                goto L90;
            }
        }

        if (!wantsn) {

            for (i = 0; i < n; i++) {

                for (j = 0; j < n; j++) {
                    bwork[j] = 0;
                }
                bwork[i] = 1;

                iwrk = n;
                iwrk1 = iwrk + n;

                if (wantse || wantsb) {
                    ctgevc("B", "S", bwork, n, A, lda, B, ldb,
                           work, n, &work[iwrk], n, 1, &m,
                           &work[iwrk1], rwork, &ierr);
                    if (ierr != 0) {
                        *info = n + 2;
                        goto L90;
                    }
                }

                ctgsna(sense, "S", bwork, n, A, lda, B, ldb,
                       work, n, &work[iwrk], n, &rconde[i],
                       &rcondv[i], 1, &m, &work[iwrk1],
                       lwork - iwrk1, iwork, &ierr);
            }
        }
    }

    /* Undo balancing on VL and VR and normalization */
    if (ilvl) {
        cggbak(balanc, "L", n, *ilo, *ihi, lscale, rscale, n, VL, ldvl, &ierr);

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
        cggbak(balanc, "R", n, *ilo, *ihi, lscale, rscale, n, VR, ldvr, &ierr);

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

    /* Undo scaling if necessary */
L90:
    if (ilascl)
        clascl("G", 0, 0, anrmto, anrm, n, 1, alpha, n, &ierr);

    if (ilbscl)
        clascl("G", 0, 0, bnrmto, bnrm, n, 1, beta, n, &ierr);

    work[0] = CMPLXF((f32)maxwrk, 0.0f);
    return;
}
