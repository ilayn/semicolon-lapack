/**
 * @file sgeesx.c
 * @brief SGEESX computes Schur form with optional eigenvalue ordering and condition numbers.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * SGEESX computes for an N-by-N real nonsymmetric matrix A, the
 * eigenvalues, the real Schur form T, and, optionally, the matrix of
 * Schur vectors Z.  This gives the Schur factorization A = Z*T*(Z**T).
 *
 * Optionally, it also orders the eigenvalues on the diagonal of the
 * real Schur form so that selected eigenvalues are at the top left;
 * computes a reciprocal condition number for the average of the
 * selected eigenvalues (RCONDE); and computes a reciprocal condition
 * number for the right invariant subspace corresponding to the
 * selected eigenvalues (RCONDV).  The leading columns of Z form an
 * orthonormal basis for this invariant subspace.
 *
 * A real matrix is in real Schur form if it is upper quasi-triangular
 * with 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in
 * the form
 *         [  a  b  ]
 *         [  c  a  ]
 * where b*c < 0. The eigenvalues of such a block are a +- sqrt(bc).
 *
 * @param[in] jobvs  = 'N': Schur vectors are not computed;
 *                   = 'V': Schur vectors are computed.
 * @param[in] sort   Specifies whether or not to order the eigenvalues:
 *                   = 'N': Eigenvalues are not ordered;
 *                   = 'S': Eigenvalues are ordered (see select).
 * @param[in] select Eigenvalue selection callback. If sort = 'S', select is used
 *                   to select eigenvalues to sort to the top left of the Schur form.
 *                   If sort = 'N', select is not referenced and may be NULL.
 * @param[in] sense  Determines which reciprocal condition numbers are computed:
 *                   = 'N': None are computed;
 *                   = 'E': Computed for average of selected eigenvalues only;
 *                   = 'V': Computed for selected right invariant subspace only;
 *                   = 'B': Computed for both.
 *                   If sense = 'E', 'V' or 'B', sort must equal 'S'.
 * @param[in] n      The order of the matrix A. n >= 0.
 * @param[in,out] A  On entry, the N-by-N matrix A.
 *                   On exit, A has been overwritten by its real Schur form T.
 *                   Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, n).
 * @param[out] sdim  If sort = 'N', sdim = 0.
 *                   If sort = 'S', sdim = number of eigenvalues (after sorting)
 *                   for which select is true. (Complex conjugate pairs for which
 *                   select is true for either eigenvalue count as 2.)
 * @param[out] wr    Array, dimension (n). Real parts of eigenvalues.
 * @param[out] wi    Array, dimension (n). Imaginary parts of eigenvalues.
 * @param[out] VS    If jobvs = 'V', VS contains the orthogonal matrix Z of Schur
 *                   vectors. Dimension (ldvs, n).
 *                   If jobvs = 'N', VS is not referenced.
 * @param[in] ldvs   The leading dimension of VS. ldvs >= 1;
 *                   if jobvs = 'V', ldvs >= n.
 * @param[out] rconde If sense = 'E' or 'B', contains the reciprocal condition number
 *                    for the average of the selected eigenvalues.
 * @param[out] rcondv If sense = 'V' or 'B', contains the reciprocal condition number
 *                    for the selected right invariant subspace.
 * @param[out] work  Workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1, 3*n).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] iwork Integer array, dimension (max(1, liwork)).
 *                   On exit, if info = 0, iwork[0] returns optimal liwork.
 * @param[in] liwork The dimension of iwork. liwork >= 1;
 *                   if sense = 'V' or 'B', liwork >= sdim*(n-sdim).
 *                   If liwork = -1, a workspace query is assumed.
 * @param[out] bwork Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info  = 0: successful exit
 *                   < 0: if info = -i, the i-th argument had an illegal value
 *                   > 0: if info = i, and i is
 *                       <= n: the QR algorithm failed to compute all eigenvalues;
 *                       = n+1: eigenvalues could not be reordered because some
 *                              eigenvalues were too close to separate;
 *                       = n+2: after reordering, roundoff changed values of some
 *                              complex eigenvalues so that leading eigenvalues in
 *                              the Schur form no longer satisfy select=true.
 */
void sgeesx(const char* jobvs, const char* sort, sselect2_t select,
            const char* sense, const int n, float* A, const int lda, int* sdim,
            float* wr, float* wi,
            float* VS, const int ldvs,
            float* rconde, float* rcondv,
            float* work, const int lwork,
            int* iwork, const int liwork, int* bwork, int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int cursl, lastsl, lquery, lst2sl, scalea, wantsb, wantse, wantsn, wantst, wantsv, wantvs;
    int hswork, i, i1, i2, ibal, icond, ierr, ieval;
    int ihi, ilo, inxt, ip, itau, iwrk, liwrk, lwrk, maxwrk, minwrk;
    float anrm, bignum, cscale = ONE, eps, smlnum;
    float dum[1];
    int nb_gehrd, nb_orghr;

    /* Test the input arguments */
    *info = 0;
    wantvs = (jobvs[0] == 'V' || jobvs[0] == 'v');
    wantst = (sort[0] == 'S' || sort[0] == 's');
    wantsn = (sense[0] == 'N' || sense[0] == 'n');
    wantse = (sense[0] == 'E' || sense[0] == 'e');
    wantsv = (sense[0] == 'V' || sense[0] == 'v');
    wantsb = (sense[0] == 'B' || sense[0] == 'b');
    lquery = (lwork == -1) || (liwork == -1);

    if (!wantvs && !(jobvs[0] == 'N' || jobvs[0] == 'n')) {
        *info = -1;
    } else if (!wantst && !(sort[0] == 'N' || sort[0] == 'n')) {
        *info = -2;
    } else if (!(wantsn || wantse || wantsv || wantsb) ||
               (!wantst && !wantsn)) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldvs < 1 || (wantvs && ldvs < n)) {
        *info = -12;
    }

    /* Compute workspace */
    if (*info == 0) {
        liwrk = 1;
        if (n == 0) {
            minwrk = 1;
            lwrk = 1;
        } else {
            nb_gehrd = lapack_get_nb("GEHRD");
            nb_orghr = lapack_get_nb("ORGHR");
            if (nb_orghr == 1) nb_orghr = 32;  /* Default for ORGHR */

            maxwrk = 2 * n + n * nb_gehrd;
            minwrk = 3 * n;

            /* Query SHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
            shseqr("S", jobvs, n, 0, n - 1, A, lda, wr, wi, VS, ldvs,
                   work, -1, &ieval);
            hswork = (int)work[0];

            if (!wantvs) {
                maxwrk = maxwrk > (n + hswork) ? maxwrk : (n + hswork);
            } else {
                maxwrk = maxwrk > (2 * n + (n - 1) * nb_orghr) ?
                         maxwrk : (2 * n + (n - 1) * nb_orghr);
                maxwrk = maxwrk > (n + hswork) ? maxwrk : (n + hswork);
            }
            lwrk = maxwrk;
            if (!wantsn)
                lwrk = lwrk > (n + (n * n) / 2) ? lwrk : (n + (n * n) / 2);
            if (wantsv || wantsb)
                liwrk = (n * n) / 4;
        }
        iwork[0] = liwrk;
        work[0] = (float)lwrk;

        if (lwork < minwrk && !lquery) {
            *info = -16;
        } else if (liwork < 1 && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("SGEESX", -(*info));
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
    bignum = ONE / smlnum;
    smlnum = sqrtf(smlnum) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = slange("M", n, n, A, lda, dum);
    scalea = 0;
    if (anrm > ZERO && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea)
        slascl("G", 0, 0, anrm, cscale, n, n, A, lda, &ierr);

    /* Permute the matrix to make it more nearly triangular (Workspace: need N) */
    ibal = 0;  /* 0-based */
    sgebal("P", n, A, lda, &ilo, &ihi, &work[ibal], &ierr);

    /* Reduce to upper Hessenberg form (Workspace: need 3*N, prefer 2*N+N*NB) */
    itau = n + ibal;
    iwrk = n + itau;
    sgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk], lwork - iwrk, &ierr);

    if (wantvs) {
        /* Copy Householder vectors to VS */
        slacpy("L", n, n, A, lda, VS, ldvs);

        /* Generate orthogonal matrix in VS (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB) */
        sorghr(n, ilo, ihi, VS, ldvs, &work[itau], &work[iwrk], lwork - iwrk, &ierr);
    }

    *sdim = 0;

    /* Perform QR iteration, accumulating Schur vectors in VS if desired */
    iwrk = itau;
    shseqr("S", jobvs, n, ilo, ihi, A, lda, wr, wi, VS, ldvs,
           &work[iwrk], lwork - iwrk, &ieval);
    if (ieval > 0)
        *info = ieval;

    /* Sort eigenvalues if desired */
    if (wantst && *info == 0) {
        if (scalea) {
            slascl("G", 0, 0, cscale, anrm, n, 1, wr, n, &ierr);
            slascl("G", 0, 0, cscale, anrm, n, 1, wi, n, &ierr);
        }
        for (i = 0; i < n; i++) {
            bwork[i] = select(&wr[i], &wi[i]);
        }

        /* Reorder eigenvalues, transform Schur vectors, and compute
         * reciprocal condition numbers */
        strsen(sense, jobvs, bwork, n, A, lda, VS, ldvs, wr, wi,
               sdim, rconde, rcondv, &work[iwrk], lwork - iwrk,
               iwork, liwork, &icond);
        if (!wantsn)
            maxwrk = maxwrk > (n + 2 * (*sdim) * (n - (*sdim))) ?
                     maxwrk : (n + 2 * (*sdim) * (n - (*sdim)));
        if (icond == -15) {
            /* Not enough real workspace */
            *info = -16;
        } else if (icond == -17) {
            /* Not enough integer workspace */
            *info = -18;
        } else if (icond > 0) {
            /* STRSEN failed to reorder or to restore standard Schur form */
            *info = icond + n;
        }
    }

    if (wantvs) {
        /* Undo balancing (Workspace: need N) */
        sgebak("P", "R", n, ilo, ihi, &work[ibal], n, VS, ldvs, &ierr);
    }

    if (scalea) {
        /* Undo scaling for the Schur form of A */
        slascl("H", 0, 0, cscale, anrm, n, n, A, lda, &ierr);
        /* Copy diagonal of A to WR */
        cblas_scopy(n, A, lda + 1, wr, 1);
        if ((wantsv || wantsb) && *info == 0) {
            dum[0] = *rcondv;
            slascl("G", 0, 0, cscale, anrm, 1, 1, dum, 1, &ierr);
            *rcondv = dum[0];
        }
        if (cscale == smlnum) {
            /*
             * If scaling back towards underflow, adjust WI if an
             * offdiagonal element of a 2-by-2 block in the Schur form
             * underflows.
             */
            if (ieval > 0) {
                i1 = ieval;  /* Already 0-based from shseqr */
                i2 = ihi - 2;
                slascl("G", 0, 0, cscale, anrm, ilo - 1, 1, wi,
                       (ilo - 1) > 1 ? (ilo - 1) : 1, &ierr);
            } else if (wantst) {
                i1 = 0;
                i2 = n - 2;
            } else {
                i1 = ilo - 1;  /* Convert to 0-based */
                i2 = ihi - 2;
            }
            inxt = i1 - 1;
            for (i = i1; i <= i2; i++) {
                if (i < inxt)
                    continue;
                if (wi[i] == ZERO) {
                    inxt = i + 1;
                } else {
                    if (A[(i + 1) + i * lda] == ZERO) {
                        wi[i] = ZERO;
                        wi[i + 1] = ZERO;
                    } else if (A[(i + 1) + i * lda] != ZERO && A[i + (i + 1) * lda] == ZERO) {
                        wi[i] = ZERO;
                        wi[i + 1] = ZERO;
                        if (i > 0)
                            cblas_sswap(i, &A[i * lda], 1, &A[(i + 1) * lda], 1);
                        if (n > i + 2)
                            cblas_sswap(n - i - 2, &A[i + (i + 2) * lda], lda,
                                        &A[(i + 1) + (i + 2) * lda], lda);
                        if (wantvs)
                            cblas_sswap(n, &VS[i * ldvs], 1, &VS[(i + 1) * ldvs], 1);
                        A[i + (i + 1) * lda] = A[(i + 1) + i * lda];
                        A[(i + 1) + i * lda] = ZERO;
                    }
                    inxt = i + 2;
                }
            }
        }
        slascl("G", 0, 0, cscale, anrm, n - ieval, 1, &wi[ieval],
               (n - ieval) > 1 ? (n - ieval) : 1, &ierr);
    }

    if (wantst && *info == 0) {
        /* Check if reordering successful */
        lastsl = 1;
        lst2sl = 1;
        *sdim = 0;
        ip = 0;
        for (i = 0; i < n; i++) {
            cursl = select(&wr[i], &wi[i]);
            if (wi[i] == ZERO) {
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
                        *sdim += 2;
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

    work[0] = (float)maxwrk;
    if (wantsv || wantsb) {
        iwork[0] = 1 > (*sdim) * (n - (*sdim)) ? 1 : (*sdim) * (n - (*sdim));
    } else {
        iwork[0] = 1;
    }
}
