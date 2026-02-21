/**
 * @file zgeesx.c
 * @brief ZGEESX computes Schur form with optional eigenvalue ordering and condition numbers.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZGEESX computes for an N-by-N complex nonsymmetric matrix A, the
 * eigenvalues, the Schur form T, and, optionally, the matrix of Schur
 * vectors Z.  This gives the Schur factorization A = Z*T*(Z**H).
 *
 * Optionally, it also orders the eigenvalues on the diagonal of the
 * Schur form so that selected eigenvalues are at the top left;
 * computes a reciprocal condition number for the average of the
 * selected eigenvalues (RCONDE); and computes a reciprocal condition
 * number for the right invariant subspace corresponding to the
 * selected eigenvalues (RCONDV).  The leading columns of Z form an
 * orthonormal basis for this invariant subspace.
 *
 * For further explanation of the reciprocal condition numbers RCONDE
 * and RCONDV, see Section 4.10 of the LAPACK Users' Guide (where
 * these quantities are called s and sep respectively).
 *
 * A complex matrix is in Schur form if it is upper triangular.
 *
 * @param[in] jobvs  = 'N': Schur vectors are not computed;
 *                   = 'V': Schur vectors are computed.
 * @param[in] sort   Specifies whether or not to order the eigenvalues:
 *                   = 'N': Eigenvalues are not ordered;
 *                   = 'S': Eigenvalues are ordered (see select).
 * @param[in] select Eigenvalue selection callback. If sort = 'S', select is used
 *                   to select eigenvalues to sort to the top left of the Schur form.
 *                   If sort = 'N', select is not referenced and may be NULL.
 *                   An eigenvalue W(j) is selected if select(W(j)) is true.
 * @param[in] sense  Determines which reciprocal condition numbers are computed:
 *                   = 'N': None are computed;
 *                   = 'E': Computed for average of selected eigenvalues only;
 *                   = 'V': Computed for selected right invariant subspace only;
 *                   = 'B': Computed for both.
 *                   If sense = 'E', 'V' or 'B', sort must equal 'S'.
 * @param[in] n      The order of the matrix A. n >= 0.
 * @param[in,out] A  On entry, the N-by-N matrix A.
 *                   On exit, A has been overwritten by its Schur form T.
 *                   Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, n).
 * @param[out] sdim  If sort = 'N', sdim = 0.
 *                   If sort = 'S', sdim = number of eigenvalues for which
 *                   select is true.
 * @param[out] W     Complex array, dimension (n). Contains the computed
 *                   eigenvalues, in the same order that they appear on the
 *                   diagonal of the output Schur form T.
 * @param[out] VS    If jobvs = 'V', VS contains the unitary matrix Z of Schur
 *                   vectors. Dimension (ldvs, n).
 *                   If jobvs = 'N', VS is not referenced.
 * @param[in] ldvs   The leading dimension of VS. ldvs >= 1;
 *                   if jobvs = 'V', ldvs >= n.
 * @param[out] rconde If sense = 'E' or 'B', contains the reciprocal condition number
 *                    for the average of the selected eigenvalues.
 * @param[out] rcondv If sense = 'V' or 'B', contains the reciprocal condition number
 *                    for the selected right invariant subspace.
 * @param[out] work  Complex workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1, 2*n).
 *                   Also, if sense = 'E' or 'V' or 'B', lwork >= 2*sdim*(n-sdim),
 *                   where sdim is the number of selected eigenvalues computed by
 *                   this routine. Note that 2*sdim*(n-sdim) <= n*n/2.
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] rwork Double precision array, dimension (n).
 * @param[out] bwork Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 *                   - > 0: if info = i, and i is
 *                   - <= n: the QR algorithm failed to compute all eigenvalues;
 *                     elements 0:ilo-1 and i:n-1 of W contain those
 *                     eigenvalues which have converged; if jobvs = 'V', VS
 *                     contains the transformation which reduces A to its
 *                     partially converged Schur form.
 *                   - = n+1: eigenvalues could not be reordered because some
 *                     eigenvalues were too close to separate (the problem
 *                     is very ill-conditioned);
 *                   - = n+2: after reordering, roundoff changed values of some
 *                     complex eigenvalues so that leading eigenvalues in
 *                     the Schur form no longer satisfy select=true.
 */
void zgeesx(const char* jobvs, const char* sort, zselect1_t select,
            const char* sense, const int n, c128* A, const int lda,
            int* sdim, c128* W,
            c128* VS, const int ldvs,
            f64* rconde, f64* rcondv,
            c128* work, const int lwork,
            f64* rwork, int* bwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int lquery, scalea, wantsb, wantse, wantsn, wantst, wantsv, wantvs;
    int hswork, i, ibal, icond, ierr, ieval;
    int ihi, ilo, itau, iwrk, lwrk, maxwrk, minwrk;
    f64 anrm, bignum, cscale = ONE, eps, smlnum;
    f64 dum[1];
    int nb_gehrd, nb_unghr;

    *info = 0;
    wantvs = (jobvs[0] == 'V' || jobvs[0] == 'v');
    wantst = (sort[0] == 'S' || sort[0] == 's');
    wantsn = (sense[0] == 'N' || sense[0] == 'n');
    wantse = (sense[0] == 'E' || sense[0] == 'e');
    wantsv = (sense[0] == 'V' || sense[0] == 'v');
    wantsb = (sense[0] == 'B' || sense[0] == 'b');
    lquery = (lwork == -1);

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
        *info = -11;
    }

    /* Compute workspace */
    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            lwrk = 1;
        } else {
            nb_gehrd = lapack_get_nb("GEHRD");
            nb_unghr = lapack_get_nb("ORGHR");
            if (nb_unghr == 1) nb_unghr = 32;

            maxwrk = n + n * nb_gehrd;
            minwrk = 2 * n;

            /* Query ZHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
            zhseqr("S", jobvs, n, 0, n - 1, A, lda, W, VS, ldvs,
                   work, -1, &ieval);
            hswork = (int)creal(work[0]);

            if (!wantvs) {
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
            } else {
                maxwrk = maxwrk > (n + (n - 1) * nb_unghr) ?
                         maxwrk : (n + (n - 1) * nb_unghr);
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
            }
            lwrk = maxwrk;
            if (!wantsn)
                lwrk = lwrk > ((n * n) / 2) ? lwrk : ((n * n) / 2);
        }
        work[0] = CMPLX((f64)lwrk, 0.0);

        if (lwork < minwrk && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("ZGEESX", -(*info));
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
    smlnum = dlamch("S");
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = zlange("M", n, n, A, lda, dum);
    scalea = 0;
    if (anrm > ZERO && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea)
        zlascl("G", 0, 0, anrm, cscale, n, n, A, lda, &ierr);

    /* Permute the matrix to make it more nearly triangular */
    ibal = 0;
    zgebal("P", n, A, lda, &ilo, &ihi, &rwork[ibal], &ierr);

    /* Reduce to upper Hessenberg form */
    itau = 0;
    iwrk = n + itau;
    zgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk],
           lwork - iwrk, &ierr);

    if (wantvs) {
        /* Copy Householder vectors to VS */
        zlacpy("L", n, n, A, lda, VS, ldvs);

        /* Generate unitary matrix in VS */
        zunghr(n, ilo, ihi, VS, ldvs, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);
    }

    *sdim = 0;

    /* Perform QR iteration, accumulating Schur vectors in VS if desired */
    iwrk = itau;
    zhseqr("S", jobvs, n, ilo, ihi, A, lda, W, VS, ldvs,
           &work[iwrk], lwork - iwrk, &ieval);
    if (ieval > 0)
        *info = ieval;

    /* Sort eigenvalues if desired */
    if (wantst && *info == 0) {
        if (scalea)
            zlascl("G", 0, 0, cscale, anrm, n, 1, W, n, &ierr);
        for (i = 0; i < n; i++) {
            bwork[i] = select(&W[i]);
        }

        /* Reorder eigenvalues, transform Schur vectors, and compute
         * reciprocal condition numbers */
        ztrsen(sense, jobvs, bwork, n, A, lda, VS, ldvs, W,
               sdim, rconde, rcondv, &work[iwrk], lwork - iwrk,
               &icond);
        if (!wantsn)
            maxwrk = maxwrk > (2 * (*sdim) * (n - (*sdim))) ?
                     maxwrk : (2 * (*sdim) * (n - (*sdim)));
        if (icond == -14) {
            /* Not enough complex workspace */
            *info = -15;
        }
    }

    if (wantvs) {
        /* Undo balancing */
        zgebak("P", "R", n, ilo, ihi, &rwork[ibal], n, VS,
               ldvs, &ierr);
    }

    if (scalea) {
        /* Undo scaling for the Schur form of A */
        zlascl("U", 0, 0, cscale, anrm, n, n, A, lda, &ierr);
        cblas_zcopy(n, A, lda + 1, W, 1);
        if ((wantsv || wantsb) && *info == 0) {
            dum[0] = *rcondv;
            dlascl("G", 0, 0, cscale, anrm, 1, 1, dum, 1, &ierr);
            *rcondv = dum[0];
        }
    }

    work[0] = CMPLX((f64)maxwrk, 0.0);
}
