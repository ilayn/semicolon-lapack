/**
 * @file cgees.c
 * @brief CGEES computes the eigenvalues, Schur form, and optionally Schur vectors.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CGEES computes for an N-by-N complex nonsymmetric matrix A, the
 * eigenvalues, the Schur form T, and, optionally, the matrix of Schur
 * vectors Z.  This gives the Schur factorization A = Z*T*(Z**H).
 *
 * Optionally, it also orders the eigenvalues on the diagonal of the
 * Schur form so that selected eigenvalues are at the top left.
 * The leading columns of Z then form an orthonormal basis for the
 * invariant subspace corresponding to the selected eigenvalues.
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
 *                   The callback should return nonzero if the eigenvalue
 *                   W(j) should be selected.
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
 * @param[out] work  Complex workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns optimal lwork.
 * @param[in] lwork  The dimension of work. lwork >= max(1, 2*n).
 *                   If lwork = -1, a workspace query is assumed.
 * @param[out] rwork Single precision array, dimension (n).
 * @param[out] bwork Integer array, dimension (n). Not referenced if sort = 'N'.
 * @param[out] info
 *                   - = 0: successful exit
 *                   - < 0: if info = -i, the i-th argument had an illegal value
 *                   - > 0: if info = i, and i is
 *                   - <= n: the QR algorithm failed to compute all eigenvalues;
 *                     elements 0:ilo-1 and i:n-1 of W contain those
 *                     eigenvalues which have converged;
 *                   - = n+1: eigenvalues could not be reordered because some
 *                     eigenvalues were too close to separate;
 *                   - = n+2: after reordering, roundoff changed values of some
 *                     complex eigenvalues so that leading eigenvalues in
 *                     the Schur form no longer satisfy select=true.
 */
void cgees(const char* jobvs, const char* sort, cselect1_t select,
           const int n, c64* A, const int lda, int* sdim,
           c64* W,
           c64* VS, const int ldvs,
           c64* work, const int lwork,
           f32* rwork, int* bwork, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int lquery, scalea, wantst, wantvs;
    int hswork, i, ibal, icond, ierr, ieval;
    int ihi, ilo, itau, iwrk, maxwrk, minwrk;
    f32 anrm, bignum, cscale = ONE, eps, s, sep, smlnum;
    f32 dum[1];
    int nb_gehrd, nb_unghr;

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    wantvs = (jobvs[0] == 'V' || jobvs[0] == 'v');
    wantst = (sort[0] == 'S' || sort[0] == 's');

    if (!wantvs && !(jobvs[0] == 'N' || jobvs[0] == 'n')) {
        *info = -1;
    } else if (!wantst && !(sort[0] == 'N' || sort[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldvs < 1 || (wantvs && ldvs < n)) {
        *info = -10;
    }

    /* Compute workspace */
    if (*info == 0) {
        if (n == 0) {
            minwrk = 1;
            maxwrk = 1;
        } else {
            nb_gehrd = lapack_get_nb("GEHRD");
            nb_unghr = lapack_get_nb("ORGHR");
            if (nb_unghr == 1) nb_unghr = 32;

            maxwrk = n + n * nb_gehrd;
            minwrk = 2 * n;

            /* Query CHSEQR for workspace (0-based: ilo=0, ihi=n-1) */
            chseqr("S", jobvs, n, 0, n - 1, A, lda, W, VS, ldvs,
                   work, -1, &ieval);
            hswork = (int)crealf(work[0]);

            if (!wantvs) {
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
            } else {
                maxwrk = maxwrk > (n + (n - 1) * nb_unghr) ?
                         maxwrk : (n + (n - 1) * nb_unghr);
                maxwrk = maxwrk > hswork ? maxwrk : hswork;
            }
        }
        work[0] = CMPLXF((f32)maxwrk, 0.0f);

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("CGEES", -(*info));
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
    anrm = clange("M", n, n, A, lda, dum);
    scalea = 0;
    if (anrm > ZERO && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea)
        clascl("G", 0, 0, anrm, cscale, n, n, A, lda, &ierr);

    /* Permute the matrix to make it more nearly triangular */
    ibal = 0;
    cgebal("P", n, A, lda, &ilo, &ihi, &rwork[ibal], &ierr);

    /* Reduce to upper Hessenberg form */
    itau = 0;
    iwrk = n + itau;
    cgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk],
           lwork - iwrk, &ierr);

    if (wantvs) {
        /* Copy Householder vectors to VS */
        clacpy("L", n, n, A, lda, VS, ldvs);

        /* Generate unitary matrix in VS */
        cunghr(n, ilo, ihi, VS, ldvs, &work[itau],
               &work[iwrk], lwork - iwrk, &ierr);
    }

    *sdim = 0;

    /* Perform QR iteration, accumulating Schur vectors in VS if desired */
    iwrk = itau;
    chseqr("S", jobvs, n, ilo, ihi, A, lda, W, VS, ldvs,
           &work[iwrk], lwork - iwrk, &ieval);
    if (ieval > 0)
        *info = ieval;

    /* Sort eigenvalues if desired */
    if (wantst && *info == 0) {
        if (scalea)
            clascl("G", 0, 0, cscale, anrm, n, 1, W, n, &ierr);
        for (i = 0; i < n; i++) {
            bwork[i] = select(&W[i]);
        }

        /* Reorder eigenvalues and transform Schur vectors */
        ctrsen("N", jobvs, bwork, n, A, lda, VS, ldvs, W,
               sdim, &s, &sep, &work[iwrk], lwork - iwrk, &icond);
        if (icond > 0)
            *info = n + icond;
    }

    if (wantvs) {
        /* Undo balancing */
        cgebak("P", "R", n, ilo, ihi, &rwork[ibal], n, VS,
               ldvs, &ierr);
    }

    if (scalea) {
        /* Undo scaling for the Schur form of A */
        clascl("U", 0, 0, cscale, anrm, n, n, A, lda, &ierr);
        cblas_ccopy(n, A, lda + 1, W, 1);
    }

    work[0] = CMPLXF((f32)maxwrk, 0.0f);
}
