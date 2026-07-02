/**
 * @file dgedmdq.c
 * @brief DGEDMDQ computes the Dynamic Mode Decomposition (DMD) for a pair of
 *        data snapshot matrices, using a QR factorization based compression.
 */

#include "semicolon_lapack_double.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/** @cond */
static inline INT imax2(INT a, INT b) { return (a > b) ? a : b; }
static inline INT imin2(INT a, INT b) { return (a < b) ? a : b; }
/** @endcond */

/**
 * DGEDMDQ computes the Dynamic Mode Decomposition (DMD) for a pair of data
 * snapshot matrices, using a QR factorization based compression of the data.
 * For the input matrix that contains a sequence of snapshots, DGEDMDQ computes
 * a certain number of Ritz pairs using the standard Rayleigh-Ritz extraction
 * from a subspace determined by the leading left singular vectors of the
 * projected snapshots.
 *
 * @param[in]     jobs    Determines whether the initial data snapshots are
 *                        scaled by a diagonal matrix ('S','C','Y','N').
 * @param[in]     jobz    Determines whether the eigenvectors (Koopman modes)
 *                        will be computed.
 *                        'V' :: computed and returned in Z.
 *                        'F' :: returned in factored form as Z*V.
 *                        'Q' :: returned in factored form as Q*Z.
 *                        'N' :: not computed.
 * @param[in]     jobr    Determines whether to compute the residuals
 *                        ('R','N').
 * @param[in]     jobq    Specifies whether to explicitly compute and return the
 *                        orthogonal matrix from the QR factorization.
 *                        'Q' :: Q computed and stored in F.
 *                        'N' :: Q not explicitly computed.
 * @param[in]     jobt    Specifies whether to return the upper triangular
 *                        factor from the QR factorization.
 *                        'R' :: R returned in Y.
 *                        'N' :: R not returned.
 * @param[in]     jobf    Specifies whether to store information needed for
 *                        post-processing ('R','E','N').
 * @param[in]     whtsvd  Selects the SVD algorithm from LAPACK, in {1,2,3,4}.
 * @param[in]     m       The state space dimension (rows of F). m >= 0.
 * @param[in]     n       The number of data snapshots (columns of F).
 *                        0 <= n <= m.
 * @param[in,out] F       Double precision M-by-N array. On entry the data
 *                        snapshots. On exit, if jobq=='Q' the orthogonal factor
 *                        of the QR factorization; otherwise the Householder
 *                        vectors as returned by dgeqrf.
 * @param[in]     ldf     The leading dimension of F. ldf >= m.
 * @param[out]    X       MIN(M,N)-by-(N-1) array, workspace to hold the leading
 *                        N-1 snapshots in the QR basis. On exit the leading K
 *                        columns contain the leading K left singular vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= n.
 * @param[out]    Y       MIN(M,N)-by-(N-1) array, workspace for the trailing
 *                        N-1 snapshots in the QR basis. On exit, if jobt=='R',
 *                        Y contains the upper triangular factor R.
 * @param[in]     ldy     The leading dimension of Y. ldy >= n.
 * @param[in]     nrnk    Determines how to compute the numerical rank
 *                        (-1, -2, or 0 < nrnk <= n-1).
 * @param[in]     tol     Tolerance for truncating small singular values,
 *                        0 <= tol < 1.
 * @param[out]    k       0 <= k <= n. The dimension of the SVD/POD basis and
 *                        the number of computed Ritz pairs.
 * @param[out]    reig    (N-1)-by-1 array. Real parts of the eigenvalues.
 * @param[out]    imeig   (N-1)-by-1 array. Imaginary parts of the eigenvalues.
 * @param[out]    Z       M-by-(N-1) array. Ritz vectors (see jobz).
 * @param[in]     ldz     The leading dimension of Z. ldz >= m.
 * @param[out]    res     (N-1)-by-1 array. Residuals for the K Ritz pairs.
 * @param[out]    B       MIN(M,N)-by-(N-1) array. Refinement / Exact DMD data
 *                        (see jobf).
 * @param[in]     ldb     The leading dimension of B. ldb >= min(m,n).
 * @param[out]    V       (N-1)-by-(N-1) array. On exit V(1:K,1:K) contains the
 *                        K eigenvectors of the Rayleigh quotient.
 * @param[in]     ldv     The leading dimension of V. ldv >= n-1.
 * @param[out]    S       (N-1)-by-(N-1) array. Used for the Rayleigh quotient,
 *                        overwritten during the eigenvalue decomposition.
 * @param[in]     lds     The leading dimension of S. lds >= n-1.
 * @param[out]    work    Workspace/output array. On exit work[0:MIN(M,N)]
 *                        contains the scalar factors from dgeqrf, and
 *                        work[MIN(M,N):MIN(M,N)+N-1] the singular values. On
 *                        workspace query, work[0] and work[1] hold the minimal
 *                        and optimal lengths.
 * @param[in]     lwork   The length of work. If lwork = -1, a workspace query.
 * @param[out]    iwork   Integer workspace, required only if whtsvd in {2,3,4}.
 * @param[in]     liwork  The length of iwork. If liwork = -1, a workspace query.
 * @param[out]    info    = -i < 0 : the i-th argument had an illegal value.
 *                        = 0 : successful return.
 *                        = 1 : void input (m=0 or n=0 or n=1), quick exit.
 *                        = 2 : the SVD computation of X did not converge.
 *                        = 3 : the computation of the eigenvalues did not
 *                              converge.
 *                        = 4 : data inconsistency warning (see dgedmd).
 */
void dgedmdq(const char* jobs, const char* jobz, const char* jobr,
             const char* jobq, const char* jobt, const char* jobf,
             const INT whtsvd, const INT m, const INT n, f64* restrict F,
             const INT ldf, f64* restrict X, const INT ldx, f64* restrict Y,
             const INT ldy, const INT nrnk, const f64 tol, INT* k,
             f64* restrict reig, f64* restrict imeig, f64* restrict Z,
             const INT ldz, f64* restrict res, f64* restrict B, const INT ldb,
             f64* restrict V, const INT ldv, f64* restrict S, const INT lds,
             f64* restrict work, const INT lwork, INT* restrict iwork,
             const INT liwork, INT* info)
{
    INT iminwr, info1, mlwdmd, mlwgqr, mlwmqr, mlwork, mlwqr, minmn,
        olwdmd, olwgqr, olwmqr, olwork, olwqr;
    INT lquery, sccolx, sccoly, wantq, wnttrf, wntres, wntvec, wntvcf,
        wntvcq, wntref, wntex;
    const char* jobvl;
    f64 rdummy[2];

    info1 = 0;
    iminwr = 0;
    olwork = 0;

    /* Test the input arguments */
    wntres = (jobr[0] == 'R' || jobr[0] == 'r');
    sccolx = (jobs[0] == 'S' || jobs[0] == 's' || jobs[0] == 'C' || jobs[0] == 'c');
    sccoly = (jobs[0] == 'Y' || jobs[0] == 'y');
    wntvec = (jobz[0] == 'V' || jobz[0] == 'v');
    wntvcf = (jobz[0] == 'F' || jobz[0] == 'f');
    wntvcq = (jobz[0] == 'Q' || jobz[0] == 'q');
    wntref = (jobf[0] == 'R' || jobf[0] == 'r');
    wntex  = (jobf[0] == 'E' || jobf[0] == 'e');
    wantq  = (jobq[0] == 'Q' || jobq[0] == 'q');
    wnttrf = (jobt[0] == 'R' || jobt[0] == 'r');
    minmn  = imin2(m, n);
    *info = 0;
    lquery = ((lwork == -1) || (liwork == -1));

    if (!(sccolx || sccoly || (jobs[0] == 'N' || jobs[0] == 'n'))) {
        *info = -1;
    } else if (!(wntvec || wntvcf || wntvcq || (jobz[0] == 'N' || jobz[0] == 'n'))) {
        *info = -2;
    } else if (!(wntres || (jobr[0] == 'N' || jobr[0] == 'n'))
               || (wntres && (jobz[0] == 'N' || jobz[0] == 'n'))) {
        *info = -3;
    } else if (!(wantq || (jobq[0] == 'N' || jobq[0] == 'n'))) {
        *info = -4;
    } else if (!(wnttrf || (jobt[0] == 'N' || jobt[0] == 'n'))) {
        *info = -5;
    } else if (!(wntref || wntex || (jobf[0] == 'N' || jobf[0] == 'n'))) {
        *info = -6;
    } else if (!((whtsvd == 1) || (whtsvd == 2) || (whtsvd == 3) || (whtsvd == 4))) {
        *info = -7;
    } else if (m < 0) {
        *info = -8;
    } else if ((n < 0) || (n > m + 1)) {
        *info = -9;
    } else if (ldf < m) {
        *info = -11;
    } else if (ldx < minmn) {
        *info = -13;
    } else if (ldy < minmn) {
        *info = -15;
    } else if (!((nrnk == -2) || (nrnk == -1) || ((nrnk >= 1) && (nrnk <= n)))) {
        *info = -16;
    } else if ((tol < ZERO) || (tol >= ONE)) {
        *info = -17;
    } else if (ldz < m) {
        *info = -22;
    } else if ((wntref || wntex) && (ldb < minmn)) {
        *info = -25;
    } else if (ldv < n - 1) {
        *info = -27;
    } else if (lds < n - 1) {
        *info = -29;
    }

    if (wntvec || wntvcf || wntvcq) {
        jobvl = "V";
    } else {
        jobvl = "N";
    }
    if (*info == 0) {
        /* Compute the minimal and the optimal workspace requirements. */
        if ((n == 0) || (n == 1)) {
            /* All output except K is void. */
            if (lquery) {
                iwork[0] = 1;
                work[0] = 2;
                work[1] = 2;
            } else {
                *k = 0;
            }
            *info = 1;
            return;
        }
        mlwqr = imax2(1, n);  /* Minimal workspace length for DGEQRF. */
        mlwork = minmn + mlwqr;
        if (lquery) {
            dgeqrf(m, n, F, ldf, work, rdummy, -1, &info1);
            olwqr = (INT)rdummy[0];
            olwork = imin2(m, n) + olwqr;
        }
        dgedmd(jobs, jobvl, jobr, jobf, whtsvd, minmn, n - 1, X, ldx, Y, ldy,
               nrnk, tol, k, reig, imeig, Z, ldz, res, B, ldb, V, ldv, S, lds,
               work, -1, iwork, liwork, &info1);
        mlwdmd = (INT)work[0];
        mlwork = imax2(mlwork, minmn + mlwdmd);
        iminwr = iwork[0];
        if (lquery) {
            olwdmd = (INT)work[1];
            olwork = imax2(olwork, minmn + olwdmd);
        }
        if (wntvec || wntvcf) {
            mlwmqr = imax2(1, n);
            mlwork = imax2(mlwork, minmn + n - 1 + mlwmqr);
            if (lquery) {
                dormqr("L", "N", m, n, minmn, F, ldf, NULL, Z, ldz, work, -1, &info1);
                olwmqr = (INT)work[0];
                olwork = imax2(olwork, minmn + n - 1 + olwmqr);
            }
        }
        if (wantq) {
            mlwgqr = n;
            mlwork = imax2(mlwork, minmn + n - 1 + mlwgqr);
            if (lquery) {
                dorgqr(m, minmn, minmn, F, ldf, NULL, work, -1, &info1);
                olwgqr = (INT)work[0];
                olwork = imax2(olwork, minmn + n - 1 + olwgqr);
            }
        }
        iminwr = imax2(1, iminwr);
        mlwork = imax2(2, mlwork);
        if (lwork < mlwork && (!lquery)) *info = -31;
        if (liwork < iminwr && (!lquery)) *info = -33;
    }
    if (*info != 0) {
        xerbla("DGEDMDQ", -(*info));
        return;
    } else if (lquery) {
        /* Return minimal and optimal workspace sizes */
        iwork[0] = iminwr;
        work[0] = (f64)mlwork;
        work[1] = (f64)olwork;
        return;
    }

    /* Initial QR factorization that is used to represent the snapshots as
     * elements of lower dimensional subspace. */
    dgeqrf(m, n, F, ldf, work, &work[minmn], lwork - minmn, &info1);

    /* Define X and Y as the snapshots representations in the orthogonal basis
     * computed in the QR factorization. */
    dlaset("L", minmn, n - 1, ZERO, ZERO, X, ldx);
    dlacpy("U", minmn, n - 1, F, ldf, X, ldx);
    dlacpy("A", minmn, n - 1, &F[ldf], ldf, Y, ldy);
    if (m >= 3) {
        dlaset("L", minmn - 2, n - 2, ZERO, ZERO, &Y[2], ldy);
    }

    /* Compute the DMD of the projected snapshot pairs (X,Y) */
    dgedmd(jobs, jobvl, jobr, jobf, whtsvd, minmn, n - 1, X, ldx, Y, ldy, nrnk,
           tol, k, reig, imeig, Z, ldz, res, B, ldb, V, ldv, S, lds,
           &work[minmn], lwork - minmn, iwork, liwork, &info1);
    if (info1 == 2 || info1 == 3) {
        /* Return with error code. See DGEDMD for details. */
        *info = info1;
        return;
    } else {
        *info = info1;
    }

    /* The Ritz vectors (Koopman modes) can be explicitly formed or returned in
     * factored form. */
    if (wntvec) {
        /* Compute the eigenvectors explicitly. */
        if (m > minmn)
            dlaset("A", m - minmn, *k, ZERO, ZERO, &Z[minmn], ldz);
        dormqr("L", "N", m, *k, minmn, F, ldf, work, Z, ldz,
               &work[minmn + n - 1], lwork - (minmn + n - 1), &info1);
    } else if (wntvcf) {
        /* Return the Ritz vectors (eigenvectors) in factored form Z*V. */
        dlacpy("A", n, *k, X, ldx, Z, ldz);
        if (m > n)
            dlaset("A", m - n, *k, ZERO, ZERO, &Z[n], ldz);
        dormqr("L", "N", m, *k, minmn, F, ldf, work, Z, ldz,
               &work[minmn + n - 1], lwork - (minmn + n - 1), &info1);
    }

    /* The upper triangular factor R in the initial QR factorization is
     * optionally returned in the array Y. */
    if (wnttrf) {
        dlaset("A", minmn, n, ZERO, ZERO, Y, ldy);
        dlacpy("U", minmn, n, F, ldf, Y, ldy);
    }

    /* The orthonormal/orthogonal factor Q in the initial QR factorization is
     * optionally returned in the array F. */
    if (wantq) {
        dorgqr(m, minmn, minmn, F, ldf, work, &work[minmn + n - 1],
               lwork - (minmn + n - 1), &info1);
    }
}
