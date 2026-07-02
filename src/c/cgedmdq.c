/**
 * @file cgedmdq.c
 * @brief CGEDMDQ computes the Dynamic Mode Decomposition (DMD) for a pair of
 *        data snapshot matrices, using a QR factorization based compression.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const c64 CZERO = CMPLXF(0.0f, 0.0f);

/** @cond */
static inline INT imax2(INT a, INT b) { return (a > b) ? a : b; }
static inline INT imin2(INT a, INT b) { return (a < b) ? a : b; }
/** @endcond */

/**
 * CGEDMDQ computes the Dynamic Mode Decomposition (DMD) for a pair of data
 * snapshot matrices, using a QR factorization based compression of the data.
 * For the input matrix that contains a sequence of snapshots, CGEDMDQ computes
 * a certain number of Ritz pairs using the standard Rayleigh-Ritz extraction
 * from a subspace determined by the leading left singular vectors of the
 * projected snapshots.
 *
 * @param[in]     jobs    Determines whether the initial data snapshots are
 *                        scaled by a diagonal matrix ('S','C','Y','N').
 * @param[in]     jobz    Determines whether the eigenvectors (Koopman modes)
 *                        will be computed ('V','F','Q','N').
 * @param[in]     jobr    Determines whether to compute the residuals
 *                        ('R','N').
 * @param[in]     jobq    Specifies whether to explicitly compute and return the
 *                        unitary matrix from the QR factorization ('Q','N').
 * @param[in]     jobt    Specifies whether to return the upper triangular
 *                        factor from the QR factorization ('R','N').
 * @param[in]     jobf    Specifies whether to store information needed for
 *                        post-processing ('R','E','N').
 * @param[in]     whtsvd  Selects the SVD algorithm from LAPACK, in {1,2,3,4}.
 * @param[in]     m       The state space dimension (rows of F). m >= 0.
 * @param[in]     n       The number of data snapshots (columns of F).
 *                        0 <= n <= m.
 * @param[in,out] F       Complex M-by-N array. On entry the data snapshots. On
 *                        exit, if jobq=='Q' the unitary factor of the QR
 *                        factorization; otherwise the Householder vectors as
 *                        returned by cgeqrf.
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
 * @param[out]    eigs    (N-1)-by-1 array. The computed eigenvalues.
 * @param[out]    Z       M-by-(N-1) array. Ritz vectors (see jobz).
 * @param[in]     ldz     The leading dimension of Z. ldz >= m.
 * @param[out]    res     (N-1)-by-1 real array. Residuals for the K Ritz pairs.
 * @param[out]    B       MIN(M,N)-by-(N-1) array. Refinement / Exact DMD data
 *                        (see jobf).
 * @param[in]     ldb     The leading dimension of B. ldb >= min(m,n).
 * @param[out]    V       (N-1)-by-(N-1) array. On exit V(1:K,1:K) contains the
 *                        K eigenvectors of the Rayleigh quotient.
 * @param[in]     ldv     The leading dimension of V. ldv >= n-1.
 * @param[out]    S       (N-1)-by-(N-1) array. Used for the Rayleigh quotient,
 *                        overwritten during the eigenvalue decomposition.
 * @param[in]     lds     The leading dimension of S. lds >= n-1.
 * @param[out]    zwork   Complex workspace/output. On workspace query, zwork[0]
 *                        and zwork[1] hold the minimal and optimal lengths.
 * @param[in]     lzwork  The length of zwork. If lzwork = -1, a workspace query.
 * @param[out]    work    Real workspace/output.
 * @param[in]     lwork   The length of work. If lwork = -1, a workspace query.
 * @param[out]    iwork   Integer workspace, required only if whtsvd in {2,3,4}.
 * @param[in]     liwork  The length of iwork. If liwork = -1, a workspace query.
 * @param[out]    info    = -i < 0 : the i-th argument had an illegal value.
 *                        = 0 : successful return.
 *                        = 1 : void input (m=0 or n=0 or n=1), quick exit.
 *                        = 2 : the SVD computation of X did not converge.
 *                        = 3 : the computation of the eigenvalues did not
 *                              converge.
 *                        = 4 : data inconsistency warning (see cgedmd).
 */
void cgedmdq(const char* jobs, const char* jobz, const char* jobr,
             const char* jobq, const char* jobt, const char* jobf,
             const INT whtsvd, const INT m, const INT n, c64* restrict F,
             const INT ldf, c64* restrict X, const INT ldx, c64* restrict Y,
             const INT ldy, const INT nrnk, const f32 tol, INT* k,
             c64* restrict eigs, c64* restrict Z, const INT ldz,
             f32* restrict res, c64* restrict B, const INT ldb,
             c64* restrict V, const INT ldv, c64* restrict S, const INT lds,
             c64* restrict zwork, const INT lzwork, f32* restrict work,
             const INT lwork, INT* restrict iwork, const INT liwork, INT* info)
{
    INT iminwr, info1, minmn, mlrwrk, mlwdmd, mlwgqr, mlwmqr, mlwork, mlwqr,
        olwdmd, olwgqr, olwmqr, olwork, olwqr;
    INT lquery, sccolx, sccoly, wantq, wnttrf, wntres, wntvec, wntvcf,
        wntvcq, wntref, wntex;
    const char* jobvl;

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
    lquery = ((lzwork == -1) || (lwork == -1) || (liwork == -1));

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
        *info = -21;
    } else if ((wntref || wntex) && (ldb < minmn)) {
        *info = -24;
    } else if (ldv < n - 1) {
        *info = -26;
    } else if (lds < n - 1) {
        *info = -28;
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
                zwork[0] = 2;
                zwork[1] = 2;
                work[0] = 2;
                work[1] = 2;
            } else {
                *k = 0;
            }
            *info = 1;
            return;
        }

        mlrwrk = 2;
        mlwork = 2;
        olwork = 2;
        iminwr = 1;
        mlwqr = imax2(1, n);  /* Minimal workspace length for CGEQRF. */
        mlwork = imax2(mlwork, minmn + mlwqr);

        if (lquery) {
            cgeqrf(m, n, F, ldf, NULL, zwork, -1, &info1);
            olwqr = (INT)crealf(zwork[0]);
            olwork = imax2(olwork, minmn + olwqr);
        }
        cgedmd(jobs, jobvl, jobr, jobf, whtsvd, minmn, n - 1, X, ldx, Y, ldy,
               nrnk, tol, k, eigs, Z, ldz, res, B, ldb, V, ldv, S, lds,
               zwork, -1, work, -1, iwork, -1, &info1);
        mlwdmd = (INT)crealf(zwork[0]);
        mlwork = imax2(mlwork, minmn + mlwdmd);
        mlrwrk = imax2(mlrwrk, (INT)work[0]);
        iminwr = imax2(iminwr, iwork[0]);
        if (lquery) {
            olwdmd = (INT)crealf(zwork[1]);
            olwork = imax2(olwork, minmn + olwdmd);
        }
        if (wntvec || wntvcf) {
            mlwmqr = imax2(1, n);
            mlwork = imax2(mlwork, minmn + mlwmqr);
            if (lquery) {
                cunmqr("L", "N", m, n, minmn, F, ldf, NULL, Z, ldz, zwork, -1, &info1);
                olwmqr = (INT)crealf(zwork[0]);
                olwork = imax2(olwork, minmn + olwmqr);
            }
        }
        if (wantq) {
            mlwgqr = imax2(1, n);
            mlwork = imax2(mlwork, minmn + mlwgqr);
            if (lquery) {
                cungqr(m, minmn, minmn, F, ldf, NULL, zwork, -1, &info1);
                olwgqr = (INT)crealf(zwork[0]);
                olwork = imax2(olwork, minmn + olwgqr);
            }
        }
        if (liwork < iminwr && (!lquery)) *info = -34;
        if (lwork < mlrwrk && (!lquery)) *info = -32;
        if (lzwork < mlwork && (!lquery)) *info = -30;
    }
    if (*info != 0) {
        xerbla("CGEDMDQ", -(*info));
        return;
    } else if (lquery) {
        /* Return minimal and optimal workspace sizes */
        iwork[0] = iminwr;
        zwork[0] = (c64)mlwork;
        zwork[1] = (c64)olwork;
        work[0] = (f32)mlrwrk;
        work[1] = (f32)mlrwrk;
        return;
    }

    /* Initial QR factorization that is used to represent the snapshots as
     * elements of lower dimensional subspace. */
    cgeqrf(m, n, F, ldf, zwork, &zwork[minmn], lzwork - minmn, &info1);

    /* Define X and Y as the snapshots representations in the orthogonal basis
     * computed in the QR factorization. */
    claset("L", minmn, n - 1, CZERO, CZERO, X, ldx);
    clacpy("U", minmn, n - 1, F, ldf, X, ldx);
    clacpy("A", minmn, n - 1, &F[ldf], ldf, Y, ldy);
    if (m >= 3) {
        claset("L", minmn - 2, n - 2, CZERO, CZERO, &Y[2], ldy);
    }

    /* Compute the DMD of the projected snapshot pairs (X,Y) */
    cgedmd(jobs, jobvl, jobr, jobf, whtsvd, minmn, n - 1, X, ldx, Y, ldy, nrnk,
           tol, k, eigs, Z, ldz, res, B, ldb, V, ldv, S, lds, &zwork[minmn],
           lzwork - minmn, work, lwork, iwork, liwork, &info1);
    if (info1 == 2 || info1 == 3) {
        /* Return with error code. See CGEDMD for details. */
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
            claset("A", m - minmn, *k, CZERO, CZERO, &Z[minmn], ldz);
        cunmqr("L", "N", m, *k, minmn, F, ldf, zwork, Z, ldz,
               &zwork[minmn], lzwork - minmn, &info1);
    } else if (wntvcf) {
        /* Return the Ritz vectors (eigenvectors) in factored form Z*V. */
        clacpy("A", n, *k, X, ldx, Z, ldz);
        if (m > n)
            claset("A", m - n, *k, CZERO, CZERO, &Z[n], ldz);
        cunmqr("L", "N", m, *k, minmn, F, ldf, zwork, Z, ldz,
               &zwork[minmn], lzwork - minmn, &info1);
    }

    /* The upper triangular factor R in the initial QR factorization is
     * optionally returned in the array Y. */
    if (wnttrf) {
        claset("A", minmn, n, CZERO, CZERO, Y, ldy);
        clacpy("U", minmn, n, F, ldf, Y, ldy);
    }

    /* The orthonormal/unitary factor Q in the initial QR factorization is
     * optionally returned in the array F. */
    if (wantq) {
        cungqr(m, minmn, minmn, F, ldf, zwork, &zwork[minmn], lzwork - minmn, &info1);
    }
}
