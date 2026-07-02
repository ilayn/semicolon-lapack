/**
 * @file cgedmd.c
 * @brief CGEDMD computes the Dynamic Mode Decomposition (DMD) for a pair of
 *        data snapshot matrices.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const c64 CZERO = CMPLXF(0.0f, 0.0f);
static const c64 CONE = CMPLXF(1.0f, 0.0f);

/** @cond */
static inline INT imax2(INT a, INT b) { return (a > b) ? a : b; }
static inline INT imin2(INT a, INT b) { return (a < b) ? a : b; }
/** @endcond */

/**
 * CGEDMD computes the Dynamic Mode Decomposition (DMD) for a pair of data
 * snapshot matrices. For the input matrices X and Y such that Y = A*X with an
 * unaccessible matrix A, CGEDMD computes a certain number of Ritz pairs of A
 * using the standard Rayleigh-Ritz extraction from a subspace of range(X) that
 * is determined using the leading left singular vectors of X. Optionally,
 * CGEDMD returns the residuals of the computed Ritz pairs, the information
 * needed for a refinement of the Ritz vectors, or the eigenvectors of the
 * Exact DMD.
 *
 * @param[in]     jobs    Determines whether the initial data snapshots are
 *                        scaled by a diagonal matrix ('S','C','Y','N').
 * @param[in]     jobz    Determines whether the eigenvectors (Koopman modes)
 *                        will be computed ('V','F','N').
 * @param[in]     jobr    Determines whether to compute the residuals
 *                        ('R','N').
 * @param[in]     jobf    Specifies whether to store information needed for
 *                        post-processing ('R','E','N').
 * @param[in]     whtsvd  Selects the SVD algorithm from LAPACK, in {1,2,3,4}.
 *                        1 :: cgesvd, 2 :: cgesdd, 3 :: cgesvdq, 4 :: cgejsv.
 * @param[in]     m       The state space dimension (rows of X, Y). m >= 0.
 * @param[in]     n       The number of data snapshot pairs. 0 <= n <= m.
 * @param[in,out] X       Complex M-by-N array. On entry the data snapshot
 *                        matrix X. On exit, the leading K columns contain a POD
 *                        basis.
 * @param[in]     ldx     The leading dimension of X. ldx >= m.
 * @param[in,out] Y       Complex M-by-N array. On entry the data snapshot
 *                        matrix Y. On exit, if jobr=='R' the leading K columns
 *                        contain the residual vectors; otherwise Y contains the
 *                        original input data scaled per jobs.
 * @param[in]     ldy     The leading dimension of Y. ldy >= m.
 * @param[in]     nrnk    Determines how to compute the numerical rank
 *                        (-1, -2, or 0 < nrnk <= n).
 * @param[in]     tol     Tolerance for truncating small singular values,
 *                        0 <= tol < 1.
 * @param[out]    k       0 <= k <= n. The dimension of the POD basis and the
 *                        number of computed Ritz pairs.
 * @param[out]    eigs    N-by-1 array. eigs[0:K] contain the computed
 *                        eigenvalues LAMBDA(i).
 * @param[out]    Z       M-by-N array. If jobz=='V', the Ritz vectors; if
 *                        jobz=='F', the descriptions hold for X(:,1:K)*W(1:K,1:K).
 * @param[in]     ldz     The leading dimension of Z. ldz >= m.
 * @param[out]    res     N-by-1 real array. res[0:K] contain the residuals.
 * @param[out]    B       M-by-N array. Refinement / Exact DMD data (see jobf).
 * @param[in]     ldb     The leading dimension of B. ldb >= m.
 * @param[out]    W       N-by-N array. On exit W(1:K,1:K) contains the K
 *                        eigenvectors of the Rayleigh quotient. Also used as
 *                        workspace for the right singular vectors of X.
 * @param[in]     ldw     The leading dimension of W. ldw >= n.
 * @param[out]    S       N-by-N array. Used for the Rayleigh quotient,
 *                        overwritten during the eigenvalue decomposition.
 * @param[in]     lds     The leading dimension of S. lds >= n.
 * @param[out]    zwork   Complex workspace/output. On workspace query, zwork[0]
 *                        is the minimal and zwork[1] the optimal length.
 * @param[in]     lzwork  The length of zwork. If lzwork = -1, a workspace query.
 * @param[out]    rwork   Real workspace/output. On exit rwork[0:N] contains the
 *                        singular values of (scaled) X. If whtsvd==4, rwork[N]
 *                        and rwork[N+1] contain a scaling factor.
 * @param[in]     lrwork  The length of rwork. If lrwork = -1, a workspace query.
 * @param[out]    iwork   Integer workspace, required only if whtsvd in {2,3,4}.
 * @param[in]     liwork  The length of iwork. If liwork = -1, a workspace query.
 * @param[out]    info    = -i < 0 : the i-th argument had an illegal value.
 *                        = 0 : successful return.
 *                        = 1 : void input (m=0 or n=0), quick exit.
 *                        = 2 : the SVD computation of X did not converge.
 *                        = 3 : the computation of the eigenvalues did not
 *                              converge.
 *                        = 4 : data inconsistency detected and (for jobs=='C')
 *                              a column of Y was zeroed; warning flag.
 */
void cgedmd(const char* jobs, const char* jobz, const char* jobr,
            const char* jobf, const INT whtsvd, const INT m, const INT n,
            c64* restrict X, const INT ldx, c64* restrict Y, const INT ldy,
            const INT nrnk, const f32 tol, INT* k, c64* restrict eigs,
            c64* restrict Z, const INT ldz, f32* restrict res,
            c64* restrict B, const INT ldb, c64* restrict W, const INT ldw,
            c64* restrict S, const INT lds, c64* restrict zwork,
            const INT lzwork, f32* restrict rwork, const INT lrwork,
            INT* restrict iwork, const INT liwork, INT* info)
{
    f32 ofl, rootsc, scale, small, ssum, xscl1, xscl2, tbig;
    INT i, j, iminwr, info1, info2, lwrkev, lwrsdd, lwrsvd, lwrsvj, lwrsvq,
        mlwork, mwrkev, mwrsdd, mwrsvd, mwrsvj, mwrsvq, numrnk, olwork, mlrwrk;
    INT badxy, lquery, sccolx, sccoly, wntex, wntref, wntres, wntvec;
    char t_or_n;
    const char* jobzl;
    const char* jsvopt;
    f32 rdummy[2];

    xscl1 = ZERO;
    xscl2 = ZERO;
    info1 = 0;
    info2 = 0;
    jsvopt = "J";

    /* Test the input arguments */
    wntres = (jobr[0] == 'R' || jobr[0] == 'r');
    sccolx = (jobs[0] == 'S' || jobs[0] == 's' || jobs[0] == 'C' || jobs[0] == 'c');
    sccoly = (jobs[0] == 'Y' || jobs[0] == 'y');
    wntvec = (jobz[0] == 'V' || jobz[0] == 'v');
    wntref = (jobf[0] == 'R' || jobf[0] == 'r');
    wntex  = (jobf[0] == 'E' || jobf[0] == 'e');
    *info  = 0;
    lquery = ((lzwork == -1) || (liwork == -1) || (lrwork == -1));

    if (!(sccolx || sccoly || (jobs[0] == 'N' || jobs[0] == 'n'))) {
        *info = -1;
    } else if (!(wntvec || (jobz[0] == 'N' || jobz[0] == 'n')
                       || (jobz[0] == 'F' || jobz[0] == 'f'))) {
        *info = -2;
    } else if (!(wntres || (jobr[0] == 'N' || jobr[0] == 'n'))
               || (wntres && (!wntvec))) {
        *info = -3;
    } else if (!(wntref || wntex || (jobf[0] == 'N' || jobf[0] == 'n'))) {
        *info = -4;
    } else if (!((whtsvd == 1) || (whtsvd == 2) || (whtsvd == 3) || (whtsvd == 4))) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if ((n < 0) || (n > m)) {
        *info = -7;
    } else if (ldx < m) {
        *info = -9;
    } else if (ldy < m) {
        *info = -11;
    } else if (!((nrnk == -2) || (nrnk == -1) || ((nrnk >= 1) && (nrnk <= n)))) {
        *info = -12;
    } else if ((tol < ZERO) || (tol >= ONE)) {
        *info = -13;
    } else if (ldz < m) {
        *info = -17;
    } else if ((wntref || wntex) && (ldb < m)) {
        *info = -20;
    } else if (ldw < n) {
        *info = -22;
    } else if (lds < n) {
        *info = -24;
    }

    if (*info == 0) {
        /* Compute the minimal and the optimal workspace requirements. */
        if (n == 0) {
            /* Quick return. All output except K is void. */
            if (lquery) {
                iwork[0] = 1;
                rwork[0] = 1;
                zwork[0] = 2;
                zwork[1] = 2;
            } else {
                *k = 0;
            }
            *info = 1;
            return;
        }

        iminwr = 1;
        mlrwrk = imax2(1, n);
        mlwork = 2;
        olwork = 2;
        switch (whtsvd) {
        case 1:
            /* MWRSVD = MAX(1,2*MIN(M,N)+MAX(M,N)) */
            mwrsvd = imax2(1, 2 * imin2(m, n) + imax2(m, n));
            mlwork = imax2(mlwork, mwrsvd);
            mlrwrk = imax2(mlrwrk, n + 5 * imin2(m, n));
            if (lquery) {
                cgesvd("O", "S", m, n, X, ldx, rwork, B, ldb, W, ldw,
                       zwork, -1, rdummy, &info1);
                lwrsvd = (INT)crealf(zwork[0]);
                olwork = imax2(olwork, lwrsvd);
            }
            break;
        case 2:
            /* MWRSDD = 2*min(M,N)^2 + 2*min(M,N) + max(M,N) */
            mwrsdd = 2 * imin2(m, n) * imin2(m, n) + 2 * imin2(m, n) + imax2(m, n);
            mlwork = imax2(mlwork, mwrsdd);
            iminwr = 8 * imin2(m, n);
            mlrwrk = imax2(mlrwrk, n +
                imax2(imax2(5 * imin2(m, n) * imin2(m, n) + 7 * imin2(m, n),
                            5 * imin2(m, n) * imin2(m, n) + 5 * imin2(m, n)),
                      2 * imax2(m, n) * imin2(m, n) + 2 * imin2(m, n) * imin2(m, n) + imin2(m, n)));
            if (lquery) {
                cgesdd("O", m, n, X, ldx, rwork, B, ldb, W, ldw,
                       zwork, -1, rdummy, iwork, &info1);
                lwrsdd = imax2(mwrsdd, (INT)crealf(zwork[0]));
                olwork = imax2(olwork, lwrsdd);
            }
            break;
        case 3:
            cgesvdq("H", "P", "N", "R", "R", m, n, X, ldx, rwork, Z, ldz, W, ldw,
                    &numrnk, iwork, -1, zwork, -1, rdummy, -1, &info1);
            iminwr = iwork[0];
            mwrsvq = (INT)crealf(zwork[1]);
            mlwork = imax2(mlwork, mwrsvq);
            mlrwrk = imax2(mlrwrk, n + (INT)rdummy[0]);
            if (lquery) {
                lwrsvq = (INT)crealf(zwork[0]);
                olwork = imax2(olwork, lwrsvq);
            }
            break;
        case 4:
            jsvopt = "J";
            cgejsv("F", "U", jsvopt, "R", "N", "P", m, n, X, ldx, rwork, Z, ldz,
                   W, ldw, zwork, -1, rdummy, -1, iwork, &info1);
            iminwr = iwork[0];
            mwrsvj = (INT)crealf(zwork[1]);
            mlwork = imax2(mlwork, mwrsvj);
            mlrwrk = imax2(mlrwrk, n + imax2(7, (INT)rdummy[0]));
            if (lquery) {
                lwrsvj = (INT)crealf(zwork[0]);
                olwork = imax2(olwork, lwrsvj);
            }
            break;
        }
        if (wntvec || wntex || (jobz[0] == 'F' || jobz[0] == 'f')) {
            jobzl = "V";
        } else {
            jobzl = "N";
        }
        /* Workspace calculation to the CGEEV call */
        mwrkev = imax2(1, 2 * n);
        mlwork = imax2(mlwork, mwrkev);
        mlrwrk = imax2(mlrwrk, n + 2 * n);
        if (lquery) {
            cgeev("N", jobzl, n, S, lds, eigs, W, ldw, W, ldw, zwork, -1,
                  rwork, &info1);
            lwrkev = (INT)crealf(zwork[0]);
            olwork = imax2(olwork, lwrkev);
        }

        if (liwork < iminwr && (!lquery)) *info = -30;
        if (lrwork < mlrwrk && (!lquery)) *info = -28;
        if (lzwork < mlwork && (!lquery)) *info = -26;
    }

    if (*info != 0) {
        xerbla("CGEDMD", -(*info));
        return;
    } else if (lquery) {
        /* Return minimal and optimal workspace sizes */
        iwork[0] = iminwr;
        rwork[0] = (f32)mlrwrk;
        zwork[0] = (c64)mlwork;
        zwork[1] = (c64)olwork;
        return;
    }

    ofl = slamch("O");
    small = slamch("S");
    badxy = 0;

    /* <1> Optional scaling of the snapshots (columns of X, Y) */
    if (sccolx) {
        /* The columns of X will be normalized. */
        *k = 0;
        for (i = 0; i < n; i++) {
            ssum = ONE;
            scale = ZERO;
            classq(m, &X[i * ldx], 1, &scale, &ssum);
            if (sisnan(scale) || sisnan(ssum)) {
                *k = 0;
                *info = -8;
                xerbla("CGEDMD", -(*info));
            }
            if ((scale != ZERO) && (ssum != ZERO)) {
                rootsc = sqrtf(ssum);
                tbig = ofl;
                if (rootsc > ONE) tbig = ofl / rootsc;
                if (scale >= tbig) {
                    /* Norm of X(:,i) overflows. */
                    clascl("G", 0, 0, scale, ONE / rootsc, m, 1, &X[i * ldx], ldx, &info2);
                    rwork[i] = -scale * (rootsc / (f32)m);
                } else {
                    /* X(:,i) will be scaled to unit 2-norm */
                    rwork[i] = scale * rootsc;
                    clascl("G", 0, 0, rwork[i], ONE, m, 1, &X[i * ldx], ldx, &info2);
                }
            } else {
                rwork[i] = ZERO;
                *k = *k + 1;
            }
        }
        if (*k == n) {
            /* All columns of X are zero. Return error code -8. */
            *k = 0;
            *info = -8;
            xerbla("CGEDMD", -(*info));
            return;
        }
        for (i = 0; i < n; i++) {
            /* Now, apply the same scaling to the columns of Y. */
            if (rwork[i] > ZERO) {
                cblas_csscal(m, ONE / rwork[i], &Y[i * ldy], 1);
            } else if (rwork[i] < ZERO) {
                clascl("G", 0, 0, -rwork[i], ONE / (f32)m, m, 1, &Y[i * ldy], ldy, &info2);
            } else if (cabsf(Y[cblas_icamax(m, &Y[i * ldy], 1) + i * ldy]) != ZERO) {
                /* X(:,i) is zero vector. For consistency, Y(:,i) should also
                 * be zero. */
                badxy = 1;
                if (jobs[0] == 'C' || jobs[0] == 'c')
                    cblas_csscal(m, ZERO, &Y[i * ldy], 1);
            }
        }
    }

    if (sccoly) {
        /* The columns of Y will be normalized. */
        for (i = 0; i < n; i++) {
            ssum = ONE;
            scale = ZERO;
            classq(m, &Y[i * ldy], 1, &scale, &ssum);
            if (sisnan(scale) || sisnan(ssum)) {
                *k = 0;
                *info = -10;
                xerbla("CGEDMD", -(*info));
            }
            if (scale != ZERO && (ssum != ZERO)) {
                rootsc = sqrtf(ssum);
                tbig = ofl;
                if (rootsc > ONE) tbig = ofl / rootsc;
                if (scale >= tbig) {
                    /* Norm of Y(:,i) overflows. */
                    clascl("G", 0, 0, scale, ONE / rootsc, m, 1, &Y[i * ldy], ldy, &info2);
                    rwork[i] = -scale * (rootsc / (f32)m);
                } else {
                    /* Y(:,i) will be scaled to unit 2-norm */
                    rwork[i] = scale * rootsc;
                    clascl("G", 0, 0, rwork[i], ONE, m, 1, &Y[i * ldy], ldy, &info2);
                }
            } else {
                rwork[i] = ZERO;
            }
        }
        for (i = 0; i < n; i++) {
            /* Now, apply the same scaling to the columns of X. */
            if (rwork[i] > ZERO) {
                cblas_csscal(m, ONE / rwork[i], &X[i * ldx], 1);
            } else if (rwork[i] < ZERO) {
                clascl("G", 0, 0, -rwork[i], ONE / (f32)m, m, 1, &X[i * ldx], ldx, &info2);
            } else if (cabsf(X[cblas_icamax(m, &X[i * ldx], 1) + i * ldx]) != ZERO) {
                /* Y(:,i) is zero vector. */
                badxy = 1;
            }
        }
    }

    /* <2> SVD of the data snapshot matrix X. */
    numrnk = n;
    switch (whtsvd) {
    case 1:
        cgesvd("O", "S", m, n, X, ldx, rwork, B, ldb, W, ldw,
               zwork, lzwork, &rwork[n], &info1);
        t_or_n = 'C';
        break;
    case 2:
        cgesdd("O", m, n, X, ldx, rwork, B, ldb, W, ldw,
               zwork, lzwork, &rwork[n], iwork, &info1);
        t_or_n = 'C';
        break;
    case 3:
        cgesvdq("H", "P", "N", "R", "R", m, n, X, ldx, rwork, Z, ldz, W, ldw,
                &numrnk, iwork, liwork, zwork, lzwork, &rwork[n], lrwork - n, &info1);
        clacpy("A", m, numrnk, Z, ldz, X, ldx);
        t_or_n = 'C';
        break;
    case 4:
        cgejsv("F", "U", jsvopt, "R", "N", "P", m, n, X, ldx, rwork, Z, ldz,
               W, ldw, zwork, lzwork, &rwork[n], lrwork - n, iwork, &info1);
        clacpy("A", m, n, Z, ldz, X, ldx);
        t_or_n = 'N';
        xscl1 = rwork[n];
        xscl2 = rwork[n + 1];
        if (xscl1 != xscl2) {
            /* Exceptional situation: CGEJSV returned the SVD in scaled form. */
            clascl("G", 0, 0, xscl1, xscl2, m, n, Y, ldy, &info2);
        }
        break;
    default:
        t_or_n = 'C';
        break;
    }

    if (info1 > 0) {
        /* The SVD selected subroutine did not converge. */
        *info = 2;
        return;
    }

    if (rwork[0] == ZERO) {
        /* The largest computed singular value of (scaled) X is zero. */
        *k = 0;
        *info = -8;
        xerbla("CGEDMD", -(*info));
        return;
    }

    /* <3> Determine the numerical rank of the data snapshots matrix X. */
    switch (nrnk) {
    case -1:
        *k = 1;
        for (i = 1; i < numrnk; i++) {
            if ((rwork[i] <= rwork[0] * tol) || (rwork[i] <= small)) break;
            *k = *k + 1;
        }
        break;
    case -2:
        *k = 1;
        for (i = 0; i < numrnk - 1; i++) {
            if ((rwork[i + 1] <= rwork[i] * tol) || (rwork[i] <= small)) break;
            *k = *k + 1;
        }
        break;
    default:
        *k = 1;
        for (i = 1; i < nrnk; i++) {
            if (rwork[i] <= small) break;
            *k = *k + 1;
        }
        break;
    }

    /* <4> Compute the Rayleigh quotient S = U^H * A * U. */
    if (t_or_n == 'N') {
        for (i = 0; i < *k; i++) {
            cblas_csscal(n, ONE / rwork[i], &W[i * ldw], 1);
        }
    } else {
        for (i = 0; i < *k; i++) {
            rwork[n + i] = ONE / rwork[i];
        }
        for (j = 0; j < n; j++) {
            for (i = 0; i < *k; i++) {
                W[i + j * ldw] = rwork[n + i] * W[i + j * ldw];
            }
        }
    }

    if (wntref) {
        /* Need A*U(:,1:K)=Y*V_k*inv(diag(WORK(1:K))). */
        cblas_cgemm(CblasColMajor, CblasNoTrans,
                    (t_or_n == 'C') ? CblasConjTrans : CblasNoTrans,
                    m, *k, n, &CONE, Y, ldy, W, ldw, &CZERO, Z, ldz);
        clacpy("A", m, *k, Z, ldz, B, ldb);
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    *k, *k, m, &CONE, X, ldx, Z, ldz, &CZERO, S, lds);
    } else {
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    *k, n, m, &CONE, X, ldx, Y, ldy, &CZERO, Z, ldz);
        cblas_cgemm(CblasColMajor, CblasNoTrans,
                    (t_or_n == 'C') ? CblasConjTrans : CblasNoTrans,
                    *k, *k, n, &CONE, Z, ldz, W, ldw, &CZERO, S, lds);
        if (wntres || wntex) {
            if (t_or_n == 'N') {
                clacpy("A", n, *k, W, ldw, Z, ldz);
            } else {
                clacpy("A", *k, n, W, ldw, Z, ldz);
            }
        }
    }

    /* <5> Compute the Ritz values and (if requested) the right eigenvectors of
     * the Rayleigh quotient. */
    cgeev("N", jobzl, *k, S, lds, eigs, W, ldw, W, ldw, zwork, lzwork,
          &rwork[n], &info1);

    if (info1 > 0) {
        /* CGEEV failed to compute the eigenvalues and eigenvectors. */
        *info = 3;
        return;
    }

    /* <6> Compute the eigenvectors (if requested) and the residuals (if
     * requested). */
    if (wntvec || wntex) {
        if (wntres) {
            if (wntref) {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, *k, *k, &CONE, Z, ldz, W, ldw, &CZERO, Y, ldy);
            } else {
                cblas_cgemm(CblasColMajor,
                            (t_or_n == 'C') ? CblasConjTrans : CblasNoTrans,
                            CblasNoTrans, n, *k, *k, &CONE, Z, ldz, W, ldw, &CZERO, S, lds);
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, *k, n, &CONE, Y, ldy, S, lds, &CZERO, Z, ldz);
                clacpy("A", m, *k, Z, ldz, Y, ldy);
                if (wntex) clacpy("A", m, *k, Z, ldz, B, ldb);
            }
        } else if (wntex) {
            cblas_cgemm(CblasColMajor,
                        (t_or_n == 'C') ? CblasConjTrans : CblasNoTrans,
                        CblasNoTrans, n, *k, *k, &CONE, Z, ldz, W, ldw, &CZERO, S, lds);
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, *k, n, &CONE, Y, ldy, S, lds, &CZERO, B, ldb);
        }

        /* Compute the Ritz vectors */
        if (wntvec)
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, *k, *k, &CONE, X, ldx, W, ldw, &CZERO, Z, ldz);

        if (wntres) {
            for (i = 0; i < *k; i++) {
                c64 neg = -eigs[i];
                cblas_caxpy(m, &neg, &Z[i * ldz], 1, &Y[i * ldy], 1);
                res[i] = cblas_scnrm2(m, &Y[i * ldy], 1);
            }
        }
    }

    if (whtsvd == 4) {
        rwork[n] = xscl1;
        rwork[n + 1] = xscl2;
    }

    /* Successful exit. */
    if (!badxy) {
        *info = 0;
    } else {
        /* A warning on possible data inconsistency. */
        *info = 4;
    }
}
