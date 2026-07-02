/**
 * @file dgedmd.c
 * @brief DGEDMD computes the Dynamic Mode Decomposition (DMD) for a pair of
 *        data snapshot matrices.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/** @cond */
static inline INT imax2(INT a, INT b) { return (a > b) ? a : b; }
static inline INT imin2(INT a, INT b) { return (a < b) ? a : b; }
static inline INT imax3(INT a, INT b, INT c) { return imax2(imax2(a, b), c); }
/** @endcond */

/**
 * DGEDMD computes the Dynamic Mode Decomposition (DMD) for a pair of data
 * snapshot matrices. For the input matrices X and Y such that Y = A*X with an
 * unaccessible matrix A, DGEDMD computes a certain number of Ritz pairs of A
 * using the standard Rayleigh-Ritz extraction from a subspace of range(X) that
 * is determined using the leading left singular vectors of X. Optionally,
 * DGEDMD returns the residuals of the computed Ritz pairs, the information
 * needed for a refinement of the Ritz vectors, or the eigenvectors of the
 * Exact DMD.
 *
 * @param[in]     jobs    Determines whether the initial data snapshots are
 *                        scaled by a diagonal matrix.
 *                        'S' :: X and Y are multiplied with a diagonal matrix D
 *                               so that X*D has unit nonzero columns.
 *                        'C' :: as 'S'; additionally if X(:,i)=0 and Y(:,i)/=0
 *                               then Y(:,i) is set to zero and a warning raised.
 *                        'Y' :: X and Y are multiplied by a diagonal matrix D
 *                               so that Y*D has unit nonzero columns.
 *                        'N' :: No data scaling.
 * @param[in]     jobz    Determines whether the eigenvectors (Koopman modes)
 *                        will be computed.
 *                        'V' :: eigenvectors computed and returned in Z.
 *                        'F' :: eigenvectors returned in factored form
 *                               X(:,1:K)*W.
 *                        'N' :: eigenvectors are not computed.
 * @param[in]     jobr    Determines whether to compute the residuals.
 *                        'R' :: residuals computed and stored in RES.
 *                        'N' :: residuals are not computed.
 * @param[in]     jobf    Specifies whether to store information needed for
 *                        post-processing.
 *                        'R' :: matrix needed for the refinement of the Ritz
 *                               vectors is stored in B.
 *                        'E' :: unscaled eigenvectors of the Exact DMD are
 *                               stored in B.
 *                        'N' :: no eigenvector refinement data is computed.
 * @param[in]     whtsvd  Selects the SVD algorithm from LAPACK, in {1,2,3,4}.
 *                        1 :: dgesvd, 2 :: dgesdd, 3 :: dgesvdq, 4 :: dgejsv.
 * @param[in]     m       The state space dimension (rows of X, Y). m >= 0.
 * @param[in]     n       The number of data snapshot pairs. 0 <= n <= m.
 * @param[in,out] X       Double precision M-by-N array. On entry the data
 *                        snapshot matrix X. On exit, the leading K columns
 *                        contain a POD basis (leading K left singular vectors).
 * @param[in]     ldx     The leading dimension of X. ldx >= m.
 * @param[in,out] Y       Double precision M-by-N array. On entry the data
 *                        snapshot matrix Y. On exit, if jobr=='R' the leading K
 *                        columns contain the residual vectors; if jobr=='N', Y
 *                        contains the original input data, scaled per jobs.
 * @param[in]     ldy     The leading dimension of Y. ldy >= m.
 * @param[in]     nrnk    Determines how to compute the numerical rank.
 *                        -1 :: sigma(i) truncated if sigma(i) <= tol*sigma(1).
 *                        -2 :: sigma(i) truncated if sigma(i) <= tol*sigma(i-1).
 *                        0 < nrnk <= n :: at most nrnk largest singular values.
 * @param[in]     tol     Tolerance for truncating small singular values,
 *                        0 <= tol < 1.
 * @param[out]    k       0 <= k <= n. The dimension of the POD basis and the
 *                        number of computed Ritz pairs.
 * @param[out]    reig    N-by-1 array. reig[0:K] contain the real parts of the
 *                        computed eigenvalues.
 * @param[out]    imeig   N-by-1 array. imeig[0:K] contain the imaginary parts
 *                        of the computed eigenvalues. Complex conjugate pairs
 *                        have consecutive indices, positive imaginary part
 *                        first.
 * @param[out]    Z       M-by-N array. If jobz=='V', contains the real Ritz
 *                        vectors. If jobz=='F', the descriptions hold for
 *                        X(:,1:K)*W(1:K,1:K).
 * @param[in]     ldz     The leading dimension of Z. ldz >= m.
 * @param[out]    res     N-by-1 array. res[0:K] contain the residuals for the K
 *                        computed Ritz pairs.
 * @param[out]    B       M-by-N array. If jobf=='R', B(1:M,1:K) contains
 *                        A*U(:,1:K). If jobf=='E', B(1:M,1:K) contains
 *                        A*U(:,1:K)*W(1:K,1:K). If jobf=='N', not referenced.
 * @param[in]     ldb     The leading dimension of B. ldb >= m.
 * @param[out]    W       N-by-N array. On exit W(1:K,1:K) contains the K
 *                        computed eigenvectors of the Rayleigh quotient. Also
 *                        used as workspace for the right singular vectors of X.
 * @param[in]     ldw     The leading dimension of W. ldw >= n.
 * @param[out]    S       N-by-N array. S(1:K,1:K) is used for the Rayleigh
 *                        quotient, overwritten during the eigenvalue
 *                        decomposition by dgeev.
 * @param[in]     lds     The leading dimension of S. lds >= n.
 * @param[out]    work    Workspace/output array. On exit work[0:N] contains the
 *                        singular values of (scaled) X. If whtsvd==4, work[N]
 *                        and work[N+1] contain a scaling factor. On workspace
 *                        query, work[0] is the minimal and work[1] the optimal
 *                        workspace length.
 * @param[in]     lwork   The length of work. If lwork = -1, a workspace query
 *                        is assumed.
 * @param[out]    iwork   Integer workspace, required only if whtsvd in {2,3,4}.
 * @param[in]     liwork  The length of iwork. If liwork = -1, a workspace query
 *                        is assumed.
 * @param[out]    info    = -i < 0 : the i-th argument had an illegal value.
 *                        = 0 : successful return.
 *                        = 1 : void input (m=0 or n=0), quick exit.
 *                        = 2 : the SVD computation of X did not converge.
 *                        = 3 : the computation of the eigenvalues did not
 *                              converge.
 *                        = 4 : data inconsistency detected and (for jobs=='C')
 *                              a column of Y was zeroed; warning flag.
 */
void dgedmd(const char* jobs, const char* jobz, const char* jobr,
            const char* jobf, const INT whtsvd, const INT m, const INT n,
            f64* restrict X, const INT ldx, f64* restrict Y, const INT ldy,
            const INT nrnk, const f64 tol, INT* k, f64* restrict reig,
            f64* restrict imeig, f64* restrict Z, const INT ldz,
            f64* restrict res, f64* restrict B, const INT ldb,
            f64* restrict W, const INT ldw, f64* restrict S, const INT lds,
            f64* restrict work, const INT lwork, INT* restrict iwork,
            const INT liwork, INT* info)
{
    f64 ofl, rootsc, scale, small, ssum, xscl1, xscl2, tbig;
    INT i, j, iminwr, info1, info2, lwrkev, lwrsdd, lwrsvd, lwrsvq,
        mlwork, mwrkev, mwrsdd, mwrsvd, mwrsvj, mwrsvq, numrnk, olwork;
    INT badxy, lquery, sccolx, sccoly, wntex, wntref, wntres, wntvec;
    char t_or_n;
    const char* jobzl;
    const char* jsvopt;
    f64 ab[4], rdummy[2], rdummy2[2];

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
    lquery = ((lwork == -1) || (liwork == -1));

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
        *info = -18;
    } else if ((wntref || wntex) && (ldb < m)) {
        *info = -21;
    } else if (ldw < n) {
        *info = -23;
    } else if (lds < n) {
        *info = -25;
    }

    if (*info == 0) {
        /* Compute the minimal and the optimal workspace requirements. */
        if (n == 0) {
            /* Quick return. All output except K is void. */
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
        mlwork = imax2(2, n);
        olwork = imax2(2, n);
        iminwr = 1;
        switch (whtsvd) {
        case 1:
            /* MWRSVD = MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N)) */
            mwrsvd = imax3(1, 3 * imin2(m, n) + imax2(m, n), 5 * imin2(m, n));
            mlwork = imax2(mlwork, n + mwrsvd);
            if (lquery) {
                dgesvd("O", "S", m, n, X, ldx, work, B, ldb, W, ldw,
                       rdummy, -1, &info1);
                lwrsvd = imax2(mwrsvd, (INT)rdummy[0]);
                olwork = imax2(olwork, n + lwrsvd);
            }
            break;
        case 2:
            /* MWRSDD = 3*MIN(M,N)^2 +
             *          MAX(MAX(M,N),5*MIN(M,N)^2+4*MIN(M,N)) */
            mwrsdd = 3 * imin2(m, n) * imin2(m, n) +
                     imax2(imax2(m, n), 5 * imin2(m, n) * imin2(m, n) + 4 * imin2(m, n));
            mlwork = imax2(mlwork, n + mwrsdd);
            iminwr = 8 * imin2(m, n);
            if (lquery) {
                dgesdd("O", m, n, X, ldx, work, B, ldb, W, ldw,
                       rdummy, -1, iwork, &info1);
                lwrsdd = imax2(mwrsdd, (INT)rdummy[0]);
                olwork = imax2(olwork, n + lwrsdd);
            }
            break;
        case 3:
            dgesvdq("H", "P", "N", "R", "R", m, n, X, ldx, work, Z, ldz,
                    W, ldw, &numrnk, iwork, liwork, rdummy, -1, rdummy2, -1,
                    &info1);
            iminwr = iwork[0];
            mwrsvq = (INT)rdummy[1];
            mlwork = imax2(mlwork, n + mwrsvq + (INT)rdummy2[0]);
            if (lquery) {
                lwrsvq = imax2(mwrsvq, (INT)rdummy[0]);
                olwork = imax2(olwork, n + lwrsvq + (INT)rdummy2[0]);
            }
            break;
        case 4:
            jsvopt = "J";
            /* MWRSVJ = MAX( 7, 2*M+N, 4*N+N*N, 2*N+N*N+6 ) */
            mwrsvj = imax2(imax2(7, 2 * m + n), imax2(4 * n + n * n, 2 * n + n * n + 6));
            mlwork = imax2(mlwork, n + mwrsvj);
            iminwr = imax2(3, m + 3 * n);
            if (lquery) {
                olwork = imax2(olwork, n + mwrsvj);
            }
            break;
        }
        if (wntvec || wntex || (jobz[0] == 'F' || jobz[0] == 'f')) {
            jobzl = "V";
        } else {
            jobzl = "N";
        }
        /* Workspace calculation to the DGEEV call */
        if (jobzl[0] == 'V' || jobzl[0] == 'v') {
            mwrkev = imax2(1, 4 * n);
        } else {
            mwrkev = imax2(1, 3 * n);
        }
        mlwork = imax2(mlwork, n + mwrkev);
        if (lquery) {
            dgeev("N", jobzl, n, S, lds, reig, imeig, W, ldw, W, ldw,
                  rdummy, -1, &info1);
            lwrkev = imax2(mwrkev, (INT)rdummy[0]);
            olwork = imax2(olwork, n + lwrkev);
        }

        if (liwork < iminwr && (!lquery)) *info = -29;
        if (lwork < mlwork && (!lquery)) *info = -27;
    }

    if (*info != 0) {
        xerbla("DGEDMD", -(*info));
        return;
    } else if (lquery) {
        /* Return minimal and optimal workspace sizes */
        iwork[0] = iminwr;
        work[0] = (f64)mlwork;
        work[1] = (f64)olwork;
        return;
    }

    ofl = dlamch("O");
    small = dlamch("S");
    badxy = 0;

    /* <1> Optional scaling of the snapshots (columns of X, Y) */
    if (sccolx) {
        /* The columns of X will be normalized. */
        *k = 0;
        for (i = 0; i < n; i++) {
            ssum = ONE;
            scale = ZERO;
            dlassq(m, &X[i * ldx], 1, &scale, &ssum);
            if (disnan(scale) || disnan(ssum)) {
                *k = 0;
                *info = -8;
                xerbla("DGEDMD", -(*info));
            }
            if ((scale != ZERO) && (ssum != ZERO)) {
                rootsc = sqrt(ssum);
                tbig = ofl;
                if (rootsc > ONE) tbig = ofl / rootsc;
                if (scale >= tbig) {
                    /* Norm of X(:,i) overflows. */
                    dlascl("G", 0, 0, scale, ONE / rootsc, m, 1, &X[i * ldx], m, &info2);
                    work[i] = -scale * (rootsc / (f64)m);
                } else {
                    /* X(:,i) will be scaled to unit 2-norm */
                    work[i] = scale * rootsc;
                    dlascl("G", 0, 0, work[i], ONE, m, 1, &X[i * ldx], m, &info2);
                }
            } else {
                work[i] = ZERO;
                *k = *k + 1;
            }
        }
        if (*k == n) {
            /* All columns of X are zero. Return error code -8. */
            *k = 0;
            *info = -8;
            xerbla("DGEDMD", -(*info));
            return;
        }
        for (i = 0; i < n; i++) {
            /* Now, apply the same scaling to the columns of Y. */
            if (work[i] > ZERO) {
                cblas_dscal(m, ONE / work[i], &Y[i * ldy], 1);
            } else if (work[i] < ZERO) {
                dlascl("G", 0, 0, -work[i], ONE / (f64)m, m, 1, &Y[i * ldy], m, &info2);
            } else if (Y[cblas_idamax(m, &Y[i * ldy], 1) + i * ldy] != ZERO) {
                /* X(:,i) is zero vector. For consistency, Y(:,i) should also
                 * be zero. */
                badxy = 1;
                if (jobs[0] == 'C' || jobs[0] == 'c')
                    cblas_dscal(m, ZERO, &Y[i * ldy], 1);
            }
        }
    }

    if (sccoly) {
        /* The columns of Y will be normalized. */
        for (i = 0; i < n; i++) {
            ssum = ONE;
            scale = ZERO;
            dlassq(m, &Y[i * ldy], 1, &scale, &ssum);
            if (disnan(scale) || disnan(ssum)) {
                *k = 0;
                *info = -10;
                xerbla("DGEDMD", -(*info));
            }
            if (scale != ZERO && (ssum != ZERO)) {
                rootsc = sqrt(ssum);
                tbig = ofl;
                if (rootsc > ONE) tbig = ofl / rootsc;
                if (scale >= tbig) {
                    /* Norm of Y(:,i) overflows. */
                    dlascl("G", 0, 0, scale, ONE / rootsc, m, 1, &Y[i * ldy], m, &info2);
                    work[i] = -scale * (rootsc / (f64)m);
                } else {
                    /* Y(:,i) will be scaled to unit 2-norm */
                    work[i] = scale * rootsc;
                    dlascl("G", 0, 0, work[i], ONE, m, 1, &Y[i * ldy], m, &info2);
                }
            } else {
                work[i] = ZERO;
            }
        }
        for (i = 0; i < n; i++) {
            /* Now, apply the same scaling to the columns of X. */
            if (work[i] > ZERO) {
                cblas_dscal(m, ONE / work[i], &X[i * ldx], 1);
            } else if (work[i] < ZERO) {
                dlascl("G", 0, 0, -work[i], ONE / (f64)m, m, 1, &X[i * ldx], m, &info2);
            } else if (X[cblas_idamax(m, &X[i * ldx], 1) + i * ldx] != ZERO) {
                /* Y(:,i) is zero vector. */
                badxy = 1;
            }
        }
    }

    /* <2> SVD of the data snapshot matrix X. */
    numrnk = n;
    switch (whtsvd) {
    case 1:
        dgesvd("O", "S", m, n, X, ldx, work, B, ldb, W, ldw,
               &work[n], lwork - n, &info1);
        t_or_n = 'T';
        break;
    case 2:
        dgesdd("O", m, n, X, ldx, work, B, ldb, W, ldw,
               &work[n], lwork - n, iwork, &info1);
        t_or_n = 'T';
        break;
    case 3:
        dgesvdq("H", "P", "N", "R", "R", m, n, X, ldx, work, Z, ldz, W, ldw,
                &numrnk, iwork, liwork, &work[n + imax2(2, m)],
                lwork - n - imax2(2, m), &work[n], imax2(2, m), &info1);
        dlacpy("A", m, numrnk, Z, ldz, X, ldx);
        t_or_n = 'T';
        break;
    case 4:
        dgejsv("F", "U", jsvopt, "N", "N", "P", m, n, X, ldx, work, Z, ldz,
               W, ldw, &work[n], lwork - n, iwork, &info1);
        dlacpy("A", m, n, Z, ldz, X, ldx);
        t_or_n = 'N';
        xscl1 = work[n];
        xscl2 = work[n + 1];
        if (xscl1 != xscl2) {
            /* Exceptional situation: DGEJSV returned the SVD in scaled form. */
            dlascl("G", 0, 0, xscl1, xscl2, m, n, Y, ldy, &info2);
        }
        break;
    default:
        t_or_n = 'T';
        break;
    }

    if (info1 > 0) {
        /* The SVD selected subroutine did not converge. */
        *info = 2;
        return;
    }

    if (work[0] == ZERO) {
        /* The largest computed singular value of (scaled) X is zero. */
        *k = 0;
        *info = -8;
        xerbla("DGEDMD", -(*info));
        return;
    }

    /* <3> Determine the numerical rank of the data snapshots matrix X. */
    switch (nrnk) {
    case -1:
        *k = 1;
        for (i = 1; i < numrnk; i++) {
            if ((work[i] <= work[0] * tol) || (work[i] <= small)) break;
            *k = *k + 1;
        }
        break;
    case -2:
        *k = 1;
        for (i = 0; i < numrnk - 1; i++) {
            if ((work[i + 1] <= work[i] * tol) || (work[i] <= small)) break;
            *k = *k + 1;
        }
        break;
    default:
        *k = 1;
        for (i = 1; i < nrnk; i++) {
            if (work[i] <= small) break;
            *k = *k + 1;
        }
        break;
    }

    /* <4> Compute the Rayleigh quotient S = U^T * A * U. */
    if (t_or_n == 'N') {
        for (i = 0; i < *k; i++) {
            cblas_dscal(n, ONE / work[i], &W[i * ldw], 1);
        }
    } else {
        for (i = 0; i < *k; i++) {
            work[n + i] = ONE / work[i];
        }
        for (j = 0; j < n; j++) {
            for (i = 0; i < *k; i++) {
                W[i + j * ldw] = (work[n + i]) * W[i + j * ldw];
            }
        }
    }

    if (wntref) {
        /* Need A*U(:,1:K)=Y*V_k*inv(diag(WORK(1:K))). */
        cblas_dgemm(CblasColMajor, CblasNoTrans,
                    (t_or_n == 'T') ? CblasTrans : CblasNoTrans,
                    m, *k, n, ONE, Y, ldy, W, ldw, ZERO, Z, ldz);
        dlacpy("A", m, *k, Z, ldz, B, ldb);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    *k, *k, m, ONE, X, ldx, Z, ldz, ZERO, S, lds);
    } else {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    *k, n, m, ONE, X, ldx, Y, ldy, ZERO, Z, ldz);
        cblas_dgemm(CblasColMajor, CblasNoTrans,
                    (t_or_n == 'T') ? CblasTrans : CblasNoTrans,
                    *k, *k, n, ONE, Z, ldz, W, ldw, ZERO, S, lds);
        if (wntres || wntex) {
            if (t_or_n == 'N') {
                dlacpy("A", n, *k, W, ldw, Z, ldz);
            } else {
                dlacpy("A", *k, n, W, ldw, Z, ldz);
            }
        }
    }

    /* <5> Compute the Ritz values and (if requested) the right eigenvectors of
     * the Rayleigh quotient. */
    dgeev("N", jobzl, *k, S, lds, reig, imeig, W, ldw, W, ldw,
          &work[n], lwork - n, &info1);

    if (info1 > 0) {
        /* DGEEV failed to compute the eigenvalues and eigenvectors. */
        *info = 3;
        return;
    }

    /* <6> Compute the eigenvectors (if requested) and the residuals (if
     * requested). */
    if (wntvec || wntex) {
        if (wntres) {
            if (wntref) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, *k, *k, ONE, Z, ldz, W, ldw, ZERO, Y, ldy);
            } else {
                cblas_dgemm(CblasColMajor,
                            (t_or_n == 'T') ? CblasTrans : CblasNoTrans,
                            CblasNoTrans, n, *k, *k, ONE, Z, ldz, W, ldw, ZERO, S, lds);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, *k, n, ONE, Y, ldy, S, lds, ZERO, Z, ldz);
                dlacpy("A", m, *k, Z, ldz, Y, ldy);
                if (wntex) dlacpy("A", m, *k, Z, ldz, B, ldb);
            }
        } else if (wntex) {
            cblas_dgemm(CblasColMajor,
                        (t_or_n == 'T') ? CblasTrans : CblasNoTrans,
                        CblasNoTrans, n, *k, *k, ONE, Z, ldz, W, ldw, ZERO, S, lds);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, *k, n, ONE, Y, ldy, S, lds, ZERO, B, ldb);
        }

        /* Compute the real form of the Ritz vectors */
        if (wntvec)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, *k, *k, ONE, X, ldx, W, ldw, ZERO, Z, ldz);

        if (wntres) {
            i = 0;
            while (i < *k) {
                if (imeig[i] == ZERO) {
                    /* have a real eigenvalue with real eigenvector */
                    cblas_daxpy(m, -reig[i], &Z[i * ldz], 1, &Y[i * ldy], 1);
                    res[i] = cblas_dnrm2(m, &Y[i * ldy], 1);
                    i = i + 1;
                } else {
                    /* Have a complex conjugate pair. */
                    ab[0] = reig[i];
                    ab[1] = -imeig[i];
                    ab[2] = imeig[i];
                    ab[3] = reig[i];
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, 2, 2, -ONE, &Z[i * ldz], ldz, ab, 2, ONE,
                                &Y[i * ldy], ldy);
                    res[i] = dlange("F", m, 2, &Y[i * ldy], ldy, &work[n]);
                    res[i + 1] = res[i];
                    i = i + 2;
                }
            }
        }
    }

    if (whtsvd == 4) {
        work[n] = xscl1;
        work[n + 1] = xscl2;
    }

    /* Successful exit. */
    if (!badxy) {
        *info = 0;
    } else {
        /* A warning on possible data inconsistency. */
        *info = 4;
    }
}
