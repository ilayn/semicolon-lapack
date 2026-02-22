/**
 * @file dlatrs3.c
 * @brief DLATRS3 solves a triangular system with scale factors to prevent overflow.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLATRS3 solves one of the triangular systems
 *
 *    A * X = B * diag(scale)  or  A**T * X = B * diag(scale)
 *
 * with scaling to prevent overflow. Here A is an upper or lower
 * triangular matrix, A**T denotes the transpose of A. X and B are
 * n by nrhs matrices and scale is an nrhs element vector of scaling
 * factors. A scaling factor scale(j) is usually less than or equal
 * to 1, chosen such that X(:,j) is less than the overflow threshold.
 * If the matrix A is singular (A(j,j) = 0 for some j), then
 * a non-trivial solution to A*X = 0 is returned. If the system is
 * so badly scaled that the solution cannot be represented as
 * (1/scale(k))*X(:,k), then x(:,k) = 0 and scale(k) is returned.
 *
 * This is a BLAS-3 version of LATRS for solving several right
 * hand sides simultaneously.
 *
 * Reference:
 *   C. C. Kjelgaard Mikkelsen, A. B. Schwarz and L. Karlsson (2019).
 *   Parallel robust solution of triangular linear systems. Concurrency
 *   and Computation: Practice and Experience, 31(19), e5064.
 *
 * @param[in]     uplo    'U': Upper triangular; 'L': Lower triangular.
 * @param[in]     trans   'N': Solve A * X = B*diag(scale);
 *                        'T'/'C': Solve A**T * X = B*diag(scale).
 * @param[in]     diag    'N': Non-unit triangular; 'U': Unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     nrhs    The number of columns of X (nrhs >= 0).
 * @param[in]     A       The triangular matrix A. Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of A (lda >= max(1,n)).
 * @param[in,out] X       On entry, the right hand side B.
 *                        On exit, overwritten by the solution matrix X.
 *                        Array of dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X (ldx >= max(1,n)).
 * @param[out]    scale   Array of dimension nrhs. The scaling factor s(k)
 *                        for the triangular system with right-hand side k.
 * @param[in,out] cnorm   Array of dimension n. If normin='Y', contains
 *                        column norms on entry. If normin='N', returns
 *                        the 1-norm of offdiagonal parts.
 * @param[out]    work    Workspace array of dimension lwork.
 *                        On exit, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work.
 *                        If min(n,nrhs) = 0, lwork >= 1, else
 *                        lwork >= 2*NBA * max(NBA, min(nrhs, 32)), where
 *                        NBA = (n + NB - 1)/NB and NB is the optimal block size.
 *                        If lwork = -1, a workspace query is performed.
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void dlatrs3(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const INT n,
    const INT nrhs,
    const f64* restrict A,
    const INT lda,
    f64* restrict X,
    const INT ldx,
    f64* restrict scale,
    f64* restrict cnorm,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    /* Parameters from Fortran - match LAPACK exactly */
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
#define NRHSMIN 2
#define NBRHS 32
#define NBMIN 8
#define NBMAX 64

    /* Local arrays - match Fortran local arrays */
    f64 W[NBMAX];
    f64 XNRM[NBRHS];

    /* Local scalars */
    INT upper, notran, nounit, lquery;
    INT awrk, i, ifirst, iinc, ilast, ii, i1, i2, j;
    INT jfirst, jinc, jlast, j1, j2, k, kk, k1, k2;
    INT lanrm, lds, lscale, nb, nba, nbx, rhs, lwmin;
    f64 anrm, bignum, bnrm, rscal, scal, scaloc;
    f64 scamin, smlnum, tmax;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    lquery = (lwork == -1);

    /* Partition A and X into blocks */
    /* NB from ILAENV for DLATRS is typically 64, but we use NBMIN=8 as floor */
    nb = NBMAX;  /* Default block size for DLATRS */
    if (nb < NBMIN) nb = NBMIN;
    if (nb > NBMAX) nb = NBMAX;

    nba = (n + nb - 1) / nb;
    if (nba < 1) nba = 1;
    nbx = (nrhs + NBRHS - 1) / NBRHS;
    if (nbx < 1) nbx = 1;

    /* Compute the workspace */
    /* The workspace comprises two parts:
     * 1. Local scale factors: NBA * max(NBA, min(NRHS, NBRHS))
     * 2. Upper bounds of blocks: NBA * NBA
     */
    INT minrhs = (nrhs < NBRHS) ? nrhs : NBRHS;
    lscale = nba * ((nba > minrhs) ? nba : minrhs);
    lds = nba;
    lanrm = nba * nba;
    awrk = lscale;

    if (n == 0 || nrhs == 0) {
        lwmin = 1;
    } else {
        lwmin = lscale + lanrm;
    }
    work[0] = (f64)lwmin;

    /* Test the input parameters */
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (!(normin[0] == 'Y' || normin[0] == 'y') &&
               !(normin[0] == 'N' || normin[0] == 'n')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (nrhs < 0) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (!lquery && lwork < lwmin) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("DLATRS3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Initialize scaling factors */
    for (kk = 0; kk < nrhs; kk++) {
        scale[kk] = ONE;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return;
    }

    /* Determine machine dependent constants to control overflow */
    bignum = dlamch("O");
    smlnum = dlamch("S");

    /* Use unblocked code for small problems */
    if (nrhs < NRHSMIN) {
        dlatrs(uplo, trans, diag, normin, n, A, lda, &X[0], &scale[0], cnorm, info);
        for (k = 1; k < nrhs; k++) {
            dlatrs(uplo, trans, diag, "Y", n, A, lda, &X[k * ldx], &scale[k], cnorm, info);
        }
        return;
    }

    /* Compute norms of blocks of A excluding diagonal blocks and find
     * the block with the largest norm TMAX. */
    tmax = ZERO;
    for (j = 0; j < nba; j++) {
        j1 = j * nb;                        /* 0-based start */
        j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;  /* 0-based exclusive end */

        if (upper) {
            ifirst = 0;
            ilast = j - 1;
        } else {
            ifirst = j + 1;
            ilast = nba - 1;
        }

        for (i = ifirst; i <= ilast; i++) {
            i1 = i * nb;
            i2 = ((i + 1) * nb < n) ? (i + 1) * nb : n;

            /* Compute upper bound of A( i1:i2-1, j1:j2-1 ) */
            if (notran) {
                anrm = dlange("I", i2 - i1, j2 - j1, &A[i1 + j1 * lda], lda, W);
                work[awrk + i + j * nba] = anrm;
            } else {
                anrm = dlange("1", i2 - i1, j2 - j1, &A[i1 + j1 * lda], lda, W);
                work[awrk + j + i * nba] = anrm;
            }
            if (tmax < anrm) {
                tmax = anrm;
            }
        }
    }

    if (!(tmax <= dlamch("O"))) {
        /* Some matrix entries have huge absolute value. Fall back to LATRS.
         * Set normin = 'N' for every right-hand side to force computation
         * of TSCAL in LATRS to avoid overflow. */
        for (k = 0; k < nrhs; k++) {
            dlatrs(uplo, trans, diag, "N", n, A, lda, &X[k * ldx], &scale[k], cnorm, info);
        }
        return;
    }

    /* Loop over block columns of X */
    for (k = 0; k < nbx; k++) {
        /* K1: column index of the first column in X( J, K ) (0-based)
         * K2: column index of the first column in X( J, K+1 ) (0-based exclusive) */
        k1 = k * NBRHS;
        k2 = ((k + 1) * NBRHS < nrhs) ? (k + 1) * NBRHS : nrhs;

        /* Initialize local scaling factors of current block column */
        for (kk = 0; kk < k2 - k1; kk++) {
            for (i = 0; i < nba; i++) {
                work[i + (kk + 1) * lds] = ONE;
            }
        }

        if (notran) {
            /* Solve A * X(:, K1:K2-1) = B * diag(scale(K1:K2-1)) */
            if (upper) {
                jfirst = nba - 1;
                jlast = 0;
                jinc = -1;
            } else {
                jfirst = 0;
                jlast = nba - 1;
                jinc = 1;
            }
        } else {
            /* Solve A**T * X(:, K1:K2-1) = B * diag(scale(K1:K2-1)) */
            if (upper) {
                jfirst = 0;
                jlast = nba - 1;
                jinc = 1;
            } else {
                jfirst = nba - 1;
                jlast = 0;
                jinc = -1;
            }
        }

        for (j = jfirst; (jinc > 0) ? (j <= jlast) : (j >= jlast); j += jinc) {
            /* J1: row index of the first row in A( J, J ) (0-based)
             * J2: row index past the last row (0-based exclusive) */
            j1 = j * nb;
            j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

            /* Solve op(A( J, J )) * X( J, RHS ) = SCALOC * B( J, RHS )
             * for all right-hand sides in the current block column. */
            for (kk = 0; kk < k2 - k1; kk++) {
                rhs = k1 + kk;
                if (kk == 0) {
                    dlatrs(uplo, trans, diag, "N", j2 - j1,
                           &A[j1 + j1 * lda], lda, &X[j1 + rhs * ldx],
                           &scaloc, cnorm, info);
                } else {
                    dlatrs(uplo, trans, diag, "Y", j2 - j1,
                           &A[j1 + j1 * lda], lda, &X[j1 + rhs * ldx],
                           &scaloc, cnorm, info);
                }

                /* Find largest absolute value entry in X( J1:J2-1, RHS ) */
                XNRM[kk] = dlange("I", j2 - j1, 1, &X[j1 + rhs * ldx], ldx, W);

                if (scaloc == ZERO) {
                    /* LATRS found that A is singular through A(j,j) = 0.
                     * Reset computation: x(0:n-1) = 0, x(j) = 1, scale = 0 */
                    scale[rhs] = ZERO;
                    for (ii = 0; ii < j1; ii++) {
                        X[ii + kk * ldx] = ZERO;
                    }
                    for (ii = j2; ii < n; ii++) {
                        X[ii + kk * ldx] = ZERO;
                    }
                    /* Discard the local scale factors */
                    for (ii = 0; ii < nba; ii++) {
                        work[ii + (kk + 1) * lds] = ONE;
                    }
                    scaloc = ONE;
                } else if (scaloc * work[j + (kk + 1) * lds] == ZERO) {
                    /* LATRS computed a valid scale factor, but combined with
                     * the current scaling the solution does not have a
                     * scale factor > 0. */
                    scal = work[j + (kk + 1) * lds] / smlnum;
                    scaloc = scaloc * scal;
                    work[j + (kk + 1) * lds] = smlnum;
                    /* If LATRS overestimated the growth, x may be
                     * rescaled to preserve a valid combined scale factor. */
                    rscal = ONE / scaloc;
                    if (XNRM[kk] * rscal <= bignum) {
                        XNRM[kk] = XNRM[kk] * rscal;
                        cblas_dscal(j2 - j1, rscal, &X[j1 + rhs * ldx], 1);
                        scaloc = ONE;
                    } else {
                        /* The system op(A) * x = b is badly scaled and its
                         * solution cannot be represented as (1/scale) * x.
                         * Set x to zero. */
                        scale[rhs] = ZERO;
                        for (ii = 0; ii < n; ii++) {
                            X[ii + kk * ldx] = ZERO;
                        }
                        /* Discard the local scale factors */
                        for (ii = 0; ii < nba; ii++) {
                            work[ii + (kk + 1) * lds] = ONE;
                        }
                        scaloc = ONE;
                    }
                }
                scaloc = scaloc * work[j + (kk + 1) * lds];
                work[j + (kk + 1) * lds] = scaloc;
            }

            /* Linear block updates */
            if (notran) {
                if (upper) {
                    ifirst = j - 1;
                    ilast = 0;
                    iinc = -1;
                } else {
                    ifirst = j + 1;
                    ilast = nba - 1;
                    iinc = 1;
                }
            } else {
                if (upper) {
                    ifirst = j + 1;
                    ilast = nba - 1;
                    iinc = 1;
                } else {
                    ifirst = j - 1;
                    ilast = 0;
                    iinc = -1;
                }
            }

            for (i = ifirst; (iinc > 0) ? (i <= ilast) : (i >= ilast); i += iinc) {
                /* I1: row index of the first row in X( I, K ) (0-based)
                 * I2: row index past the last row (0-based exclusive) */
                i1 = i * nb;
                i2 = ((i + 1) * nb < n) ? (i + 1) * nb : n;

                /* Prepare the linear update to be executed with GEMM. */
                for (kk = 0; kk < k2 - k1; kk++) {
                    rhs = k1 + kk;
                    /* Compute consistent scaling */
                    scamin = work[i + (kk + 1) * lds];
                    if (work[j + (kk + 1) * lds] < scamin) {
                        scamin = work[j + (kk + 1) * lds];
                    }

                    /* Compute scaling factor to survive the linear update */
                    bnrm = dlange("I", i2 - i1, 1, &X[i1 + rhs * ldx], ldx, W);
                    bnrm = bnrm * (scamin / work[i + (kk + 1) * lds]);
                    XNRM[kk] = XNRM[kk] * (scamin / work[j + (kk + 1) * lds]);
                    anrm = work[awrk + i + j * nba];
                    scaloc = dlarmm(anrm, XNRM[kk], bnrm);

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to B( I, KK ) and B( J, KK ). */
                    scal = (scamin / work[i + (kk + 1) * lds]) * scaloc;
                    if (scal != ONE) {
                        cblas_dscal(i2 - i1, scal, &X[i1 + rhs * ldx], 1);
                        work[i + (kk + 1) * lds] = scamin * scaloc;
                    }

                    scal = (scamin / work[j + (kk + 1) * lds]) * scaloc;
                    if (scal != ONE) {
                        cblas_dscal(j2 - j1, scal, &X[j1 + rhs * ldx], 1);
                        work[j + (kk + 1) * lds] = scamin * scaloc;
                    }
                }

                if (notran) {
                    /* B( I, K ) := B( I, K ) - A( I, J ) * X( J, K ) */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                i2 - i1, k2 - k1, j2 - j1, -ONE,
                                &A[i1 + j1 * lda], lda, &X[j1 + k1 * ldx], ldx,
                                ONE, &X[i1 + k1 * ldx], ldx);
                } else {
                    /* B( I, K ) := B( I, K ) - A( J, I )**T * X( J, K ) */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                i2 - i1, k2 - k1, j2 - j1, -ONE,
                                &A[j1 + i1 * lda], lda, &X[j1 + k1 * ldx], ldx,
                                ONE, &X[i1 + k1 * ldx], ldx);
                }
            }
        }

        /* Reduce local scaling factors */
        for (kk = 0; kk < k2 - k1; kk++) {
            rhs = k1 + kk;
            for (i = 0; i < nba; i++) {
                if (scale[rhs] > work[i + (kk + 1) * lds]) {
                    scale[rhs] = work[i + (kk + 1) * lds];
                }
            }
        }

        /* Realize consistent scaling */
        for (kk = 0; kk < k2 - k1; kk++) {
            rhs = k1 + kk;
            if (scale[rhs] != ONE && scale[rhs] != ZERO) {
                for (i = 0; i < nba; i++) {
                    i1 = i * nb;
                    i2 = ((i + 1) * nb < n) ? (i + 1) * nb : n;
                    scal = scale[rhs] / work[i + (kk + 1) * lds];
                    if (scal != ONE) {
                        cblas_dscal(i2 - i1, scal, &X[i1 + rhs * ldx], 1);
                    }
                }
            }
        }
    }

    work[0] = (f64)lwmin;
}
