/**
 * @file clatrs3.c
 * @brief CLATRS3 solves a triangular system with scale factors to prevent overflow.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"


/**
 * CLATRS3 solves one of the triangular systems
 *
 *    A * X = B * diag(scale),  A**T * X = B * diag(scale), or
 *    A**H * X = B * diag(scale)
 *
 * with scaling to prevent overflow. Here A is an upper or lower
 * triangular matrix, A**T denotes the transpose of A, A**H denotes the
 * conjugate transpose of A. X and B are n by nrhs matrices and scale
 * is an nrhs element vector of scaling factors. A scaling factor scale(j)
 * is usually less than or equal to 1, chosen such that X(:,j) is less
 * than the overflow threshold. If the matrix A is singular (A(j,j) = 0
 * for some j), then a non-trivial solution to A*X = 0 is returned. If
 * the system is so badly scaled that the solution cannot be represented as
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
 *                        'T': Solve A**T * X = B*diag(scale);
 *                        'C': Solve A**H * X = B*diag(scale).
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
void clatrs3(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const INT n,
    const INT nrhs,
    const c64* restrict A,
    const INT lda,
    c64* restrict X,
    const INT ldx,
    f32* restrict scale,
    f32* restrict cnorm,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    /* Parameters from Fortran - match LAPACK exactly */
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
#define NRHSMIN 2
#define NBRHS 32
#define NBMIN 8
#define NBMAX 64

    /* Local arrays - match Fortran local arrays */
    f32 W[NBMAX];
    f32 XNRM[NBRHS];

    /* Local scalars */
    INT upper, notran, nounit, lquery;
    INT awrk, i, ifirst, iinc, ilast, ii, i1, i2, j;
    INT jfirst, jinc, jlast, j1, j2, k, kk, k1, k2;
    INT lanrm, lds, lscale, nb, nba, nbx, rhs, lwmin;
    f32 anrm, bignum, bnrm, rscal, scal, scaloc;
    f32 scamin, smlnum, tmax;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    lquery = (lwork == -1);

    /* Partition A and X into blocks */
    nb = NBMAX;
    if (nb < NBMIN) nb = NBMIN;
    if (nb > NBMAX) nb = NBMAX;

    nba = (n + nb - 1) / nb;
    if (nba < 1) nba = 1;
    nbx = (nrhs + NBRHS - 1) / NBRHS;
    if (nbx < 1) nbx = 1;

    /* Compute the workspace */
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
    work[0] = (f32)lwmin;

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
        xerbla("CLATRS3", -(*info));
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
    bignum = slamch("O");
    smlnum = slamch("S");

    /* Use unblocked code for small problems */
    if (nrhs < NRHSMIN) {
        clatrs(uplo, trans, diag, normin, n, A, lda, &X[0], &scale[0], cnorm, info);
        for (k = 1; k < nrhs; k++) {
            clatrs(uplo, trans, diag, "Y", n, A, lda, &X[k * ldx], &scale[k], cnorm, info);
        }
        return;
    }

    /* Compute norms of blocks of A excluding diagonal blocks and find
     * the block with the largest norm TMAX. */
    tmax = ZERO;
    for (j = 0; j < nba; j++) {
        j1 = j * nb;
        j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

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
                anrm = clange("I", i2 - i1, j2 - j1, &A[i1 + j1 * lda], lda, W);
                work[awrk + i + j * nba] = anrm;
            } else {
                anrm = clange("1", i2 - i1, j2 - j1, &A[i1 + j1 * lda], lda, W);
                work[awrk + j + i * nba] = anrm;
            }
            if (tmax < anrm) {
                tmax = anrm;
            }
        }
    }

    if (!(tmax <= slamch("O"))) {
        /* Some matrix entries have huge absolute value. Fall back to LATRS. */
        for (k = 0; k < nrhs; k++) {
            clatrs(uplo, trans, diag, "N", n, A, lda, &X[k * ldx], &scale[k], cnorm, info);
        }
        return;
    }

    /* Loop over block columns of X */
    for (k = 0; k < nbx; k++) {
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
            /* Solve op(A) * X(:, K1:K2-1) = B * diag(scale(K1:K2-1)) */
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
            j1 = j * nb;
            j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

            /* Solve op(A( J, J )) * X( J, RHS ) = SCALOC * B( J, RHS ) */
            for (kk = 0; kk < k2 - k1; kk++) {
                rhs = k1 + kk;
                if (kk == 0) {
                    clatrs(uplo, trans, diag, "N", j2 - j1,
                           &A[j1 + j1 * lda], lda, &X[j1 + rhs * ldx],
                           &scaloc, cnorm, info);
                } else {
                    clatrs(uplo, trans, diag, "Y", j2 - j1,
                           &A[j1 + j1 * lda], lda, &X[j1 + rhs * ldx],
                           &scaloc, cnorm, info);
                }

                /* Find largest absolute value entry in X( J1:J2-1, RHS ) */
                XNRM[kk] = clange("I", j2 - j1, 1, &X[j1 + rhs * ldx], ldx, W);

                if (scaloc == ZERO) {
                    /* LATRS found that A is singular through A(j,j) = 0. */
                    scale[rhs] = ZERO;
                    for (ii = 0; ii < j1; ii++) {
                        X[ii + kk * ldx] = CZERO;
                    }
                    for (ii = j2; ii < n; ii++) {
                        X[ii + kk * ldx] = CZERO;
                    }
                    /* Discard the local scale factors */
                    for (ii = 0; ii < nba; ii++) {
                        work[ii + (kk + 1) * lds] = ONE;
                    }
                    scaloc = ONE;
                } else if (scaloc * work[j + (kk + 1) * lds] == ZERO) {
                    scal = work[j + (kk + 1) * lds] / smlnum;
                    scaloc = scaloc * scal;
                    work[j + (kk + 1) * lds] = smlnum;
                    rscal = ONE / scaloc;
                    if (XNRM[kk] * rscal <= bignum) {
                        XNRM[kk] = XNRM[kk] * rscal;
                        cblas_csscal(j2 - j1, rscal, &X[j1 + rhs * ldx], 1);
                        scaloc = ONE;
                    } else {
                        scale[rhs] = ZERO;
                        for (ii = 0; ii < n; ii++) {
                            X[ii + kk * ldx] = CZERO;
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
                    bnrm = clange("I", i2 - i1, 1, &X[i1 + rhs * ldx], ldx, W);
                    bnrm = bnrm * (scamin / work[i + (kk + 1) * lds]);
                    XNRM[kk] = XNRM[kk] * (scamin / work[j + (kk + 1) * lds]);
                    anrm = work[awrk + i + j * nba];
                    scaloc = slarmm(anrm, XNRM[kk], bnrm);

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to B( I, KK ) and B( J, KK ). */
                    scal = (scamin / work[i + (kk + 1) * lds]) * scaloc;
                    if (scal != ONE) {
                        cblas_csscal(i2 - i1, scal, &X[i1 + rhs * ldx], 1);
                        work[i + (kk + 1) * lds] = scamin * scaloc;
                    }

                    scal = (scamin / work[j + (kk + 1) * lds]) * scaloc;
                    if (scal != ONE) {
                        cblas_csscal(j2 - j1, scal, &X[j1 + rhs * ldx], 1);
                        work[j + (kk + 1) * lds] = scamin * scaloc;
                    }
                }

                if (notran) {
                    /* B( I, K ) := B( I, K ) - A( I, J ) * X( J, K ) */
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                i2 - i1, k2 - k1, j2 - j1, &NEG_CONE,
                                &A[i1 + j1 * lda], lda, &X[j1 + k1 * ldx], ldx,
                                &CONE, &X[i1 + k1 * ldx], ldx);
                } else if (trans[0] == 'T' || trans[0] == 't') {
                    /* B( I, K ) := B( I, K ) - A( J, I )**T * X( J, K ) */
                    cblas_cgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                i2 - i1, k2 - k1, j2 - j1, &NEG_CONE,
                                &A[j1 + i1 * lda], lda, &X[j1 + k1 * ldx], ldx,
                                &CONE, &X[i1 + k1 * ldx], ldx);
                } else {
                    /* B( I, K ) := B( I, K ) - A( J, I )**H * X( J, K ) */
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                i2 - i1, k2 - k1, j2 - j1, &NEG_CONE,
                                &A[j1 + i1 * lda], lda, &X[j1 + k1 * ldx], ldx,
                                &CONE, &X[i1 + k1 * ldx], ldx);
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
                        cblas_csscal(i2 - i1, scal, &X[i1 + rhs * ldx], 1);
                    }
                }
            }
        }
    }

    work[0] = (f32)lwmin;
}
