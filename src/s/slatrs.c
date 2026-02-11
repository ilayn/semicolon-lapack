/**
 * @file slatrs.c
 * @brief Solves a triangular system with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLATRS solves one of the triangular systems
 *    A * x = s*b  or  A**T * x = s*b
 * with scaling to prevent overflow. Here A is an upper or lower
 * triangular matrix, A**T denotes the transpose of A, x and b are
 * n-element vectors, and s is a scaling factor, usually less than
 * or equal to 1, chosen so that the components of x will be less than
 * the overflow threshold. If the unscaled problem will not cause
 * overflow, the Level 2 BLAS routine DTRSV is called. If the matrix A
 * is singular (A(j,j) = 0 for some j), then s is set to 0 and a
 * non-trivial solution to A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A * x = s*b; 'T'/'C': Solve A**T * x = s*b.
 * @param[in]     diag    'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     A       The triangular matrix A. Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of A (lda >= max(1,n)).
 * @param[in,out] X       On entry, the right hand side b.
 *                        On exit, overwritten by the solution x.
 *                        Array of dimension n.
 * @param[out]    scale   The scaling factor s for the triangular system.
 * @param[in,out] cnorm   If normin='Y', cnorm contains column norms on entry.
 *                        If normin='N', cnorm returns the 1-norm of offdiagonal
 *                        part of column j. Array of dimension n.
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void slatrs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const float * const restrict A,
    const int lda,
    float * const restrict X,
    float *scale,
    float * const restrict cnorm,
    int *info)
{
    const float ZERO = 0.0f;
    const float HALF = 0.5f;
    const float ONE = 1.0f;

    int upper, notran, nounit, normin_n;
    int i, imax, j, jfirst, jinc, jlast;
    float bignum, grow, rec, smlnum, sumj, tjj, tjjs = 0.0f, tmax, tscal, uscal, xbnd, xj, xmax;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    normin_n = (normin[0] == 'N' || normin[0] == 'n');

    // Test the input parameters
    if (!upper && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -1;
    } else if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -2;
    } else if (!nounit && diag[0] != 'U' && diag[0] != 'u') {
        *info = -3;
    } else if (!normin_n && normin[0] != 'Y' && normin[0] != 'y') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("SLATRS", -(*info));
        return;
    }

    // Quick return if possible
    *scale = ONE;
    if (n == 0) {
        return;
    }

    // Determine machine dependent parameters to control overflow
    smlnum = FLT_MIN / FLT_EPSILON;
    bignum = ONE / smlnum;

    if (normin_n) {
        // Compute the 1-norm of each column, not including the diagonal
        if (upper) {
            for (j = 0; j < n; j++) {
                cnorm[j] = cblas_sasum(j, &A[j * lda], 1);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                cnorm[j] = cblas_sasum(n - j - 1, &A[j + 1 + j * lda], 1);
            }
            if (n > 0) cnorm[n - 1] = ZERO;
        }
    }

    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM
    imax = cblas_isamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum) {
        tscal = ONE;
    } else {
        // Avoid NaN generation if entries in CNORM exceed the overflow threshold
        if (tmax <= FLT_MAX) {
            tscal = ONE / (smlnum * tmax);
            cblas_sscal(n, tscal, cnorm, 1);
        } else {
            // At least one entry of A is not a valid floating-point entry.
            // Rely on TRSV to propagate Inf and NaN.
            if (upper) {
                cblas_strsv(CblasColMajor, CblasUpper,
                            notran ? CblasNoTrans : CblasTrans,
                            nounit ? CblasNonUnit : CblasUnit,
                            n, A, lda, X, 1);
            } else {
                cblas_strsv(CblasColMajor, CblasLower,
                            notran ? CblasNoTrans : CblasTrans,
                            nounit ? CblasNonUnit : CblasUnit,
                            n, A, lda, X, 1);
            }
            return;
        }
    }

    // Compute a bound on the computed solution vector to see if the
    // Level 2 BLAS routine DTRSV can be used
    j = cblas_isamax(n, X, 1);
    xmax = fabsf(X[j]);
    xbnd = xmax;

    if (notran) {
        // Compute the growth in A * x = b
        if (upper) {
            jfirst = n - 1;
            jlast = 0;
            jinc = -1;
        } else {
            jfirst = 0;
            jlast = n - 1;
            jinc = 1;
        }

        if (tscal != ONE) {
            grow = ZERO;
        } else if (nounit) {
            // A is non-unit triangular
            // Compute GROW = 1/G(j) and XBND = 1/M(j)
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                tjj = fabsf(A[j + j * lda]);
                xbnd = (xbnd < (ONE < tjj ? ONE : tjj) * grow) ? xbnd : (ONE < tjj ? ONE : tjj) * grow;
                if (tjj + cnorm[j] >= smlnum) {
                    grow = grow * (tjj / (tjj + cnorm[j]));
                } else {
                    grow = ZERO;
                }
            }
            if (grow > smlnum) {
                grow = xbnd;
            }
        } else {
            // A is unit triangular
            grow = (ONE < ONE / (xbnd > smlnum ? xbnd : smlnum)) ? ONE / (xbnd > smlnum ? xbnd : smlnum) : ONE;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                grow = grow * (ONE / (ONE + cnorm[j]));
            }
        }
    } else {
        // Compute the growth in A**T * x = b
        if (upper) {
            jfirst = 0;
            jlast = n - 1;
            jinc = 1;
        } else {
            jfirst = n - 1;
            jlast = 0;
            jinc = -1;
        }

        if (tscal != ONE) {
            grow = ZERO;
        } else if (nounit) {
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = (grow < xbnd / xj) ? grow : xbnd / xj;
                tjj = fabsf(A[j + j * lda]);
                if (xj > tjj) {
                    xbnd = xbnd * (tjj / xj);
                }
            }
            if (grow > smlnum) {
                grow = (grow < xbnd) ? grow : xbnd;
            }
        } else {
            grow = (ONE < ONE / (xbnd > smlnum ? xbnd : smlnum)) ? ONE / (xbnd > smlnum ? xbnd : smlnum) : ONE;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = grow / xj;
            }
        }
    }

    if ((grow * tscal) > smlnum) {
        // Use the Level 2 BLAS solve if the reciprocal of the bound on
        // elements of X is not too small
        if (upper) {
            cblas_strsv(CblasColMajor, CblasUpper,
                        notran ? CblasNoTrans : CblasTrans,
                        nounit ? CblasNonUnit : CblasUnit,
                        n, A, lda, X, 1);
        } else {
            cblas_strsv(CblasColMajor, CblasLower,
                        notran ? CblasNoTrans : CblasTrans,
                        nounit ? CblasNonUnit : CblasUnit,
                        n, A, lda, X, 1);
        }
    } else {
        // Use a Level 1 BLAS solve, scaling intermediate results

        if (xmax > bignum) {
            *scale = bignum / xmax;
            cblas_sscal(n, *scale, X, 1);
            xmax = bignum;
        }

        if (notran) {
            // Solve A * x = b
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabsf(X[j]);
                if (nounit) {
                    tjjs = A[j + j * lda] * tscal;
                } else {
                    tjjs = tscal;
                }
                if (nounit || tscal != ONE) {
                    tjj = fabsf(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_sscal(n, rec, X, 1);
                                *scale = *scale * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = X[j] / tjjs;
                        xj = fabsf(X[j]);
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            if (cnorm[j] > ONE) {
                                rec = rec / cnorm[j];
                            }
                            cblas_sscal(n, rec, X, 1);
                            *scale = *scale * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = X[j] / tjjs;
                        xj = fabsf(X[j]);
                    } else {
                        // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        xj = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
                }
                // Scale x if necessary to avoid overflow when adding a
                // multiple of column j of A
                if (xj > ONE) {
                    rec = ONE / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        rec = rec * HALF;
                        cblas_sscal(n, rec, X, 1);
                        *scale = *scale * rec;
                    }
                } else if (xj * cnorm[j] > (bignum - xmax)) {
                    cblas_sscal(n, HALF, X, 1);
                    *scale = *scale * HALF;
                }

                if (upper) {
                    if (j > 0) {
                        cblas_saxpy(j, -X[j] * tscal, &A[j * lda], 1, X, 1);
                        i = cblas_isamax(j, X, 1);
                        xmax = fabsf(X[i]);
                    }
                } else {
                    if (j < n - 1) {
                        cblas_saxpy(n - j - 1, -X[j] * tscal, &A[j + 1 + j * lda], 1, &X[j + 1], 1);
                        i = j + 1 + cblas_isamax(n - j - 1, &X[j + 1], 1);
                        xmax = fabsf(X[i]);
                    }
                }
            }
        } else {
            // Solve A**T * x = b
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabsf(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = A[j + j * lda] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = fabsf(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE) ? rec * tjj : ONE;
                        uscal = uscal / tjjs;
                    }
                    if (rec < ONE) {
                        cblas_sscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                }

                sumj = ZERO;
                if (uscal == ONE) {
                    if (upper) {
                        sumj = cblas_sdot(j, &A[j * lda], 1, X, 1);
                    } else if (j < n - 1) {
                        sumj = cblas_sdot(n - j - 1, &A[j + 1 + j * lda], 1, &X[j + 1], 1);
                    }
                } else {
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            sumj = sumj + (A[i + j * lda] * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = j + 1; i < n; i++) {
                            sumj = sumj + (A[i + j * lda] * uscal) * X[i];
                        }
                    }
                }

                if (uscal == tscal) {
                    X[j] = X[j] - sumj;
                    xj = fabsf(X[j]);
                    if (nounit) {
                        tjjs = A[j + j * lda] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    if (nounit || tscal != ONE) {
                        tjj = fabsf(tjjs);
                        if (tjj > smlnum) {
                            if (tjj < ONE) {
                                if (xj > tjj * bignum) {
                                    rec = ONE / xj;
                                    cblas_sscal(n, rec, X, 1);
                                    *scale = *scale * rec;
                                    xmax = xmax * rec;
                                }
                            }
                            X[j] = X[j] / tjjs;
                        } else if (tjj > ZERO) {
                            if (xj > tjj * bignum) {
                                rec = (tjj * bignum) / xj;
                                cblas_sscal(n, rec, X, 1);
                                *scale = *scale * rec;
                                xmax = xmax * rec;
                            }
                            X[j] = X[j] / tjjs;
                        } else {
                            for (i = 0; i < n; i++) {
                                X[i] = ZERO;
                            }
                            X[j] = ONE;
                            *scale = ZERO;
                            xmax = ZERO;
                        }
                    }
                } else {
                    X[j] = X[j] / tjjs - sumj;
                }
                xmax = (xmax > fabsf(X[j])) ? xmax : fabsf(X[j]);
            }
        }
        *scale = *scale / tscal;
    }

    // Scale the column norms by 1/TSCAL for return
    if (tscal != ONE) {
        cblas_sscal(n, ONE / tscal, cnorm, 1);
    }
}
