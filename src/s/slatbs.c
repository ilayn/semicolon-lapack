/**
 * @file slatbs.c
 * @brief Solves a triangular banded system with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLATBS solves one of the triangular systems
 *    A * x = s*b  or  A**T * x = s*b
 * with scaling to prevent overflow, where A is an upper or lower
 * triangular band matrix. Here A**T denotes the transpose of A, x and b
 * are n-element vectors, and s is a scaling factor, usually less than
 * or equal to 1, chosen so that the components of x will be less than
 * the overflow threshold. If the unscaled problem will not cause
 * overflow, the Level 2 BLAS routine DTBSV is called. If the matrix A
 * is singular (A(j,j) = 0 for some j), then s is set to 0 and a
 * non-trivial solution to A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A * x = s*b; 'T'/'C': Solve A**T * x = s*b.
 * @param[in]     diag    'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     kd      The number of subdiagonals or superdiagonals in A (kd >= 0).
 * @param[in]     AB      The triangular band matrix A, stored in the first kd+1 rows.
 *                        Array of dimension (ldab, n).
 *                        If uplo = "U", AB[kd+i-j + j*ldab] = A(i,j) for max(0,j-kd)<=i<=j.
 *                        If uplo = "L", AB[i-j + j*ldab] = A(i,j) for j<=i<=min(n-1,j+kd).
 * @param[in]     ldab    The leading dimension of AB (ldab >= kd+1).
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
void slatbs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const int kd,
    const f32* restrict AB,
    const int ldab,
    f32* restrict X,
    f32* scale,
    f32* restrict cnorm,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 HALF = 0.5f;
    const f32 ONE = 1.0f;

    int upper, notran, nounit, normin_n;
    int i, imax, j, jfirst, jinc, jlast, jlen, maind;
    f32 bignum, grow, rec, smlnum, sumj, tjj, tjjs = 0.0f, tmax, tscal, uscal, xbnd, xj, xmax;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    normin_n = (normin[0] == 'N' || normin[0] == 'n');

    /* Test the input parameters */
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
    } else if (kd < 0) {
        *info = -6;
    } else if (ldab < kd + 1) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("SLATBS", -(*info));
        return;
    }

    /* Quick return if possible */
    *scale = ONE;
    if (n == 0) {
        return;
    }

    /* Determine machine dependent parameters to control overflow */
    smlnum = FLT_MIN / FLT_EPSILON;
    bignum = ONE / smlnum;

    if (normin_n) {
        /* Compute the 1-norm of each column, not including the diagonal */
        if (upper) {
            /* A is upper triangular */
            for (j = 0; j < n; j++) {
                /* 0-based: jlen = min(kd, j) elements above diagonal */
                jlen = (kd < j) ? kd : j;
                if (jlen > 0) {
                    cnorm[j] = cblas_sasum(jlen, &AB[kd - jlen + j * ldab], 1);
                } else {
                    cnorm[j] = ZERO;
                }
            }
        } else {
            /* A is lower triangular */
            for (j = 0; j < n; j++) {
                /* 0-based: jlen = min(kd, n-1-j) elements below diagonal */
                jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                if (jlen > 0) {
                    cnorm[j] = cblas_sasum(jlen, &AB[1 + j * ldab], 1);
                } else {
                    cnorm[j] = ZERO;
                }
            }
        }
    }

    /* Scale the column norms by TSCAL if the maximum element in CNORM is
       greater than BIGNUM */
    imax = cblas_isamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum) {
        tscal = ONE;
    } else {
        tscal = ONE / (smlnum * tmax);
        cblas_sscal(n, tscal, cnorm, 1);
    }

    /* Compute a bound on the computed solution vector to see if the
       Level 2 BLAS routine DTBSV can be used */
    j = cblas_isamax(n, X, 1);
    xmax = fabsf(X[j]);
    xbnd = xmax;

    if (notran) {
        /* Compute the growth in A * x = b */
        if (upper) {
            jfirst = n - 1;
            jlast = 0;
            jinc = -1;
            maind = kd;  /* 0-based: diagonal at row kd for upper */
        } else {
            jfirst = 0;
            jlast = n - 1;
            jinc = 1;
            maind = 0;   /* 0-based: diagonal at row 0 for lower */
        }

        if (tscal != ONE) {
            grow = ZERO;
        } else if (nounit) {
            /* A is non-unit triangular */
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                tjj = fabsf(AB[maind + j * ldab]);
                xbnd = (xbnd < (ONE < tjj ? ONE : tjj) * grow) ? xbnd : (ONE < tjj ? ONE : tjj) * grow;
                if (tjj + cnorm[j] >= smlnum) {
                    grow = grow * (tjj / (tjj + cnorm[j]));
                } else {
                    grow = ZERO;
                }
            }
            grow = xbnd;
        } else {
            /* A is unit triangular */
            f32 denom = (xbnd > smlnum) ? xbnd : smlnum;
            grow = (ONE < ONE / denom) ? ONE : ONE / denom;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                grow = grow * (ONE / (ONE + cnorm[j]));
            }
        }
    } else {
        /* Compute the growth in A**T * x = b */
        if (upper) {
            jfirst = 0;
            jlast = n - 1;
            jinc = 1;
            maind = kd;
        } else {
            jfirst = n - 1;
            jlast = 0;
            jinc = -1;
            maind = 0;
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
                tjj = fabsf(AB[maind + j * ldab]);
                if (xj > tjj) {
                    xbnd = xbnd * (tjj / xj);
                }
            }
            grow = (grow < xbnd) ? grow : xbnd;
        } else {
            f32 denom = (xbnd > smlnum) ? xbnd : smlnum;
            grow = (ONE < ONE / denom) ? ONE : ONE / denom;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = grow / xj;
            }
        }
    }

    if ((grow * tscal) > smlnum) {
        /* Use the Level 2 BLAS solve if the reciprocal of the bound on
           elements of X is not too small */
        cblas_stbsv(CblasColMajor,
                    upper ? CblasUpper : CblasLower,
                    notran ? CblasNoTrans : CblasTrans,
                    nounit ? CblasNonUnit : CblasUnit,
                    n, kd, AB, ldab, X, 1);
    } else {
        /* Use a Level 1 BLAS solve, scaling intermediate results */

        if (xmax > bignum) {
            *scale = bignum / xmax;
            cblas_sscal(n, *scale, X, 1);
            xmax = bignum;
        }

        if (notran) {
            /* Solve A * x = b */
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabsf(X[j]);
                if (nounit) {
                    tjjs = AB[maind + j * ldab] * tscal;
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
                        /* A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0 */
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        xj = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
                }
                /* Scale x if necessary to avoid overflow when adding a
                   multiple of column j of A */
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
                        /* Compute the update
                           x(max(0,j-kd):j-1) := x(max(0,j-kd):j-1) - x(j) * A(max(0,j-kd):j-1,j) */
                        jlen = (kd < j) ? kd : j;
                        cblas_saxpy(jlen, -X[j] * tscal, &AB[kd - jlen + j * ldab], 1, &X[j - jlen], 1);
                        i = cblas_isamax(j, X, 1);
                        xmax = fabsf(X[i]);
                    }
                } else {
                    if (j < n - 1) {
                        /* Compute the update
                           x(j+1:min(j+kd,n-1)) := x(j+1:min(j+kd,n-1)) - x(j) * A(j+1:min(j+kd,n-1),j) */
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        if (jlen > 0) {
                            cblas_saxpy(jlen, -X[j] * tscal, &AB[1 + j * ldab], 1, &X[j + 1], 1);
                        }
                        i = j + 1 + cblas_isamax(n - j - 1, &X[j + 1], 1);
                        xmax = fabsf(X[i]);
                    }
                }
            }
        } else {
            /* Solve A**T * x = b */
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabsf(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = AB[maind + j * ldab] * tscal;
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
                        jlen = (kd < j) ? kd : j;
                        if (jlen > 0) {
                            sumj = cblas_sdot(jlen, &AB[kd - jlen + j * ldab], 1, &X[j - jlen], 1);
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        if (jlen > 0) {
                            sumj = cblas_sdot(jlen, &AB[1 + j * ldab], 1, &X[j + 1], 1);
                        }
                    }
                } else {
                    if (upper) {
                        jlen = (kd < j) ? kd : j;
                        for (i = 0; i < jlen; i++) {
                            sumj = sumj + (AB[kd - jlen + i + j * ldab] * uscal) * X[j - jlen + i];
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        for (i = 0; i < jlen; i++) {
                            sumj = sumj + (AB[1 + i + j * ldab] * uscal) * X[j + 1 + i];
                        }
                    }
                }

                if (uscal == tscal) {
                    X[j] = X[j] - sumj;
                    xj = fabsf(X[j]);
                    if (nounit) {
                        tjjs = AB[maind + j * ldab] * tscal;
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

    /* Scale the column norms by 1/TSCAL for return */
    if (tscal != ONE) {
        cblas_sscal(n, ONE / tscal, cnorm, 1);
    }
}
