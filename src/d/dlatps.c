/**
 * @file dlatps.c
 * @brief DLATPS solves a triangular system with the matrix held in packed storage, with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLATPS solves one of the triangular systems
 *    A * x = s*b  or  A**T * x = s*b
 * with scaling to prevent overflow, where A is an upper or lower
 * triangular matrix stored in packed form. Here A**T denotes the
 * transpose of A, x and b are n-element vectors, and s is a scaling
 * factor, usually less than or equal to 1, chosen so that the
 * components of x will be less than the overflow threshold. If the
 * unscaled problem will not cause overflow, the Level 2 BLAS routine
 * DTPSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
 * then s is set to 0 and a non-trivial solution to A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A * x = s*b; 'T'/'C': Solve A**T * x = s*b.
 * @param[in]     diag    'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     AP      The triangular matrix A, packed columnwise.
 *                        Array of dimension (n*(n+1)/2).
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
void dlatps(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const f64* const restrict AP,
    f64* const restrict X,
    f64* scale,
    f64* const restrict cnorm,
    int* info)
{
    // dlatps.f lines 245-246: Parameters
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;

    // dlatps.f lines 248-252: Local variables
    int upper, notran, nounit;
    int i, imax, ip, j, jfirst, jinc, jlast, jlen;
    f64 bignum, grow, rec, smlnum, sumj, tjj, tjjs = 0.0, tmax, tscal, uscal, xbnd, xj, xmax;

    // dlatps.f lines 268-271
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

    // dlatps.f lines 275-291: Test the input parameters
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't') && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (!(normin[0] == 'Y' || normin[0] == 'y') && !(normin[0] == 'N' || normin[0] == 'n')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("DLATPS", -(*info));
        return;
    }

    // dlatps.f lines 295-296: Quick return if possible
    if (n == 0) {
        return;
    }

    // dlatps.f lines 300-302: Determine machine dependent parameters
    smlnum = dlamch("S") / dlamch("P");
    bignum = ONE / smlnum;
    *scale = ONE;

    // dlatps.f lines 304-328: Compute the 1-norm of each column if needed
    if (normin[0] == 'N' || normin[0] == 'n') {
        if (upper) {
            // dlatps.f lines 312-316: A is upper triangular
            ip = 0;
            for (j = 0; j < n; j++) {
                cnorm[j] = cblas_dasum(j, &AP[ip], 1);
                ip = ip + (j + 1);
            }
        } else {
            // dlatps.f lines 321-326: A is lower triangular
            ip = 0;
            for (j = 0; j < n - 1; j++) {
                cnorm[j] = cblas_dasum(n - j - 1, &AP[ip + 1], 1);
                ip = ip + n - j;
            }
            cnorm[n - 1] = ZERO;
        }
    }

    // dlatps.f lines 333-340: Scale the column norms by TSCAL
    imax = cblas_idamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum) {
        tscal = ONE;
    } else {
        tscal = ONE / (smlnum * tmax);
        cblas_dscal(n, tscal, cnorm, 1);
    }

    // dlatps.f lines 345-347: Compute a bound on the computed solution vector
    j = cblas_idamax(n, X, 1);
    xmax = fabs(X[j]);
    xbnd = xmax;

    if (notran) {
        // dlatps.f lines 352-360: Compute the growth in A * x = b
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
            goto L50;
        }

        if (nounit) {
            // dlatps.f lines 374-403: A is non-unit triangular
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            // dlatps.f line 376: IP = JFIRST*(JFIRST+1)/2
            // In 0-based: ip = jfirst*(jfirst+1)/2 but need diagonal position
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;  // Position of diagonal A(jfirst,jfirst) 0-based
            jlen = n;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L50;
                }
                tjj = fabs(AP[ip]);
                xbnd = (xbnd < ONE ? xbnd : ONE);
                xbnd = (xbnd < tjj ? xbnd : tjj) * grow;
                if (tjj + cnorm[j] >= smlnum) {
                    grow = grow * (tjj / (tjj + cnorm[j]));
                } else {
                    grow = ZERO;
                }
                ip = ip + jinc * jlen;
                jlen = jlen - 1;
            }
            grow = xbnd;
        } else {
            // dlatps.f lines 410-422: A is unit triangular
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            grow = (grow < ONE ? grow : ONE);
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L50;
                }
                grow = grow * (ONE / (ONE + cnorm[j]));
            }
        }
L50:;
    } else {
        // dlatps.f lines 429-437: Compute the growth in A**T * x = b
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
            goto L80;
        }

        if (nounit) {
            // dlatps.f lines 451-475: A is non-unit triangular
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L80;
                }
                xj = ONE + cnorm[j];
                grow = (grow < xbnd / xj ? grow : xbnd / xj);
                tjj = fabs(AP[ip]);
                if (xj > tjj) {
                    xbnd = xbnd * (tjj / xj);
                }
                jlen = jlen + 1;
                ip = ip + jinc * jlen;
            }
            grow = (grow < xbnd ? grow : xbnd);
        } else {
            // dlatps.f lines 482-494: A is unit triangular
            grow = ONE / (xbnd > smlnum ? xbnd : smlnum);
            grow = (grow < ONE ? grow : ONE);
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L80;
                }
                xj = ONE + cnorm[j];
                grow = grow / xj;
            }
        }
L80:;
    }

    if ((grow * tscal) > smlnum) {
        // dlatps.f lines 499-504: Use the Level 2 BLAS solve
        CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = notran ? CblasNoTrans : CblasTrans;
        CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;
        cblas_dtpsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag, n, AP, X, 1);
    } else {
        // dlatps.f lines 509-517: Use a Level 1 BLAS solve, scaling intermediate results
        if (xmax > bignum) {
            *scale = bignum / xmax;
            cblas_dscal(n, *scale, X, 1);
            xmax = bignum;
        }

        if (notran) {
            // dlatps.f lines 523-639: Solve A * x = b
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabs(X[j]);
                if (nounit) {
                    tjjs = AP[ip] * tscal;
                } else {
                    tjjs = tscal;
                    if (tscal == ONE) {
                        goto L100;
                    }
                }
                tjj = fabs(tjjs);
                if (tjj > smlnum) {
                    if (tjj < ONE) {
                        if (xj > tjj * bignum) {
                            rec = ONE / xj;
                            cblas_dscal(n, rec, X, 1);
                            *scale = (*scale) * rec;
                            xmax = xmax * rec;
                        }
                    }
                    X[j] = X[j] / tjjs;
                    xj = fabs(X[j]);
                } else if (tjj > ZERO) {
                    if (xj > tjj * bignum) {
                        rec = (tjj * bignum) / xj;
                        if (cnorm[j] > ONE) {
                            rec = rec / cnorm[j];
                        }
                        cblas_dscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                        xmax = xmax * rec;
                    }
                    X[j] = X[j] / tjjs;
                    xj = fabs(X[j]);
                } else {
                    // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, scale = 0
                    for (i = 0; i < n; i++) {
                        X[i] = ZERO;
                    }
                    X[j] = ONE;
                    xj = ONE;
                    *scale = ZERO;
                    xmax = ZERO;
                }
L100:
                // Scale x if necessary to avoid overflow when adding column j
                if (xj > ONE) {
                    rec = ONE / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        rec = rec * HALF;
                        cblas_dscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                    }
                } else if (xj * cnorm[j] > (bignum - xmax)) {
                    cblas_dscal(n, HALF, X, 1);
                    *scale = (*scale) * HALF;
                }

                if (upper) {
                    if (j > 0) {
                        // x(0:j-1) := x(0:j-1) - x(j) * A(0:j-1,j)
                        cblas_daxpy(j, -X[j] * tscal, &AP[ip - j], 1, X, 1);
                        i = cblas_idamax(j, X, 1);
                        xmax = fabs(X[i]);
                    }
                    ip = ip - (j + 1);
                } else {
                    if (j < n - 1) {
                        // x(j+1:n-1) := x(j+1:n-1) - x(j) * A(j+1:n-1,j)
                        cblas_daxpy(n - j - 1, -X[j] * tscal, &AP[ip + 1], 1, &X[j + 1], 1);
                        i = j + 1 + cblas_idamax(n - j - 1, &X[j + 1], 1);
                        xmax = fabs(X[i]);
                    }
                    ip = ip + n - j;
                }
            }
        } else {
            // dlatps.f lines 645-777: Solve A**T * x = b
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = fabs(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = AP[ip] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = fabs(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE ? rec * tjj : ONE);
                        uscal = uscal / tjjs;
                    }
                    if (rec < ONE) {
                        cblas_dscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                        xmax = xmax * rec;
                    }
                }

                sumj = ZERO;
                if (uscal == ONE) {
                    if (upper) {
                        sumj = cblas_ddot(j, &AP[ip - j], 1, X, 1);
                    } else if (j < n - 1) {
                        sumj = cblas_ddot(n - j - 1, &AP[ip + 1], 1, &X[j + 1], 1);
                    }
                } else {
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            sumj = sumj + (AP[ip - j + i] * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = 1; i <= n - j - 1; i++) {
                            sumj = sumj + (AP[ip + i] * uscal) * X[j + i];
                        }
                    }
                }

                if (uscal == tscal) {
                    X[j] = X[j] - sumj;
                    xj = fabs(X[j]);
                    if (nounit) {
                        tjjs = AP[ip] * tscal;
                    } else {
                        tjjs = tscal;
                        if (tscal == ONE) {
                            goto L150;
                        }
                    }
                    tjj = fabs(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_dscal(n, rec, X, 1);
                                *scale = (*scale) * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = X[j] / tjjs;
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            cblas_dscal(n, rec, X, 1);
                            *scale = (*scale) * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = X[j] / tjjs;
                    } else {
                        // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, scale = 0
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
L150:;
                } else {
                    X[j] = X[j] / tjjs - sumj;
                }
                xmax = (xmax > fabs(X[j]) ? xmax : fabs(X[j]));
                jlen = jlen + 1;
                ip = ip + jinc * jlen;
            }
        }
        *scale = (*scale) / tscal;
    }

    // dlatps.f lines 784-786: Scale the column norms by 1/TSCAL for return
    if (tscal != ONE) {
        cblas_dscal(n, ONE / tscal, cnorm, 1);
    }
}
