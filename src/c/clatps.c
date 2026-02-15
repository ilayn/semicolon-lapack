/**
 * @file clatps.c
 * @brief CLATPS solves a triangular system with the matrix held in packed storage, with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLATPS solves one of the triangular systems
 *    A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b
 * with scaling to prevent overflow, where A is an upper or lower
 * triangular matrix stored in packed form. Here A**T denotes the
 * transpose of A, A**H denotes the conjugate transpose of A, x and b
 * are n-element vectors, and s is a scaling factor, usually less than
 * or equal to 1, chosen so that the components of x will be less than
 * the overflow threshold. If the unscaled problem will not cause
 * overflow, the Level 2 BLAS routine ZTPSV is called. If the matrix A
 * is singular (A(j,j) = 0 for some j), then s is set to 0 and a
 * non-trivial solution to A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A * x = s*b; 'T': Solve A**T * x = s*b;
 *                        'C': Solve A**H * x = s*b.
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
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -k, the k-th argument had an illegal value
 */
void clatps(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const c64* restrict AP,
    c64* restrict X,
    f32* scale,
    f32* restrict cnorm,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 HALF = 0.5f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    int upper, notran, nounit;
    int i, imax, ip, j, jfirst, jinc, jlast, jlen;
    f32 bignum, grow, rec, smlnum, tjj, tmax, tscal, xbnd, xj, xmax;
    c64 csumj, tjjs = 0.0f, uscal;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

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
        xerbla("CLATPS", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    smlnum = FLT_MIN / FLT_EPSILON;
    bignum = ONE / smlnum;
    *scale = ONE;

    if (normin[0] == 'N' || normin[0] == 'n') {
        if (upper) {
            ip = 0;
            for (j = 0; j < n; j++) {
                cnorm[j] = cblas_scasum(j, &AP[ip], 1);
                ip = ip + (j + 1);
            }
        } else {
            ip = 0;
            for (j = 0; j < n - 1; j++) {
                cnorm[j] = cblas_scasum(n - j - 1, &AP[ip + 1], 1);
                ip = ip + n - j;
            }
            cnorm[n - 1] = ZERO;
        }
    }

    imax = cblas_isamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum * HALF) {
        tscal = ONE;
    } else {
        tscal = HALF / (smlnum * tmax);
        cblas_sscal(n, tscal, cnorm, 1);
    }

    xmax = ZERO;
    for (j = 0; j < n; j++) {
        f32 tmp = cabs2f(X[j]);
        if (xmax < tmp) xmax = tmp;
    }
    xbnd = xmax;

    if (notran) {

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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = n;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L50;
                }
                tjj = cabs1f(AP[ip]);
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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) {
                    goto L80;
                }
                xj = ONE + cnorm[j];
                grow = (grow < xbnd / xj ? grow : xbnd / xj);
                tjj = cabs1f(AP[ip]);
                if (xj > tjj) {
                    xbnd = xbnd * (tjj / xj);
                }
                jlen = jlen + 1;
                ip = ip + jinc * jlen;
            }
            grow = (grow < xbnd ? grow : xbnd);
        } else {
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
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
        CBLAS_TRANSPOSE cblas_trans;
        if (notran) {
            cblas_trans = CblasNoTrans;
        } else if (trans[0] == 'T' || trans[0] == 't') {
            cblas_trans = CblasTrans;
        } else {
            cblas_trans = CblasConjTrans;
        }
        cblas_ctpsv(CblasColMajor,
                    upper ? CblasUpper : CblasLower,
                    cblas_trans,
                    nounit ? CblasNonUnit : CblasUnit,
                    n, AP, X, 1);
    } else {

        if (xmax > bignum * HALF) {
            *scale = (bignum * HALF) / xmax;
            cblas_csscal(n, *scale, X, 1);
            xmax = bignum;
        } else {
            xmax = xmax * TWO;
        }

        if (notran) {
            /* Solve A * x = b */
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                if (nounit) {
                    tjjs = AP[ip] * tscal;
                } else {
                    tjjs = tscal;
                    if (tscal == ONE) {
                        goto L100;
                    }
                }
                tjj = cabs1f(tjjs);
                if (tjj > smlnum) {
                    if (tjj < ONE) {
                        if (xj > tjj * bignum) {
                            rec = ONE / xj;
                            cblas_csscal(n, rec, X, 1);
                            *scale = (*scale) * rec;
                            xmax = xmax * rec;
                        }
                    }
                    X[j] = cladiv(X[j], tjjs);
                    xj = cabs1f(X[j]);
                } else if (tjj > ZERO) {
                    if (xj > tjj * bignum) {
                        rec = (tjj * bignum) / xj;
                        if (cnorm[j] > ONE) {
                            rec = rec / cnorm[j];
                        }
                        cblas_csscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                        xmax = xmax * rec;
                    }
                    X[j] = cladiv(X[j], tjjs);
                    xj = cabs1f(X[j]);
                } else {
                    for (i = 0; i < n; i++) {
                        X[i] = ZERO;
                    }
                    X[j] = ONE;
                    xj = ONE;
                    *scale = ZERO;
                    xmax = ZERO;
                }
L100:
                if (xj > ONE) {
                    rec = ONE / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        rec = rec * HALF;
                        cblas_csscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                    }
                } else if (xj * cnorm[j] > (bignum - xmax)) {
                    cblas_csscal(n, HALF, X, 1);
                    *scale = (*scale) * HALF;
                }

                if (upper) {
                    if (j > 0) {
                        c64 neg_xj_tscal = -X[j] * tscal;
                        cblas_caxpy(j, &neg_xj_tscal, &AP[ip - j], 1, X, 1);
                        i = cblas_icamax(j, X, 1);
                        xmax = cabs1f(X[i]);
                    }
                    ip = ip - (j + 1);
                } else {
                    if (j < n - 1) {
                        c64 neg_xj_tscal = -X[j] * tscal;
                        cblas_caxpy(n - j - 1, &neg_xj_tscal, &AP[ip + 1], 1, &X[j + 1], 1);
                        i = j + 1 + cblas_icamax(n - j - 1, &X[j + 1], 1);
                        xmax = cabs1f(X[i]);
                    }
                    ip = ip + n - j;
                }
            }

        } else if (trans[0] == 'T' || trans[0] == 't') {
            /* Solve A**T * x = b */
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = AP[ip] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE ? rec * tjj : ONE);
                        uscal = cladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_csscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (uscal == (c64)ONE) {
                    if (upper) {
                        if (j > 0) {
                            cblas_cdotu_sub(j, &AP[ip - j], 1, X, 1, &csumj);
                        }
                    } else if (j < n - 1) {
                        cblas_cdotu_sub(n - j - 1, &AP[ip + 1], 1, &X[j + 1], 1, &csumj);
                    }
                } else {
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            csumj = csumj + (AP[ip - j + i] * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = 1; i <= n - j - 1; i++) {
                            csumj = csumj + (AP[ip + i] * uscal) * X[j + i];
                        }
                    }
                }

                if (uscal == (c64)tscal) {
                    X[j] = X[j] - csumj;
                    xj = cabs1f(X[j]);
                    if (nounit) {
                        tjjs = AP[ip] * tscal;
                    } else {
                        tjjs = tscal;
                        if (tscal == ONE) {
                            goto L150;
                        }
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_csscal(n, rec, X, 1);
                                *scale = (*scale) * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = cladiv(X[j], tjjs);
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            cblas_csscal(n, rec, X, 1);
                            *scale = (*scale) * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = cladiv(X[j], tjjs);
                    } else {
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
L150:;
                } else {
                    X[j] = cladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1f(X[j]) ? xmax : cabs1f(X[j]));
                jlen = jlen + 1;
                ip = ip + jinc * jlen;
            }

        } else {
            /* Solve A**H * x = b */
            ip = (jfirst + 1) * (jfirst + 2) / 2 - 1;
            jlen = 1;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = conjf(AP[ip]) * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE ? rec * tjj : ONE);
                        uscal = cladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_csscal(n, rec, X, 1);
                        *scale = (*scale) * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (uscal == (c64)ONE) {
                    if (upper) {
                        if (j > 0) {
                            cblas_cdotc_sub(j, &AP[ip - j], 1, X, 1, &csumj);
                        }
                    } else if (j < n - 1) {
                        cblas_cdotc_sub(n - j - 1, &AP[ip + 1], 1, &X[j + 1], 1, &csumj);
                    }
                } else {
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            csumj = csumj + (conjf(AP[ip - j + i]) * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = 1; i <= n - j - 1; i++) {
                            csumj = csumj + (conjf(AP[ip + i]) * uscal) * X[j + i];
                        }
                    }
                }

                if (uscal == (c64)tscal) {
                    X[j] = X[j] - csumj;
                    xj = cabs1f(X[j]);
                    if (nounit) {
                        tjjs = conjf(AP[ip]) * tscal;
                    } else {
                        tjjs = tscal;
                        if (tscal == ONE) {
                            goto L210;
                        }
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_csscal(n, rec, X, 1);
                                *scale = (*scale) * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = cladiv(X[j], tjjs);
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            cblas_csscal(n, rec, X, 1);
                            *scale = (*scale) * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = cladiv(X[j], tjjs);
                    } else {
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
L210:;
                } else {
                    X[j] = cladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1f(X[j]) ? xmax : cabs1f(X[j]));
                jlen = jlen + 1;
                ip = ip + jinc * jlen;
            }
        }
        *scale = (*scale) / tscal;
    }

    if (tscal != ONE) {
        cblas_sscal(n, ONE / tscal, cnorm, 1);
    }
}
