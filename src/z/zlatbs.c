/**
 * @file zlatbs.c
 * @brief Solves a triangular banded system with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLATBS solves one of the triangular systems
 *    A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b
 * with scaling to prevent overflow, where A is an upper or lower
 * triangular band matrix. Here A**T denotes the transpose of A, A**H
 * denotes the conjugate transpose of A, x and b are n-element vectors,
 * and s is a scaling factor, usually less than or equal to 1, chosen so
 * that the components of x will be less than the overflow threshold.
 * If the unscaled problem will not cause overflow, the Level 2 BLAS
 * routine ZTBSV is called. If the matrix A is singular (A(j,j) = 0
 * for some j), then s is set to 0 and a non-trivial solution to
 * A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A * x = s*b; 'T': Solve A**T * x = s*b;
 *                        'C': Solve A**H * x = s*b.
 * @param[in]     diag    'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     kd      The number of subdiagonals or superdiagonals in A (kd >= 0).
 * @param[in]     AB      The triangular band matrix A, stored in the first kd+1 rows.
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of AB (ldab >= kd+1).
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
void zlatbs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const int kd,
    const c128* restrict AB,
    const int ldab,
    c128* restrict X,
    f64* scale,
    f64* restrict cnorm,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    int upper, notran, nounit, normin_n;
    int i, imax, j, jfirst, jinc, jlast, jlen, maind;
    f64 bignum, grow, rec, smlnum, tjj, tmax, tscal, xbnd, xj, xmax;
    c128 csumj, tjjs = 0.0, uscal;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    normin_n = (normin[0] == 'N' || normin[0] == 'n');

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
        xerbla("ZLATBS", -(*info));
        return;
    }

    *scale = ONE;
    if (n == 0) {
        return;
    }

    smlnum = DBL_MIN / DBL_EPSILON;
    bignum = ONE / smlnum;

    if (normin_n) {
        if (upper) {
            for (j = 0; j < n; j++) {
                jlen = (kd < j) ? kd : j;
                if (jlen > 0) {
                    cnorm[j] = cblas_dzasum(jlen, &AB[kd - jlen + j * ldab], 1);
                } else {
                    cnorm[j] = ZERO;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                if (jlen > 0) {
                    cnorm[j] = cblas_dzasum(jlen, &AB[1 + j * ldab], 1);
                } else {
                    cnorm[j] = ZERO;
                }
            }
        }
    }

    imax = cblas_idamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum * HALF) {
        tscal = ONE;
    } else {
        tscal = HALF / (smlnum * tmax);
        cblas_dscal(n, tscal, cnorm, 1);
    }

    xmax = ZERO;
    for (j = 0; j < n; j++) {
        f64 tmp = cabs2(X[j]);
        if (xmax < tmp) xmax = tmp;
    }
    xbnd = xmax;

    if (notran) {

        if (upper) {
            jfirst = n - 1;
            jlast = 0;
            jinc = -1;
            maind = kd;
        } else {
            jfirst = 0;
            jlast = n - 1;
            jinc = 1;
            maind = 0;
        }

        if (tscal != ONE) {
            grow = ZERO;
        } else if (nounit) {
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                tjjs = AB[maind + j * ldab];
                tjj = cabs1(tjjs);
                if (tjj >= smlnum) {
                    xbnd = (xbnd < (ONE < tjj ? ONE : tjj) * grow) ? xbnd : (ONE < tjj ? ONE : tjj) * grow;
                } else {
                    xbnd = ZERO;
                }
                if (tjj + cnorm[j] >= smlnum) {
                    grow = grow * (tjj / (tjj + cnorm[j]));
                } else {
                    grow = ZERO;
                }
            }
            grow = xbnd;
        } else {
            grow = (ONE < HALF / (xbnd > smlnum ? xbnd : smlnum)) ? ONE : HALF / (xbnd > smlnum ? xbnd : smlnum);
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                grow = grow * (ONE / (ONE + cnorm[j]));
            }
        }

    } else {

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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = (grow < xbnd / xj) ? grow : xbnd / xj;
                tjjs = AB[maind + j * ldab];
                tjj = cabs1(tjjs);
                if (tjj >= smlnum) {
                    if (xj > tjj)
                        xbnd = xbnd * (tjj / xj);
                } else {
                    xbnd = ZERO;
                }
            }
            grow = (grow < xbnd) ? grow : xbnd;
        } else {
            grow = (ONE < HALF / (xbnd > smlnum ? xbnd : smlnum)) ? ONE : HALF / (xbnd > smlnum ? xbnd : smlnum);
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = grow / xj;
            }
        }
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
        cblas_ztbsv(CblasColMajor,
                    upper ? CblasUpper : CblasLower,
                    cblas_trans,
                    nounit ? CblasNonUnit : CblasUnit,
                    n, kd, AB, ldab, X, 1);
    } else {

        if (xmax > bignum * HALF) {
            *scale = (bignum * HALF) / xmax;
            cblas_zdscal(n, *scale, X, 1);
            xmax = bignum;
        } else {
            xmax = xmax * TWO;
        }

        if (notran) {
            /* Solve A * x = b */
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1(X[j]);
                if (nounit) {
                    tjjs = AB[maind + j * ldab] * tscal;
                } else {
                    tjjs = tscal;
                    if (tscal == ONE)
                        goto notran_skip;
                }
                tjj = cabs1(tjjs);
                if (tjj > smlnum) {
                    if (tjj < ONE) {
                        if (xj > tjj * bignum) {
                            rec = ONE / xj;
                            cblas_zdscal(n, rec, X, 1);
                            *scale = *scale * rec;
                            xmax = xmax * rec;
                        }
                    }
                    X[j] = zladiv(X[j], tjjs);
                    xj = cabs1(X[j]);
                } else if (tjj > ZERO) {
                    if (xj > tjj * bignum) {
                        rec = (tjj * bignum) / xj;
                        if (cnorm[j] > ONE) {
                            rec = rec / cnorm[j];
                        }
                        cblas_zdscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                    X[j] = zladiv(X[j], tjjs);
                    xj = cabs1(X[j]);
                } else {
                    for (i = 0; i < n; i++) {
                        X[i] = ZERO;
                    }
                    X[j] = ONE;
                    xj = ONE;
                    *scale = ZERO;
                    xmax = ZERO;
                }
notran_skip:
                if (xj > ONE) {
                    rec = ONE / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        rec = rec * HALF;
                        cblas_zdscal(n, rec, X, 1);
                        *scale = *scale * rec;
                    }
                } else if (xj * cnorm[j] > (bignum - xmax)) {
                    cblas_zdscal(n, HALF, X, 1);
                    *scale = *scale * HALF;
                }

                if (upper) {
                    if (j > 0) {
                        jlen = (kd < j) ? kd : j;
                        c128 neg_xj_tscal = -X[j] * tscal;
                        cblas_zaxpy(jlen, &neg_xj_tscal, &AB[kd - jlen + j * ldab], 1, &X[j - jlen], 1);
                        i = cblas_izamax(j, X, 1);
                        xmax = cabs1(X[i]);
                    }
                } else {
                    if (j < n - 1) {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        if (jlen > 0) {
                            c128 neg_xj_tscal = -X[j] * tscal;
                            cblas_zaxpy(jlen, &neg_xj_tscal, &AB[1 + j * ldab], 1, &X[j + 1], 1);
                        }
                        i = j + 1 + cblas_izamax(n - j - 1, &X[j + 1], 1);
                        xmax = cabs1(X[i]);
                    }
                }
            }

        } else if (trans[0] == 'T' || trans[0] == 't') {
            /* Solve A**T * x = b */
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = AB[maind + j * ldab] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE) ? rec * tjj : ONE;
                        uscal = zladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_zdscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (uscal == (c128)ONE) {
                    if (upper) {
                        jlen = (kd < j) ? kd : j;
                        if (jlen > 0) {
                            cblas_zdotu_sub(jlen, &AB[kd - jlen + j * ldab], 1, &X[j - jlen], 1, &csumj);
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        if (jlen > 1) {
                            cblas_zdotu_sub(jlen, &AB[1 + j * ldab], 1, &X[j + 1], 1, &csumj);
                        }
                    }
                } else {
                    if (upper) {
                        jlen = (kd < j) ? kd : j;
                        for (i = 0; i < jlen; i++) {
                            csumj = csumj + (AB[kd - jlen + i + j * ldab] * uscal) * X[j - jlen + i];
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        for (i = 0; i < jlen; i++) {
                            csumj = csumj + (AB[1 + i + j * ldab] * uscal) * X[j + 1 + i];
                        }
                    }
                }

                if (uscal == (c128)tscal) {
                    X[j] = X[j] - csumj;
                    xj = cabs1(X[j]);
                    if (nounit) {
                        tjjs = AB[maind + j * ldab] * tscal;
                    } else {
                        tjjs = tscal;
                        if (tscal == ONE)
                            goto trans_skip;
                    }
                    tjj = cabs1(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_zdscal(n, rec, X, 1);
                                *scale = *scale * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = zladiv(X[j], tjjs);
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            cblas_zdscal(n, rec, X, 1);
                            *scale = *scale * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = zladiv(X[j], tjjs);
                    } else {
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
trans_skip: ;
                } else {
                    X[j] = zladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1(X[j])) ? xmax : cabs1(X[j]);
            }

        } else {
            /* Solve A**H * x = b */
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = conj(AB[maind + j * ldab]) * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE) ? rec * tjj : ONE;
                        uscal = zladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_zdscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (uscal == (c128)ONE) {
                    if (upper) {
                        jlen = (kd < j) ? kd : j;
                        if (jlen > 0) {
                            cblas_zdotc_sub(jlen, &AB[kd - jlen + j * ldab], 1, &X[j - jlen], 1, &csumj);
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        if (jlen > 1) {
                            cblas_zdotc_sub(jlen, &AB[1 + j * ldab], 1, &X[j + 1], 1, &csumj);
                        }
                    }
                } else {
                    if (upper) {
                        jlen = (kd < j) ? kd : j;
                        for (i = 0; i < jlen; i++) {
                            csumj = csumj + (conj(AB[kd - jlen + i + j * ldab]) * uscal) * X[j - jlen + i];
                        }
                    } else {
                        jlen = (kd < n - 1 - j) ? kd : n - 1 - j;
                        for (i = 0; i < jlen; i++) {
                            csumj = csumj + (conj(AB[1 + i + j * ldab]) * uscal) * X[j + 1 + i];
                        }
                    }
                }

                if (uscal == (c128)tscal) {
                    X[j] = X[j] - csumj;
                    xj = cabs1(X[j]);
                    if (nounit) {
                        tjjs = conj(AB[maind + j * ldab]) * tscal;
                    } else {
                        tjjs = tscal;
                        if (tscal == ONE)
                            goto conjtr_skip;
                    }
                    tjj = cabs1(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_zdscal(n, rec, X, 1);
                                *scale = *scale * rec;
                                xmax = xmax * rec;
                            }
                        }
                        X[j] = zladiv(X[j], tjjs);
                    } else if (tjj > ZERO) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            cblas_zdscal(n, rec, X, 1);
                            *scale = *scale * rec;
                            xmax = xmax * rec;
                        }
                        X[j] = zladiv(X[j], tjjs);
                    } else {
                        for (i = 0; i < n; i++) {
                            X[i] = ZERO;
                        }
                        X[j] = ONE;
                        *scale = ZERO;
                        xmax = ZERO;
                    }
conjtr_skip: ;
                } else {
                    X[j] = zladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1(X[j])) ? xmax : cabs1(X[j]);
            }
        }
        *scale = *scale / tscal;
    }

    if (tscal != ONE) {
        cblas_dscal(n, ONE / tscal, cnorm, 1);
    }
}
