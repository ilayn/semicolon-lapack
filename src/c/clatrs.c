/**
 * @file clatrs.c
 * @brief CLATRS solves a triangular system with scaling to prevent overflow.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLATRS solves one of the triangular systems
 *    A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b
 * with scaling to prevent overflow. Here A is an upper or lower
 * triangular matrix, A**T denotes the transpose of A, A**H denotes the
 * conjugate transpose of A, x and b are n-element vectors, and s is a
 * scaling factor, usually less than or equal to 1, chosen so that the
 * components of x will be less than the overflow threshold. If the
 * unscaled problem will not cause overflow, the Level 2 BLAS routine
 * ZTRSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
 * then s is set to 0 and a non-trivial solution to A*x = 0 is returned.
 *
 * @param[in]     uplo    'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     trans   'N': Solve A*x = s*b; 'T': Solve A**T*x = s*b;
 *                        'C': Solve A**H*x = s*b.
 * @param[in]     diag    'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     normin  'Y': CNORM contains column norms on entry;
 *                        'N': CNORM is not set, will be computed.
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     A       The triangular matrix A. Complex*16 array, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A (lda >= max(1,n)).
 * @param[in,out] X       On entry, the right hand side b.
 *                        On exit, overwritten by the solution x.
 *                        Complex*16 array, dimension n.
 * @param[out]    scale   The scaling factor s for the triangular system.
 * @param[in,out] cnorm   If normin='Y', cnorm contains column norms on entry.
 *                        If normin='N', cnorm returns the 1-norm of offdiagonal
 *                        part of column j. Single precision array, dimension n.
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void clatrs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* normin,
    const int n,
    const c64* restrict A,
    const int lda,
    c64* restrict X,
    f32* scale,
    f32* restrict cnorm,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 HALF = 0.5f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    int upper, notran, nounit, normin_n;
    int i, imax, j, jfirst, jinc, jlast;
    f32 bignum, grow, rec, smlnum, tjj, tmax, tscal, xbnd, xj, xmax;
    c64 csumj, tjjs = 0.0f, uscal;

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
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("CLATRS", -(*info));
        return;
    }

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
                cnorm[j] = cblas_scasum(j, &A[j * lda], 1);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                cnorm[j] = cblas_scasum(n - j - 1, &A[j + 1 + j * lda], 1);
            }
            if (n > 0) cnorm[n - 1] = ZERO;
        }
    }

    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM/2
    imax = cblas_isamax(n, cnorm, 1);
    tmax = cnorm[imax];
    if (tmax <= bignum * HALF) {
        tscal = ONE;
    } else {
        if (tmax <= FLT_MAX) {
            // Case 1: All entries in CNORM are valid floating-point numbers
            tscal = HALF / (smlnum * tmax);
            cblas_sscal(n, tscal, cnorm, 1);
        } else {
            // Case 2: At least one column norm of A cannot be represented
            // as a floating-point number. Find the maximum offdiagonal
            // absolute value max( |Re(A(i,j))|, |Im(A(i,j))| ).
            tmax = ZERO;
            if (upper) {
                for (j = 1; j < n; j++) {
                    for (i = 0; i < j; i++) {
                        f32 re = fabsf(crealf(A[i + j * lda]));
                        f32 im = fabsf(cimagf(A[i + j * lda]));
                        if (re > tmax) tmax = re;
                        if (im > tmax) tmax = im;
                    }
                }
            } else {
                for (j = 0; j < n - 1; j++) {
                    for (i = j + 1; i < n; i++) {
                        f32 re = fabsf(crealf(A[i + j * lda]));
                        f32 im = fabsf(cimagf(A[i + j * lda]));
                        if (re > tmax) tmax = re;
                        if (im > tmax) tmax = im;
                    }
                }
            }

            if (tmax <= FLT_MAX) {
                tscal = ONE / (smlnum * tmax);
                for (j = 0; j < n; j++) {
                    if (cnorm[j] <= FLT_MAX) {
                        cnorm[j] = cnorm[j] * tscal;
                    } else {
                        // Recompute the 1-norm without introducing Infinity
                        f32 tscal2 = TWO * tscal;
                        cnorm[j] = ZERO;
                        if (upper) {
                            for (i = 0; i < j; i++) {
                                cnorm[j] = cnorm[j] + tscal2 * cabs2f(A[i + j * lda]);
                            }
                        } else {
                            for (i = j + 1; i < n; i++) {
                                cnorm[j] = cnorm[j] + tscal2 * cabs2f(A[i + j * lda]);
                            }
                        }
                        tscal = tscal2 * HALF;
                    }
                }
            } else {
                // At least one entry of A is not a valid floating-point
                // entry. Rely on TRSV to propagate Inf and NaN.
                CBLAS_TRANSPOSE cblas_trans;
                if (notran) cblas_trans = CblasNoTrans;
                else if (trans[0] == 'T' || trans[0] == 't') cblas_trans = CblasTrans;
                else cblas_trans = CblasConjTrans;

                cblas_ctrsv(CblasColMajor,
                            upper ? CblasUpper : CblasLower,
                            cblas_trans,
                            nounit ? CblasNonUnit : CblasUnit,
                            n, A, lda, X, 1);
                return;
            }
        }
    }

    // Compute a bound on the computed solution vector to see if the
    // Level 2 BLAS routine ZTRSV can be used
    xmax = ZERO;
    for (j = 0; j < n; j++) {
        f32 temp = cabs2f(X[j]);
        if (temp > xmax) xmax = temp;
    }
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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                tjjs = A[j + j * lda];
                tjj = cabs1f(tjjs);
                xbnd = (xbnd < ((ONE < tjj ? ONE : tjj) * grow))
                     ? xbnd : ((ONE < tjj ? ONE : tjj) * grow);
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
            grow = (ONE < HALF / (xbnd > smlnum ? xbnd : smlnum))
                 ? HALF / (xbnd > smlnum ? xbnd : smlnum) : ONE;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                grow = grow * (ONE / (ONE + cnorm[j]));
            }
        }
    } else {
        // Compute the growth in A**T * x = b or A**H * x = b
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
            grow = HALF / (xbnd > smlnum ? xbnd : smlnum);
            xbnd = grow;
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                if (grow <= smlnum) break;
                xj = ONE + cnorm[j];
                grow = (grow < xbnd / xj) ? grow : xbnd / xj;
                tjjs = A[j + j * lda];
                tjj = cabs1f(tjjs);
                if (xj > tjj) {
                    xbnd = xbnd * (tjj / xj);
                }
            }
            if (grow > smlnum) {
                grow = (grow < xbnd) ? grow : xbnd;
            }
        } else {
            grow = (ONE < HALF / (xbnd > smlnum ? xbnd : smlnum))
                 ? HALF / (xbnd > smlnum ? xbnd : smlnum) : ONE;
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
        CBLAS_TRANSPOSE cblas_trans;
        if (notran) cblas_trans = CblasNoTrans;
        else if (trans[0] == 'T' || trans[0] == 't') cblas_trans = CblasTrans;
        else cblas_trans = CblasConjTrans;

        cblas_ctrsv(CblasColMajor,
                    upper ? CblasUpper : CblasLower,
                    cblas_trans,
                    nounit ? CblasNonUnit : CblasUnit,
                    n, A, lda, X, 1);
    } else {
        // Use a Level 1 BLAS solve, scaling intermediate results

        if (xmax > bignum * HALF) {
            // Scale X so that its components are less than or equal to
            // BIGNUM in absolute value
            *scale = (bignum * HALF) / xmax;
            cblas_csscal(n, *scale, X, 1);
            xmax = bignum;
        } else {
            xmax = xmax * TWO;
        }

        if (notran) {
            // Solve A * x = b
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                if (nounit) {
                    tjjs = A[j + j * lda] * tscal;
                } else {
                    tjjs = tscal;
                }
                if (nounit || tscal != ONE) {
                    tjj = cabs1f(tjjs);
                    if (tjj > smlnum) {
                        if (tjj < ONE) {
                            if (xj > tjj * bignum) {
                                rec = ONE / xj;
                                cblas_csscal(n, rec, X, 1);
                                *scale = *scale * rec;
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
                            *scale = *scale * rec;
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
                }
                // Scale x if necessary to avoid overflow when adding a
                // multiple of column j of A
                if (xj > ONE) {
                    rec = ONE / xj;
                    if (cnorm[j] > (bignum - xmax) * rec) {
                        rec = rec * HALF;
                        cblas_csscal(n, rec, X, 1);
                        *scale = *scale * rec;
                    }
                } else if (xj * cnorm[j] > (bignum - xmax)) {
                    cblas_csscal(n, HALF, X, 1);
                    *scale = *scale * HALF;
                }

                if (upper) {
                    if (j > 0) {
                        c64 alpha = -X[j] * tscal;
                        cblas_caxpy(j, &alpha, &A[j * lda], 1, X, 1);
                        i = cblas_icamax(j, X, 1);
                        xmax = cabs1f(X[i]);
                    }
                } else {
                    if (j < n - 1) {
                        c64 alpha = -X[j] * tscal;
                        cblas_caxpy(n - j - 1, &alpha, &A[j + 1 + j * lda], 1, &X[j + 1], 1);
                        i = j + 1 + cblas_icamax(n - j - 1, &X[j + 1], 1);
                        xmax = cabs1f(X[i]);
                    }
                }
            }

        } else if (trans[0] == 'T' || trans[0] == 't') {
            // Solve A**T * x = b
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = A[j + j * lda] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE) ? rec * tjj : ONE;
                        uscal = cladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_csscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (crealf(uscal) == ONE && cimagf(uscal) == 0.0f) {
                    // If the scaling needed for A in the dot product is 1,
                    // call ZDOTU to perform the dot product
                    if (upper) {
                        csumj = cblas_cdotu(j, &A[j * lda], 1, X, 1);
                    } else if (j < n - 1) {
                        csumj = cblas_cdotu(n - j - 1, &A[j + 1 + j * lda], 1, &X[j + 1], 1);
                    }
                } else {
                    // Otherwise, use in-line code for the dot product
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            csumj = csumj + (A[i + j * lda] * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = j + 1; i < n; i++) {
                            csumj = csumj + (A[i + j * lda] * uscal) * X[i];
                        }
                    }
                }

                if (crealf(uscal) == tscal && cimagf(uscal) == 0.0f) {
                    // Compute x(j) := ( x(j) - CSUMJ ) / A(j,j)
                    X[j] = X[j] - csumj;
                    xj = cabs1f(X[j]);
                    if (nounit) {
                        tjjs = A[j + j * lda] * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    if (nounit || tscal != ONE) {
                        tjj = cabs1f(tjjs);
                        if (tjj > smlnum) {
                            if (tjj < ONE) {
                                if (xj > tjj * bignum) {
                                    rec = ONE / xj;
                                    cblas_csscal(n, rec, X, 1);
                                    *scale = *scale * rec;
                                    xmax = xmax * rec;
                                }
                            }
                            X[j] = cladiv(X[j], tjjs);
                        } else if (tjj > ZERO) {
                            if (xj > tjj * bignum) {
                                rec = (tjj * bignum) / xj;
                                cblas_csscal(n, rec, X, 1);
                                *scale = *scale * rec;
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
                    }
                } else {
                    // Compute x(j) := x(j) / A(j,j) - CSUMJ
                    X[j] = cladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1f(X[j])) ? xmax : cabs1f(X[j]);
            }

        } else {
            // Solve A**H * x = b
            for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
                xj = cabs1f(X[j]);
                uscal = tscal;
                rec = ONE / (xmax > ONE ? xmax : ONE);
                if (cnorm[j] > (bignum - xj) * rec) {
                    rec = rec * HALF;
                    if (nounit) {
                        tjjs = conjf(A[j + j * lda]) * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    tjj = cabs1f(tjjs);
                    if (tjj > ONE) {
                        rec = (rec * tjj < ONE) ? rec * tjj : ONE;
                        uscal = cladiv(uscal, tjjs);
                    }
                    if (rec < ONE) {
                        cblas_csscal(n, rec, X, 1);
                        *scale = *scale * rec;
                        xmax = xmax * rec;
                    }
                }

                csumj = ZERO;
                if (crealf(uscal) == ONE && cimagf(uscal) == 0.0f) {
                    // If the scaling needed for A in the dot product is 1,
                    // call ZDOTC to perform the conjugated dot product
                    if (upper) {
                        csumj = cblas_cdotc(j, &A[j * lda], 1, X, 1);
                    } else if (j < n - 1) {
                        csumj = cblas_cdotc(n - j - 1, &A[j + 1 + j * lda], 1, &X[j + 1], 1);
                    }
                } else {
                    // Otherwise, use in-line code for the dot product
                    if (upper) {
                        for (i = 0; i < j; i++) {
                            csumj = csumj + (conjf(A[i + j * lda]) * uscal) * X[i];
                        }
                    } else if (j < n - 1) {
                        for (i = j + 1; i < n; i++) {
                            csumj = csumj + (conjf(A[i + j * lda]) * uscal) * X[i];
                        }
                    }
                }

                if (crealf(uscal) == tscal && cimagf(uscal) == 0.0f) {
                    // Compute x(j) := ( x(j) - CSUMJ ) / conj(A(j,j))
                    X[j] = X[j] - csumj;
                    xj = cabs1f(X[j]);
                    if (nounit) {
                        tjjs = conjf(A[j + j * lda]) * tscal;
                    } else {
                        tjjs = tscal;
                    }
                    if (nounit || tscal != ONE) {
                        tjj = cabs1f(tjjs);
                        if (tjj > smlnum) {
                            if (tjj < ONE) {
                                if (xj > tjj * bignum) {
                                    rec = ONE / xj;
                                    cblas_csscal(n, rec, X, 1);
                                    *scale = *scale * rec;
                                    xmax = xmax * rec;
                                }
                            }
                            X[j] = cladiv(X[j], tjjs);
                        } else if (tjj > ZERO) {
                            if (xj > tjj * bignum) {
                                rec = (tjj * bignum) / xj;
                                cblas_csscal(n, rec, X, 1);
                                *scale = *scale * rec;
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
                    }
                } else {
                    // Compute x(j) := x(j) / conj(A(j,j)) - CSUMJ
                    X[j] = cladiv(X[j], tjjs) - csumj;
                }
                xmax = (xmax > cabs1f(X[j])) ? xmax : cabs1f(X[j]);
            }
        }
        *scale = *scale / tscal;
    }

    // Scale the column norms by 1/TSCAL for return
    if (tscal != ONE) {
        cblas_sscal(n, ONE / tscal, cnorm, 1);
    }
}
