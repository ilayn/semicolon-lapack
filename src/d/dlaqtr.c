/**
 * @file dlaqtr.c
 * @brief DLAQTR solves a real quasi-triangular system of equations.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/**
 * DLAQTR solves the real quasi-triangular system
 *
 *              op(T)*p = scale*c,               if lreal = 1
 *
 * or the complex quasi-triangular systems
 *
 *            op(T + iB)*(p+iq) = scale*(c+id),  if lreal = 0
 *
 * in real arithmetic, where T is upper quasi-triangular.
 * If lreal = 0, then the first diagonal block of T must be
 * 1 by 1, B is the specially structured matrix
 *
 *                B = [ b(0) b(1) ... b(n-1) ]
 *                    [       w              ]
 *                    [           w          ]
 *                    [              .       ]
 *                    [                 w    ]
 *
 * op(A) = A or A**T, A**T denotes the transpose of matrix A.
 *
 * On input, X = [ c ].  On output, X = [ p ].
 *               [ d ]                  [ q ]
 *
 * This subroutine is designed for the condition number estimation
 * in routine DTRSNA.
 *
 * @param[in] ltran   Specifies the option of conjugate transpose:
 *                    = 0: op(T+i*B) = T+i*B,
 *                    = 1: op(T+i*B) = (T+i*B)**T.
 * @param[in] lreal   Specifies the input matrix structure:
 *                    = 0: the input is complex,
 *                    = 1: the input is real.
 * @param[in] n       The order of T+i*B. n >= 0.
 * @param[in] T       The upper quasi-triangular matrix T, in Schur canonical form.
 *                    Dimension (ldt, n).
 * @param[in] ldt     The leading dimension of T. ldt >= max(1, n).
 * @param[in] B       Array, dimension (n). The elements to form the matrix B.
 *                    If lreal = 1, B is not referenced.
 * @param[in] w       The diagonal element of the matrix B.
 *                    If lreal = 1, w is not referenced.
 * @param[out] scale  The scale factor.
 * @param[in,out] X   Array, dimension (2*n). On entry, the right hand side.
 *                    On exit, overwritten by the solution.
 * @param[out] work   Workspace array, dimension (n).
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - = 1: some diagonal 1 by 1 block has been perturbed by
 *                           a small number SMIN to keep nonsingularity.
 *                         - = 2: some diagonal 2 by 2 block has been perturbed by
 *                           a small number in DLALN2 to keep nonsingularity.
 */
void dlaqtr(const int ltran, const int lreal, const int n,
            const f64* T, const int ldt,
            const f64* B, const f64 w,
            f64* scale, f64* X, f64* work, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int notran;
    int i, ierr, j, j1, j2, jnext, k, n1, n2;
    f64 bignum, eps, rec, scaloc, si, smin, sminw;
    f64 smlnum, sr, tjj, tmp, xj, xmax, xnorm, z;
    f64 d[4];  /* 2x2 stored column-major: d[0]=d(1,1), d[1]=d(2,1), d[2]=d(1,2), d[3]=d(2,2) */
    f64 v[4];  /* 2x2 stored column-major */

    /* Do not test the input parameters for errors */
    notran = !ltran;
    *info = 0;

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Set constants to control overflow */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    bignum = ONE / smlnum;

    xnorm = dlange("M", n, n, T, ldt, d);
    if (!lreal)
        xnorm = fmax(xnorm, fmax(fabs(w), dlange("M", n, 1, B, n, d)));
    smin = fmax(smlnum, eps * xnorm);

    /* Compute 1-norm of each column of strictly upper triangular
     * part of T to control overflow in triangular solver. */
    work[0] = ZERO;
    for (j = 1; j < n; j++) {
        work[j] = cblas_dasum(j, &T[j * ldt], 1);
    }

    if (!lreal) {
        for (i = 1; i < n; i++) {
            work[i] = work[i] + fabs(B[i]);
        }
    }

    n2 = 2 * n;
    n1 = n;
    if (!lreal)
        n1 = n2;
    k = cblas_idamax(n1, X, 1);
    xmax = fabs(X[k]);
    *scale = ONE;

    if (xmax > bignum) {
        *scale = bignum / xmax;
        cblas_dscal(n1, *scale, X, 1);
        xmax = bignum;
    }

    if (lreal) {

        if (notran) {

            /* Solve T*p = scale*c */
            jnext = n - 1;
            for (j = n - 1; j >= 0; j--) {
                if (j > jnext)
                    continue;
                j1 = j;
                j2 = j;
                jnext = j - 1;
                if (j > 0) {
                    if (T[j + (j - 1) * ldt] != ZERO) {
                        j1 = j - 1;
                        jnext = j - 2;
                    }
                }

                if (j1 == j2) {

                    /* Meet 1 by 1 diagonal block
                     * Scale to avoid overflow when computing x(j) = b(j)/T(j,j) */
                    xj = fabs(X[j1]);
                    tjj = fabs(T[j1 + j1 * ldt]);
                    tmp = T[j1 + j1 * ldt];
                    if (tjj < smin) {
                        tmp = smin;
                        tjj = smin;
                        *info = 1;
                    }

                    if (xj == ZERO)
                        continue;

                    if (tjj < ONE) {
                        if (xj > bignum * tjj) {
                            rec = ONE / xj;
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }
                    X[j1] = X[j1] / tmp;
                    xj = fabs(X[j1]);

                    /* Scale x if necessary to avoid overflow when adding a
                     * multiple of column j1 of T. */
                    if (xj > ONE) {
                        rec = ONE / xj;
                        if (work[j1] > (bignum - xmax) * rec) {
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                        }
                    }
                    if (j1 > 0) {
                        cblas_daxpy(j1, -X[j1], &T[j1 * ldt], 1, X, 1);
                        k = cblas_idamax(j1, X, 1);
                        xmax = fabs(X[k]);
                    }

                } else {

                    /* Meet 2 by 2 diagonal block
                     * Call 2 by 2 linear system solve, to take
                     * care of possible overflow by scaling factor. */
                    d[0] = X[j1];
                    d[1] = X[j2];
                    dlaln2(0, 2, 1, smin, ONE, &T[j1 + j1 * ldt],
                           ldt, ONE, ONE, d, 2, ZERO, ZERO, v, 2,
                           &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 2;

                    if (scaloc != ONE) {
                        cblas_dscal(n, scaloc, X, 1);
                        *scale *= scaloc;
                    }
                    X[j1] = v[0];
                    X[j2] = v[1];

                    /* Scale V(1,1) (= X(J1)) and/or V(2,1) (=X(J2))
                     * to avoid overflow in updating right-hand side. */
                    xj = fmax(fabs(v[0]), fabs(v[1]));
                    if (xj > ONE) {
                        rec = ONE / xj;
                        if (fmax(work[j1], work[j2]) > (bignum - xmax) * rec) {
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                        }
                    }

                    /* Update right-hand side */
                    if (j1 > 0) {
                        cblas_daxpy(j1, -X[j1], &T[j1 * ldt], 1, X, 1);
                        cblas_daxpy(j1, -X[j2], &T[j2 * ldt], 1, X, 1);
                        k = cblas_idamax(j1, X, 1);
                        xmax = fabs(X[k]);
                    }

                }
            }

        } else {

            /* Solve T**T*p = scale*c */
            jnext = 0;
            for (j = 0; j < n; j++) {
                if (j < jnext)
                    continue;
                j1 = j;
                j2 = j;
                jnext = j + 1;
                if (j < n - 1) {
                    if (T[(j + 1) + j * ldt] != ZERO) {
                        j2 = j + 1;
                        jnext = j + 2;
                    }
                }

                if (j1 == j2) {

                    /* 1 by 1 diagonal block
                     * Scale if necessary to avoid overflow in forming the
                     * right-hand side element by inner product. */
                    xj = fabs(X[j1]);
                    if (xmax > ONE) {
                        rec = ONE / xmax;
                        if (work[j1] > (bignum - xj) * rec) {
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }

                    X[j1] = X[j1] - cblas_ddot(j1, &T[j1 * ldt], 1, X, 1);

                    xj = fabs(X[j1]);
                    tjj = fabs(T[j1 + j1 * ldt]);
                    tmp = T[j1 + j1 * ldt];
                    if (tjj < smin) {
                        tmp = smin;
                        tjj = smin;
                        *info = 1;
                    }

                    if (tjj < ONE) {
                        if (xj > bignum * tjj) {
                            rec = ONE / xj;
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }
                    X[j1] = X[j1] / tmp;
                    xmax = fmax(xmax, fabs(X[j1]));

                } else {

                    /* 2 by 2 diagonal block
                     * Scale if necessary to avoid overflow in forming the
                     * right-hand side elements by inner product. */
                    xj = fmax(fabs(X[j1]), fabs(X[j2]));
                    if (xmax > ONE) {
                        rec = ONE / xmax;
                        if (fmax(work[j2], work[j1]) > (bignum - xj) * rec) {
                            cblas_dscal(n, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }

                    d[0] = X[j1] - cblas_ddot(j1, &T[j1 * ldt], 1, X, 1);
                    d[1] = X[j2] - cblas_ddot(j1, &T[j2 * ldt], 1, X, 1);

                    dlaln2(1, 2, 1, smin, ONE, &T[j1 + j1 * ldt],
                           ldt, ONE, ONE, d, 2, ZERO, ZERO, v, 2,
                           &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 2;

                    if (scaloc != ONE) {
                        cblas_dscal(n, scaloc, X, 1);
                        *scale *= scaloc;
                    }
                    X[j1] = v[0];
                    X[j2] = v[1];
                    xmax = fmax(fabs(X[j1]), fmax(fabs(X[j2]), xmax));

                }
            }
        }

    } else {

        sminw = fmax(eps * fabs(w), smin);
        if (notran) {

            /* Solve (T + iB)*(p+iq) = c+id */
            jnext = n - 1;
            for (j = n - 1; j >= 0; j--) {
                if (j > jnext)
                    continue;
                j1 = j;
                j2 = j;
                jnext = j - 1;
                if (j > 0) {
                    if (T[j + (j - 1) * ldt] != ZERO) {
                        j1 = j - 1;
                        jnext = j - 2;
                    }
                }

                if (j1 == j2) {

                    /* 1 by 1 diagonal block
                     * Scale if necessary to avoid overflow in division */
                    z = w;
                    if (j1 == 0)
                        z = B[0];
                    xj = fabs(X[j1]) + fabs(X[n + j1]);
                    tjj = fabs(T[j1 + j1 * ldt]) + fabs(z);
                    tmp = T[j1 + j1 * ldt];
                    if (tjj < sminw) {
                        tmp = sminw;
                        tjj = sminw;
                        *info = 1;
                    }

                    if (xj == ZERO)
                        continue;

                    if (tjj < ONE) {
                        if (xj > bignum * tjj) {
                            rec = ONE / xj;
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }
                    dladiv(X[j1], X[n + j1], tmp, z, &sr, &si);
                    X[j1] = sr;
                    X[n + j1] = si;
                    xj = fabs(X[j1]) + fabs(X[n + j1]);

                    /* Scale x if necessary to avoid overflow when adding a
                     * multiple of column j1 of T. */
                    if (xj > ONE) {
                        rec = ONE / xj;
                        if (work[j1] > (bignum - xmax) * rec) {
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                        }
                    }

                    if (j1 > 0) {
                        cblas_daxpy(j1, -X[j1], &T[j1 * ldt], 1, X, 1);
                        cblas_daxpy(j1, -X[n + j1], &T[j1 * ldt], 1, &X[n], 1);

                        X[0] = X[0] + B[j1] * X[n + j1];
                        X[n] = X[n] - B[j1] * X[j1];

                        xmax = ZERO;
                        for (k = 0; k < j1; k++) {
                            xmax = fmax(xmax, fabs(X[k]) + fabs(X[k + n]));
                        }
                    }

                } else {

                    /* Meet 2 by 2 diagonal block */
                    d[0] = X[j1];
                    d[1] = X[j2];
                    d[2] = X[n + j1];
                    d[3] = X[n + j2];
                    dlaln2(0, 2, 2, sminw, ONE, &T[j1 + j1 * ldt],
                           ldt, ONE, ONE, d, 2, ZERO, -w, v, 2,
                           &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 2;

                    if (scaloc != ONE) {
                        cblas_dscal(2 * n, scaloc, X, 1);
                        *scale = scaloc * (*scale);
                    }
                    X[j1] = v[0];
                    X[j2] = v[1];
                    X[n + j1] = v[2];
                    X[n + j2] = v[3];

                    /* Scale X(J1), .... to avoid overflow in
                     * updating right hand side. */
                    xj = fmax(fabs(v[0]) + fabs(v[2]), fabs(v[1]) + fabs(v[3]));
                    if (xj > ONE) {
                        rec = ONE / xj;
                        if (fmax(work[j1], work[j2]) > (bignum - xmax) * rec) {
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                        }
                    }

                    /* Update the right-hand side. */
                    if (j1 > 0) {
                        cblas_daxpy(j1, -X[j1], &T[j1 * ldt], 1, X, 1);
                        cblas_daxpy(j1, -X[j2], &T[j2 * ldt], 1, X, 1);

                        cblas_daxpy(j1, -X[n + j1], &T[j1 * ldt], 1, &X[n], 1);
                        cblas_daxpy(j1, -X[n + j2], &T[j2 * ldt], 1, &X[n], 1);

                        X[0] = X[0] + B[j1] * X[n + j1] + B[j2] * X[n + j2];
                        X[n] = X[n] - B[j1] * X[j1] - B[j2] * X[j2];

                        xmax = ZERO;
                        for (k = 0; k < j1; k++) {
                            xmax = fmax(fabs(X[k]) + fabs(X[k + n]), xmax);
                        }
                    }

                }
            }

        } else {

            /* Solve (T + iB)**T*(p+iq) = c+id */
            jnext = 0;
            for (j = 0; j < n; j++) {
                if (j < jnext)
                    continue;
                j1 = j;
                j2 = j;
                jnext = j + 1;
                if (j < n - 1) {
                    if (T[(j + 1) + j * ldt] != ZERO) {
                        j2 = j + 1;
                        jnext = j + 2;
                    }
                }

                if (j1 == j2) {

                    /* 1 by 1 diagonal block
                     * Scale if necessary to avoid overflow in forming the
                     * right-hand side element by inner product. */
                    xj = fabs(X[j1]) + fabs(X[j1 + n]);
                    if (xmax > ONE) {
                        rec = ONE / xmax;
                        if (work[j1] > (bignum - xj) * rec) {
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }

                    X[j1] = X[j1] - cblas_ddot(j1, &T[j1 * ldt], 1, X, 1);
                    X[n + j1] = X[n + j1] - cblas_ddot(j1, &T[j1 * ldt], 1, &X[n], 1);
                    if (j1 > 0) {
                        X[j1] = X[j1] - B[j1] * X[n];
                        X[n + j1] = X[n + j1] + B[j1] * X[0];
                    }
                    xj = fabs(X[j1]) + fabs(X[j1 + n]);

                    z = w;
                    if (j1 == 0)
                        z = B[0];

                    /* Scale if necessary to avoid overflow in
                     * complex division */
                    tjj = fabs(T[j1 + j1 * ldt]) + fabs(z);
                    tmp = T[j1 + j1 * ldt];
                    if (tjj < sminw) {
                        tmp = sminw;
                        tjj = sminw;
                        *info = 1;
                    }

                    if (tjj < ONE) {
                        if (xj > bignum * tjj) {
                            rec = ONE / xj;
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }
                    dladiv(X[j1], X[n + j1], tmp, -z, &sr, &si);
                    X[j1] = sr;
                    X[j1 + n] = si;
                    xmax = fmax(fabs(X[j1]) + fabs(X[j1 + n]), xmax);

                } else {

                    /* 2 by 2 diagonal block
                     * Scale if necessary to avoid overflow in forming the
                     * right-hand side element by inner product. */
                    xj = fmax(fabs(X[j1]) + fabs(X[n + j1]),
                              fabs(X[j2]) + fabs(X[n + j2]));
                    if (xmax > ONE) {
                        rec = ONE / xmax;
                        if (fmax(work[j1], work[j2]) > (bignum - xj) / xmax) {
                            cblas_dscal(n2, rec, X, 1);
                            *scale *= rec;
                            xmax *= rec;
                        }
                    }

                    d[0] = X[j1] - cblas_ddot(j1, &T[j1 * ldt], 1, X, 1);
                    d[1] = X[j2] - cblas_ddot(j1, &T[j2 * ldt], 1, X, 1);
                    d[2] = X[n + j1] - cblas_ddot(j1, &T[j1 * ldt], 1, &X[n], 1);
                    d[3] = X[n + j2] - cblas_ddot(j1, &T[j2 * ldt], 1, &X[n], 1);
                    d[0] = d[0] - B[j1] * X[n];
                    d[1] = d[1] - B[j2] * X[n];
                    d[2] = d[2] + B[j1] * X[0];
                    d[3] = d[3] + B[j2] * X[0];

                    dlaln2(1, 2, 2, sminw, ONE, &T[j1 + j1 * ldt],
                           ldt, ONE, ONE, d, 2, ZERO, w, v, 2,
                           &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 2;

                    if (scaloc != ONE) {
                        cblas_dscal(n2, scaloc, X, 1);
                        *scale = scaloc * (*scale);
                    }
                    X[j1] = v[0];
                    X[j2] = v[1];
                    X[n + j1] = v[2];
                    X[n + j2] = v[3];
                    xmax = fmax(fabs(X[j1]) + fabs(X[n + j1]),
                           fmax(fabs(X[j2]) + fabs(X[n + j2]), xmax));

                }
            }

        }

    }
}
