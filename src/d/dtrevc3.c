/**
 * @file dtrevc3.c
 * @brief DTREVC3 computes eigenvectors of a real upper quasi-triangular matrix.
 *
 * This is the Level 3 BLAS version with blocked back-transformation.
 * Faithful port from LAPACK SRC/dtrevc3.f
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <float.h>
#include <cblas.h>

/* Internal constants */
#define NBMIN 8
#define NBMAX 128

/**
 * DTREVC3 computes some or all of the right and/or left eigenvectors of
 * a real upper quasi-triangular matrix T.
 *
 * This uses a Level 3 BLAS version of the back transformation.
 *
 * @param[in] side    'R': right eigenvectors only
 *                    'L': left eigenvectors only
 *                    'B': both right and left eigenvectors
 * @param[in] howmny  'A': compute all eigenvectors
 *                    'B': compute all eigenvectors, backtransformed
 *                    'S': compute selected eigenvectors
 * @param[in,out] select  Logical array, dimension (n).
 *                        If howmny='S', specifies which eigenvectors to compute.
 * @param[in] n       Order of matrix T. n >= 0.
 * @param[in] T       Upper quasi-triangular matrix T, dimension (ldt, n).
 * @param[in] ldt     Leading dimension of T. ldt >= max(1, n).
 * @param[in,out] VL  Left eigenvectors, dimension (ldvl, mm).
 * @param[in] ldvl    Leading dimension of VL.
 * @param[in,out] VR  Right eigenvectors, dimension (ldvr, mm).
 * @param[in] ldvr    Leading dimension of VR.
 * @param[in] mm      Number of columns in VL and/or VR.
 * @param[out] m      Number of columns actually used.
 * @param[out] work   Workspace, dimension (max(1, lwork)).
 * @param[in] lwork   Dimension of work. lwork >= max(1, 3*n).
 *                    For optimum performance, lwork >= n + 2*n*nb.
 *                    If lwork = -1, workspace query.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
SEMICOLON_API void dtrevc3(const char* side, const char* howmny, int* select,
                           const int n, double* T, const int ldt,
                           double* VL, const int ldvl,
                           double* VR, const int ldvr,
                           const int mm, int* m,
                           double* work, const int lwork, int* info)
{
    const double zero = 0.0;
    const double one = 1.0;

    /* Local variables */
    int bothv, rightv, leftv;
    int allv, over, somev;
    int lquery;
    int i, j, k, ki, ki2, is, ip, ii;
    int j1, j2, jnxt, ierr;
    int iv, nb, maxwrk;
    double beta, bignum, emax, rec, remax, scale;
    double smin, smlnum, ulp, unfl, vcrit, vmax, wi, wr, xnorm;
    double x[4];  /* 2x2 matrix, column-major */
    int iscomplex[NBMAX];
    int pair;

    /* Decode and test input parameters */
    bothv = (side[0] == 'B' || side[0] == 'b');
    rightv = (side[0] == 'R' || side[0] == 'r') || bothv;
    leftv = (side[0] == 'L' || side[0] == 'l') || bothv;

    allv = (howmny[0] == 'A' || howmny[0] == 'a');
    over = (howmny[0] == 'B' || howmny[0] == 'b');
    somev = (howmny[0] == 'S' || howmny[0] == 's');

    *info = 0;

    /* Workspace query - use NB=64 as a reasonable default for DTREVC */
    nb = 64;
    maxwrk = n + 2 * n * nb;
    if (maxwrk < 1) maxwrk = 1;
    work[0] = (double)maxwrk;
    lquery = (lwork == -1);

    if (!rightv && !leftv) {
        *info = -1;
    } else if (!allv && !over && !somev) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        *info = -8;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        *info = -10;
    } else if (lwork < (3 * n > 1 ? 3 * n : 1) && !lquery) {
        *info = -14;
    } else {
        /* Set m to the number of columns required */
        if (somev) {
            *m = 0;
            pair = 0;
            for (j = 0; j < n; j++) {
                if (pair) {
                    pair = 0;
                    select[j] = 0;
                } else {
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] == zero) {
                            if (select[j])
                                (*m)++;
                        } else {
                            pair = 1;
                            if (select[j] || select[j + 1]) {
                                select[j] = 1;
                                *m += 2;
                            }
                        }
                    } else {
                        if (select[n - 1])
                            (*m)++;
                    }
                }
            }
        } else {
            *m = n;
        }

        if (mm < *m) {
            *info = -11;
        }
    }

    if (*info != 0) {
        /* Call xerbla equivalent - return with error */
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Use blocked version of back-transformation if sufficient workspace.
     * Zero out the workspace to avoid potential NaN propagation. */
    if (over && lwork >= n + 2 * n * NBMIN) {
        nb = (lwork - n) / (2 * n);
        if (nb > NBMAX) nb = NBMAX;
        dlaset("F", n, 1 + 2 * nb, zero, zero, work, n);
    } else {
        nb = 1;
    }

    /* Set the constants to control overflow */
    unfl = dlamch("S");
    /* ovfl = one / unfl; -- not used in this routine */
    ulp = dlamch("P");
    smlnum = unfl * ((double)n / ulp);
    bignum = (one - ulp) / smlnum;

    /* Compute 1-norm of each column of strictly upper triangular
     * part of T to control overflow in triangular solver. */
    work[0] = zero;
    for (j = 1; j < n; j++) {
        work[j] = zero;
        for (i = 0; i < j; i++) {
            work[j] += fabs(T[i + j * ldt]);
        }
    }

    /* ============================================================
     * Compute right eigenvectors.
     */
    if (rightv) {
        /* IV is index of column in current block.
         * For complex right vector, uses IV-1 for real part and IV for complex part.
         * Non-blocked version always uses IV=1 (0-based: 1);
         * blocked version starts with IV=NB-1, goes down to 0 or 1.
         * (Note the "0-th" column is used for 1-norms computed above.) */
        iv = 1;
        if (nb > 2) {
            iv = nb - 1;
        }

        ip = 0;
        is = *m - 1;  /* 0-based index */

        for (ki = n - 1; ki >= 0; ki--) {
            if (ip == -1) {
                /* Previous iteration (ki+1) was second of conjugate pair,
                 * so this ki is first of conjugate pair; skip to next iteration */
                ip = 1;
                continue;
            } else if (ki == 0) {
                /* Last column, so this ki must be real eigenvalue */
                ip = 0;
            } else if (T[ki + (ki - 1) * ldt] == zero) {
                /* Zero on sub-diagonal, so this ki is real eigenvalue */
                ip = 0;
            } else {
                /* Non-zero on sub-diagonal, so this ki is second of conjugate pair */
                ip = -1;
            }

            if (somev) {
                if (ip == 0) {
                    if (!select[ki])
                        continue;
                } else {
                    if (!select[ki - 1])
                        continue;
                }
            }

            /* Compute the ki-th eigenvalue (wr, wi) */
            wr = T[ki + ki * ldt];
            wi = zero;
            if (ip != 0) {
                wi = sqrt(fabs(T[ki + (ki - 1) * ldt])) *
                     sqrt(fabs(T[(ki - 1) + ki * ldt]));
            }
            smin = ulp * (fabs(wr) + fabs(wi));
            if (smin < smlnum) smin = smlnum;

            if (ip == 0) {
                /* --------------------------------------------------------
                 * Real right eigenvector
                 */
                work[ki + iv * n] = one;

                /* Form right-hand side */
                for (k = 0; k < ki; k++) {
                    work[k + iv * n] = -T[k + ki * ldt];
                }

                /* Solve upper quasi-triangular system:
                 * [T(0:ki-1, 0:ki-1) - wr] * X = scale * work */
                jnxt = ki - 1;
                for (j = ki - 1; j >= 0; j--) {
                    if (j > jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j - 1;
                    if (j > 0) {
                        if (T[j + (j - 1) * ldt] != zero) {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */
                        dlaln2(0, 1, 1, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + iv * n], n, wr, zero,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale X(0,0) to avoid overflow when updating
                         * the right-hand side */
                        if (xnorm > one) {
                            if (work[j] > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(ki + 1, scale, &work[iv * n], 1);
                        }
                        work[j + iv * n] = x[0];

                        /* Update right-hand side */
                        cblas_daxpy(j, -x[0], &T[j * ldt], 1, &work[iv * n], 1);

                    } else {
                        /* 2-by-2 diagonal block */
                        dlaln2(0, 2, 1, smin, one, &T[(j - 1) + (j - 1) * ldt], ldt,
                               one, one, &work[(j - 1) + iv * n], n, wr, zero,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale X(0,0) and X(1,0) to avoid overflow */
                        if (xnorm > one) {
                            beta = work[j - 1] > work[j] ? work[j - 1] : work[j];
                            if (beta > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                x[1] = x[1] / xnorm;
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(ki + 1, scale, &work[iv * n], 1);
                        }
                        work[(j - 1) + iv * n] = x[0];
                        work[j + iv * n] = x[1];

                        /* Update right-hand side */
                        cblas_daxpy(j - 1, -x[0], &T[(j - 1) * ldt], 1, &work[iv * n], 1);
                        cblas_daxpy(j - 1, -x[1], &T[j * ldt], 1, &work[iv * n], 1);
                    }
                }

                /* Copy the vector x or Q*x to VR and normalize */
                if (!over) {
                    /* No back-transform: copy x to VR and normalize */
                    cblas_dcopy(ki + 1, &work[iv * n], 1, &VR[is * ldvr], 1);

                    ii = cblas_idamax(ki + 1, &VR[is * ldvr], 1);
                    remax = one / fabs(VR[ii + is * ldvr]);
                    cblas_dscal(ki + 1, remax, &VR[is * ldvr], 1);

                    for (k = ki + 1; k < n; k++) {
                        VR[k + is * ldvr] = zero;
                    }

                } else if (nb == 1) {
                    /* Version 1: back-transform each vector with GEMV, Q*x */
                    if (ki > 0) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, ki,
                                    one, VR, ldvr, &work[iv * n], 1,
                                    work[ki + iv * n], &VR[ki * ldvr], 1);
                    }

                    ii = cblas_idamax(n, &VR[ki * ldvr], 1);
                    remax = one / fabs(VR[ii + ki * ldvr]);
                    cblas_dscal(n, remax, &VR[ki * ldvr], 1);

                } else {
                    /* Version 2: back-transform block of vectors with GEMM
                     * Zero out below vector */
                    for (k = ki + 1; k < n; k++) {
                        work[k + iv * n] = zero;
                    }
                    iscomplex[iv] = ip;
                    /* back-transform and normalization is done below */
                }

            } else {
                /* --------------------------------------------------------
                 * Complex right eigenvector.
                 *
                 * Initial solve
                 * [(T(ki-1,ki-1) T(ki-1,ki)) - (wr + i*wi)] * X = 0
                 * [(T(ki,ki-1)   T(ki,ki)  )              ]
                 */
                if (fabs(T[(ki - 1) + ki * ldt]) >= fabs(T[ki + (ki - 1) * ldt])) {
                    work[(ki - 1) + (iv - 1) * n] = one;
                    work[ki + iv * n] = wi / T[(ki - 1) + ki * ldt];
                } else {
                    work[(ki - 1) + (iv - 1) * n] = -wi / T[ki + (ki - 1) * ldt];
                    work[ki + iv * n] = one;
                }
                work[ki + (iv - 1) * n] = zero;
                work[(ki - 1) + iv * n] = zero;

                /* Form right-hand side */
                for (k = 0; k < ki - 1; k++) {
                    work[k + (iv - 1) * n] = -work[(ki - 1) + (iv - 1) * n] * T[k + (ki - 1) * ldt];
                    work[k + iv * n] = -work[ki + iv * n] * T[k + ki * ldt];
                }

                /* Solve upper quasi-triangular system:
                 * [T(0:ki-2, 0:ki-2) - (wr+i*wi)] * X = scale * (work + i*work2) */
                jnxt = ki - 2;
                for (j = ki - 2; j >= 0; j--) {
                    if (j > jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j - 1;
                    if (j > 0) {
                        if (T[j + (j - 1) * ldt] != zero) {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */
                        dlaln2(0, 1, 2, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + (iv - 1) * n], n, wr, wi,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale X(0,0) and X(0,1) to avoid overflow */
                        if (xnorm > one) {
                            if (work[j] > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                x[2] = x[2] / xnorm;  /* X(0,1) in column-major */
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(ki + 1, scale, &work[(iv - 1) * n], 1);
                            cblas_dscal(ki + 1, scale, &work[iv * n], 1);
                        }
                        work[j + (iv - 1) * n] = x[0];
                        work[j + iv * n] = x[2];

                        /* Update the right-hand side */
                        cblas_daxpy(j, -x[0], &T[j * ldt], 1, &work[(iv - 1) * n], 1);
                        cblas_daxpy(j, -x[2], &T[j * ldt], 1, &work[iv * n], 1);

                    } else {
                        /* 2-by-2 diagonal block */
                        dlaln2(0, 2, 2, smin, one, &T[(j - 1) + (j - 1) * ldt], ldt,
                               one, one, &work[(j - 1) + (iv - 1) * n], n, wr, wi,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale X to avoid overflow */
                        if (xnorm > one) {
                            beta = work[j - 1] > work[j] ? work[j - 1] : work[j];
                            if (beta > bignum / xnorm) {
                                rec = one / xnorm;
                                x[0] *= rec;
                                x[2] *= rec;
                                x[1] *= rec;
                                x[3] *= rec;
                                scale *= rec;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(ki + 1, scale, &work[(iv - 1) * n], 1);
                            cblas_dscal(ki + 1, scale, &work[iv * n], 1);
                        }
                        work[(j - 1) + (iv - 1) * n] = x[0];
                        work[j + (iv - 1) * n] = x[1];
                        work[(j - 1) + iv * n] = x[2];
                        work[j + iv * n] = x[3];

                        /* Update the right-hand side */
                        cblas_daxpy(j - 1, -x[0], &T[(j - 1) * ldt], 1, &work[(iv - 1) * n], 1);
                        cblas_daxpy(j - 1, -x[1], &T[j * ldt], 1, &work[(iv - 1) * n], 1);
                        cblas_daxpy(j - 1, -x[2], &T[(j - 1) * ldt], 1, &work[iv * n], 1);
                        cblas_daxpy(j - 1, -x[3], &T[j * ldt], 1, &work[iv * n], 1);
                    }
                }

                /* Copy the vector x or Q*x to VR and normalize */
                if (!over) {
                    /* No back-transform: copy x to VR and normalize */
                    cblas_dcopy(ki, &work[(iv - 1) * n], 1, &VR[(is - 1) * ldvr], 1);
                    cblas_dcopy(ki, &work[iv * n], 1, &VR[is * ldvr], 1);

                    emax = zero;
                    for (k = 0; k < ki; k++) {
                        double val = fabs(VR[k + (is - 1) * ldvr]) + fabs(VR[k + is * ldvr]);
                        if (val > emax) emax = val;
                    }
                    remax = one / emax;
                    cblas_dscal(ki, remax, &VR[(is - 1) * ldvr], 1);
                    cblas_dscal(ki, remax, &VR[is * ldvr], 1);

                    for (k = ki; k < n; k++) {
                        VR[k + (is - 1) * ldvr] = zero;
                        VR[k + is * ldvr] = zero;
                    }

                } else if (nb == 1) {
                    /* Version 1: back-transform each vector with GEMV, Q*x */
                    if (ki > 1) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, ki - 1,
                                    one, VR, ldvr, &work[(iv - 1) * n], 1,
                                    work[(ki - 1) + (iv - 1) * n], &VR[(ki - 1) * ldvr], 1);
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, ki - 1,
                                    one, VR, ldvr, &work[iv * n], 1,
                                    work[ki + iv * n], &VR[ki * ldvr], 1);
                    } else {
                        cblas_dscal(n, work[(ki - 1) + (iv - 1) * n], &VR[(ki - 1) * ldvr], 1);
                        cblas_dscal(n, work[ki + iv * n], &VR[ki * ldvr], 1);
                    }

                    emax = zero;
                    for (k = 0; k < n; k++) {
                        double val = fabs(VR[k + (ki - 1) * ldvr]) + fabs(VR[k + ki * ldvr]);
                        if (val > emax) emax = val;
                    }
                    remax = one / emax;
                    cblas_dscal(n, remax, &VR[(ki - 1) * ldvr], 1);
                    cblas_dscal(n, remax, &VR[ki * ldvr], 1);

                } else {
                    /* Version 2: back-transform block of vectors with GEMM
                     * Zero out below vector */
                    for (k = ki + 1; k < n; k++) {
                        work[k + (iv - 1) * n] = zero;
                        work[k + iv * n] = zero;
                    }
                    iscomplex[iv - 1] = -ip;
                    iscomplex[iv] = ip;
                    iv = iv - 1;
                    /* back-transform and normalization is done below */
                }
            }

            if (nb > 1) {
                /* --------------------------------------------------------
                 * Blocked version of back-transform
                 * For complex case, ki2 includes both vectors (ki-1 and ki)
                 */
                if (ip == 0) {
                    ki2 = ki;
                } else {
                    ki2 = ki - 1;
                }

                /* Columns iv:nb-1 of work are valid vectors.
                 * When the number of vectors stored reaches nb-1 or nb,
                 * or if this was last vector, do the GEMM */
                if ((iv <= 1) || (ki2 == 0)) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n, nb - iv, ki2 + nb - iv,
                                one, VR, ldvr,
                                &work[iv * n], n,
                                zero, &work[(nb + iv) * n], n);

                    /* Normalize vectors */
                    remax = one;  /* Initialize for compiler; reused for complex pairs */
                    for (k = iv; k < nb; k++) {
                        if (iscomplex[k] == 0) {
                            /* Real eigenvector */
                            ii = cblas_idamax(n, &work[(nb + k) * n], 1);
                            remax = one / fabs(work[ii + (nb + k) * n]);
                        } else if (iscomplex[k] == 1) {
                            /* First eigenvector of conjugate pair */
                            emax = zero;
                            for (ii = 0; ii < n; ii++) {
                                double val = fabs(work[ii + (nb + k) * n]) +
                                             fabs(work[ii + (nb + k + 1) * n]);
                                if (val > emax) emax = val;
                            }
                            remax = one / emax;
                            /* else if iscomplex[k] == -1:
                             * second eigenvector of conjugate pair
                             * reuse same remax as previous k */
                        }
                        cblas_dscal(n, remax, &work[(nb + k) * n], 1);
                    }
                    dlacpy("F", n, nb - iv, &work[(nb + iv) * n], n,
                           &VR[ki2 * ldvr], ldvr);
                    iv = nb - 1;
                } else {
                    iv = iv - 1;
                }
            }

            is = is - 1;
            if (ip != 0)
                is = is - 1;
        }
    }

    /* ============================================================
     * Compute left eigenvectors.
     */
    if (leftv) {
        /* IV is index of column in current block.
         * For complex left vector, uses IV for real part and IV+1 for complex part.
         * Non-blocked version always uses IV=1;
         * blocked version starts with IV=1, goes up to nb-1 or nb.
         * (Note the "0-th" column is used for 1-norms computed above.) */
        iv = 1;
        ip = 0;
        is = 0;

        for (ki = 0; ki < n; ki++) {
            if (ip == 1) {
                /* Previous iteration (ki-1) was first of conjugate pair,
                 * so this ki is second of conjugate pair; skip to next iteration */
                ip = -1;
                continue;
            } else if (ki == n - 1) {
                /* Last column, so this ki must be real eigenvalue */
                ip = 0;
            } else if (T[(ki + 1) + ki * ldt] == zero) {
                /* Zero on sub-diagonal, so this ki is real eigenvalue */
                ip = 0;
            } else {
                /* Non-zero on sub-diagonal, so this ki is first of conjugate pair */
                ip = 1;
            }

            if (somev) {
                if (!select[ki])
                    continue;
            }

            /* Compute the ki-th eigenvalue (wr, wi) */
            wr = T[ki + ki * ldt];
            wi = zero;
            if (ip != 0) {
                wi = sqrt(fabs(T[ki + (ki + 1) * ldt])) *
                     sqrt(fabs(T[(ki + 1) + ki * ldt]));
            }
            smin = ulp * (fabs(wr) + fabs(wi));
            if (smin < smlnum) smin = smlnum;

            if (ip == 0) {
                /* --------------------------------------------------------
                 * Real left eigenvector
                 */
                work[ki + iv * n] = one;

                /* Form right-hand side */
                for (k = ki + 1; k < n; k++) {
                    work[k + iv * n] = -T[ki + k * ldt];
                }

                /* Solve transposed quasi-triangular system:
                 * [T(ki+1:n-1, ki+1:n-1) - wr]^T * X = scale * work */
                vmax = one;
                vcrit = bignum;

                jnxt = ki + 1;
                for (j = ki + 1; j < n; j++) {
                    if (j < jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j + 1;
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] != zero) {
                            j2 = j + 1;
                            jnxt = j + 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */

                        /* Scale if necessary to avoid overflow */
                        if (work[j] > vcrit) {
                            rec = one / vmax;
                            cblas_dscal(n - ki, rec, &work[ki + iv * n], 1);
                            vmax = one;
                            vcrit = bignum;
                        }

                        work[j + iv * n] = work[j + iv * n] -
                            cblas_ddot(j - ki - 1, &T[(ki + 1) + j * ldt], 1,
                                       &work[(ki + 1) + iv * n], 1);

                        /* Solve [T(j,j) - wr]^T * X = work */
                        dlaln2(0, 1, 1, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + iv * n], n, wr, zero,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(n - ki, scale, &work[ki + iv * n], 1);
                        }
                        work[j + iv * n] = x[0];
                        vmax = fabs(work[j + iv * n]);
                        if (vmax > one) vmax = fabs(work[j + iv * n]);
                        vcrit = bignum / vmax;

                    } else {
                        /* 2-by-2 diagonal block */

                        /* Scale if necessary to avoid overflow */
                        beta = work[j] > work[j + 1] ? work[j] : work[j + 1];
                        if (beta > vcrit) {
                            rec = one / vmax;
                            cblas_dscal(n - ki, rec, &work[ki + iv * n], 1);
                            vmax = one;
                            vcrit = bignum;
                        }

                        work[j + iv * n] = work[j + iv * n] -
                            cblas_ddot(j - ki - 1, &T[(ki + 1) + j * ldt], 1,
                                       &work[(ki + 1) + iv * n], 1);

                        work[(j + 1) + iv * n] = work[(j + 1) + iv * n] -
                            cblas_ddot(j - ki - 1, &T[(ki + 1) + (j + 1) * ldt], 1,
                                       &work[(ki + 1) + iv * n], 1);

                        /* Solve
                         * [T(j,j)-wr    T(j,j+1)  ]^T * X = scale * (work1)
                         * [T(j+1,j)   T(j+1,j+1)-wr]              (work2) */
                        dlaln2(1, 2, 1, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + iv * n], n, wr, zero,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(n - ki, scale, &work[ki + iv * n], 1);
                        }
                        work[j + iv * n] = x[0];
                        work[(j + 1) + iv * n] = x[1];

                        double tmp = fabs(work[j + iv * n]);
                        if (fabs(work[(j + 1) + iv * n]) > tmp)
                            tmp = fabs(work[(j + 1) + iv * n]);
                        if (tmp > vmax) vmax = tmp;
                        vcrit = bignum / vmax;
                    }
                }

                /* Copy the vector x or Q*x to VL and normalize */
                if (!over) {
                    /* No back-transform: copy x to VL and normalize */
                    cblas_dcopy(n - ki, &work[ki + iv * n], 1, &VL[ki + is * ldvl], 1);

                    ii = cblas_idamax(n - ki, &VL[ki + is * ldvl], 1) + ki;
                    remax = one / fabs(VL[ii + is * ldvl]);
                    cblas_dscal(n - ki, remax, &VL[ki + is * ldvl], 1);

                    for (k = 0; k < ki; k++) {
                        VL[k + is * ldvl] = zero;
                    }

                } else if (nb == 1) {
                    /* Version 1: back-transform each vector with GEMV, Q*x */
                    if (ki < n - 1) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n - ki - 1,
                                    one, &VL[(ki + 1) * ldvl], ldvl,
                                    &work[(ki + 1) + iv * n], 1,
                                    work[ki + iv * n], &VL[ki * ldvl], 1);
                    }

                    ii = cblas_idamax(n, &VL[ki * ldvl], 1);
                    remax = one / fabs(VL[ii + ki * ldvl]);
                    cblas_dscal(n, remax, &VL[ki * ldvl], 1);

                } else {
                    /* Version 2: back-transform block of vectors with GEMM
                     * Zero out above vector */
                    for (k = 0; k < ki; k++) {
                        work[k + iv * n] = zero;
                    }
                    iscomplex[iv] = ip;
                    /* back-transform and normalization is done below */
                }

            } else {
                /* --------------------------------------------------------
                 * Complex left eigenvector.
                 *
                 * Initial solve:
                 * [(T(ki,ki)    T(ki,ki+1)  )^T - (wr - i*wi)] * X = 0
                 * [(T(ki+1,ki) T(ki+1,ki+1))]
                 */
                if (fabs(T[ki + (ki + 1) * ldt]) >= fabs(T[(ki + 1) + ki * ldt])) {
                    work[ki + iv * n] = wi / T[ki + (ki + 1) * ldt];
                    work[(ki + 1) + (iv + 1) * n] = one;
                } else {
                    work[ki + iv * n] = one;
                    work[(ki + 1) + (iv + 1) * n] = -wi / T[(ki + 1) + ki * ldt];
                }
                work[(ki + 1) + iv * n] = zero;
                work[ki + (iv + 1) * n] = zero;

                /* Form right-hand side */
                for (k = ki + 2; k < n; k++) {
                    work[k + iv * n] = -work[ki + iv * n] * T[ki + k * ldt];
                    work[k + (iv + 1) * n] = -work[(ki + 1) + (iv + 1) * n] * T[(ki + 1) + k * ldt];
                }

                /* Solve transposed quasi-triangular system:
                 * [T(ki+2:n-1, ki+2:n-1)^T - (wr - i*wi)] * X = work1 + i*work2 */
                vmax = one;
                vcrit = bignum;

                jnxt = ki + 2;
                for (j = ki + 2; j < n; j++) {
                    if (j < jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j + 1;
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] != zero) {
                            j2 = j + 1;
                            jnxt = j + 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */

                        /* Scale if necessary to avoid overflow */
                        if (work[j] > vcrit) {
                            rec = one / vmax;
                            cblas_dscal(n - ki, rec, &work[ki + iv * n], 1);
                            cblas_dscal(n - ki, rec, &work[ki + (iv + 1) * n], 1);
                            vmax = one;
                            vcrit = bignum;
                        }

                        work[j + iv * n] = work[j + iv * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                       &work[(ki + 2) + iv * n], 1);
                        work[j + (iv + 1) * n] = work[j + (iv + 1) * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                       &work[(ki + 2) + (iv + 1) * n], 1);

                        /* Solve [T(j,j) - (wr - i*wi)] * (X11 + i*X12) = wk + i*wk2 */
                        dlaln2(0, 1, 2, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + iv * n], n, wr, -wi,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(n - ki, scale, &work[ki + iv * n], 1);
                            cblas_dscal(n - ki, scale, &work[ki + (iv + 1) * n], 1);
                        }
                        work[j + iv * n] = x[0];
                        work[j + (iv + 1) * n] = x[2];
                        vmax = fabs(work[j + iv * n]);
                        if (fabs(work[j + (iv + 1) * n]) > vmax)
                            vmax = fabs(work[j + (iv + 1) * n]);
                        vcrit = bignum / vmax;

                    } else {
                        /* 2-by-2 diagonal block */

                        /* Scale if necessary to avoid overflow */
                        beta = work[j] > work[j + 1] ? work[j] : work[j + 1];
                        if (beta > vcrit) {
                            rec = one / vmax;
                            cblas_dscal(n - ki, rec, &work[ki + iv * n], 1);
                            cblas_dscal(n - ki, rec, &work[ki + (iv + 1) * n], 1);
                            vmax = one;
                            vcrit = bignum;
                        }

                        work[j + iv * n] = work[j + iv * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                       &work[(ki + 2) + iv * n], 1);
                        work[j + (iv + 1) * n] = work[j + (iv + 1) * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                       &work[(ki + 2) + (iv + 1) * n], 1);
                        work[(j + 1) + iv * n] = work[(j + 1) + iv * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + (j + 1) * ldt], 1,
                                       &work[(ki + 2) + iv * n], 1);
                        work[(j + 1) + (iv + 1) * n] = work[(j + 1) + (iv + 1) * n] -
                            cblas_ddot(j - ki - 2, &T[(ki + 2) + (j + 1) * ldt], 1,
                                       &work[(ki + 2) + (iv + 1) * n], 1);

                        /* Solve 2-by-2 complex linear equation
                         * [(T(j,j)   T(j,j+1)  )^T - (wr - i*wi)*I] * X = scale*B
                         * [(T(j+1,j) T(j+1,j+1))]                     */
                        dlaln2(1, 2, 2, smin, one, &T[j + j * ldt], ldt,
                               one, one, &work[j + iv * n], n, wr, -wi,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != one) {
                            cblas_dscal(n - ki, scale, &work[ki + iv * n], 1);
                            cblas_dscal(n - ki, scale, &work[ki + (iv + 1) * n], 1);
                        }
                        work[j + iv * n] = x[0];
                        work[j + (iv + 1) * n] = x[2];
                        work[(j + 1) + iv * n] = x[1];
                        work[(j + 1) + (iv + 1) * n] = x[3];
                        vmax = fabs(x[0]);
                        if (fabs(x[2]) > vmax) vmax = fabs(x[2]);
                        if (fabs(x[1]) > vmax) vmax = fabs(x[1]);
                        if (fabs(x[3]) > vmax) vmax = fabs(x[3]);
                        vcrit = bignum / vmax;
                    }
                }

                /* Copy the vector x or Q*x to VL and normalize */
                if (!over) {
                    /* No back-transform: copy x to VL and normalize */
                    cblas_dcopy(n - ki, &work[ki + iv * n], 1, &VL[ki + is * ldvl], 1);
                    cblas_dcopy(n - ki, &work[ki + (iv + 1) * n], 1, &VL[ki + (is + 1) * ldvl], 1);

                    emax = zero;
                    for (k = ki; k < n; k++) {
                        double val = fabs(VL[k + is * ldvl]) + fabs(VL[k + (is + 1) * ldvl]);
                        if (val > emax) emax = val;
                    }
                    remax = one / emax;
                    cblas_dscal(n - ki, remax, &VL[ki + is * ldvl], 1);
                    cblas_dscal(n - ki, remax, &VL[ki + (is + 1) * ldvl], 1);

                    for (k = 0; k < ki; k++) {
                        VL[k + is * ldvl] = zero;
                        VL[k + (is + 1) * ldvl] = zero;
                    }

                } else if (nb == 1) {
                    /* Version 1: back-transform each vector with GEMV, Q*x */
                    if (ki < n - 2) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n - ki - 2,
                                    one, &VL[(ki + 2) * ldvl], ldvl,
                                    &work[(ki + 2) + iv * n], 1,
                                    work[ki + iv * n], &VL[ki * ldvl], 1);
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n - ki - 2,
                                    one, &VL[(ki + 2) * ldvl], ldvl,
                                    &work[(ki + 2) + (iv + 1) * n], 1,
                                    work[(ki + 1) + (iv + 1) * n], &VL[(ki + 1) * ldvl], 1);
                    } else {
                        cblas_dscal(n, work[ki + iv * n], &VL[ki * ldvl], 1);
                        cblas_dscal(n, work[(ki + 1) + (iv + 1) * n], &VL[(ki + 1) * ldvl], 1);
                    }

                    emax = zero;
                    for (k = 0; k < n; k++) {
                        double val = fabs(VL[k + ki * ldvl]) + fabs(VL[k + (ki + 1) * ldvl]);
                        if (val > emax) emax = val;
                    }
                    remax = one / emax;
                    cblas_dscal(n, remax, &VL[ki * ldvl], 1);
                    cblas_dscal(n, remax, &VL[(ki + 1) * ldvl], 1);

                } else {
                    /* Version 2: back-transform block of vectors with GEMM
                     * Zero out above vector */
                    for (k = 0; k < ki; k++) {
                        work[k + iv * n] = zero;
                        work[k + (iv + 1) * n] = zero;
                    }
                    iscomplex[iv] = ip;
                    iscomplex[iv + 1] = -ip;
                    iv = iv + 1;
                    /* back-transform and normalization is done below */
                }
            }

            if (nb > 1) {
                /* --------------------------------------------------------
                 * Blocked version of back-transform
                 * For complex case, ki2 includes both vectors (ki and ki+1)
                 */
                if (ip == 0) {
                    ki2 = ki;
                } else {
                    ki2 = ki + 1;
                }

                /* Columns 1:iv of work are valid vectors.
                 * When the number of vectors stored reaches nb-1 or nb,
                 * or if this was last vector, do the GEMM */
                if ((iv >= nb - 1) || (ki2 == n - 1)) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n, iv, n - ki2 + iv - 1,
                                one, &VL[(ki2 - iv + 1) * ldvl], ldvl,
                                &work[(ki2 - iv + 1) + 1 * n], n,
                                zero, &work[(nb + 1) * n], n);

                    /* Normalize vectors */
                    remax = one;  /* Initialize for compiler; overwritten before use */
                    for (k = 1; k <= iv; k++) {
                        if (iscomplex[k] == 0) {
                            /* Real eigenvector */
                            ii = cblas_idamax(n, &work[(nb + k) * n], 1);
                            remax = one / fabs(work[ii + (nb + k) * n]);
                        } else if (iscomplex[k] == 1) {
                            /* First eigenvector of conjugate pair */
                            emax = zero;
                            for (ii = 0; ii < n; ii++) {
                                double val = fabs(work[ii + (nb + k) * n]) +
                                             fabs(work[ii + (nb + k + 1) * n]);
                                if (val > emax) emax = val;
                            }
                            remax = one / emax;
                            /* else if iscomplex[k] == -1:
                             * second eigenvector of conjugate pair
                             * reuse same remax as previous k */
                        }
                        cblas_dscal(n, remax, &work[(nb + k) * n], 1);
                    }
                    dlacpy("F", n, iv, &work[(nb + 1) * n], n,
                           &VL[(ki2 - iv + 1) * ldvl], ldvl);
                    iv = 1;
                } else {
                    iv = iv + 1;
                }
            }

            is = is + 1;
            if (ip != 0)
                is = is + 1;
        }
    }
}
