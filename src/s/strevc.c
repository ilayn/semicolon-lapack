/**
 * @file strevc.c
 * @brief STREVC computes eigenvectors of a real upper quasi-triangular matrix.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <float.h>

/**
 * STREVC computes some or all of the right and/or left eigenvectors of
 * a real upper quasi-triangular matrix T.
 * Matrices of this type are produced by the Schur factorization of
 * a real general matrix:  A = Q*T*Q**T, as computed by SHSEQR.
 *
 * The right eigenvector x and the left eigenvector y of T corresponding
 * to an eigenvalue w are defined by:
 *
 *    T*x = w*x,     (y**H)*T = w*(y**H)
 *
 * where y**H denotes the conjugate transpose of y.
 * The eigenvalues are not input to this routine, but are read directly
 * from the diagonal blocks of T.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 * input matrix.  If Q is the orthogonal factor that reduces a matrix
 * A to Schur form T, then Q*X and Q*Y are the matrices of right and
 * left eigenvectors of A.
 *
 * @param[in] side    'R': compute right eigenvectors only;
 *                    'L': compute left eigenvectors only;
 *                    'B': compute both right and left eigenvectors.
 * @param[in] howmny  'A': compute all right and/or left eigenvectors;
 *                    'B': compute all, backtransformed by VR and/or VL;
 *                    'S': compute selected eigenvectors (as indicated by select).
 * @param[in,out] select  Integer array, dimension (n).
 *                    If howmny = 'S', select specifies which eigenvectors to compute.
 *                    Nonzero = selected. For complex pairs, if either is selected,
 *                    both are computed.
 * @param[in] n       The order of the matrix T. n >= 0.
 * @param[in] T       Double precision array, dimension (ldt, n).
 *                    The upper quasi-triangular matrix T in Schur canonical form.
 * @param[in] ldt     The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] VL  Double precision array, dimension (ldvl, mm).
 *                    On entry, if howmny = 'B', must contain an n-by-n matrix Q.
 *                    On exit, contains the left eigenvectors.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1, and if
 *                    side = 'L' or 'B', ldvl >= n.
 * @param[in,out] VR  Double precision array, dimension (ldvr, mm).
 *                    On entry, if howmny = 'B', must contain an n-by-n matrix Q.
 *                    On exit, contains the right eigenvectors.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1, and if
 *                    side = 'R' or 'B', ldvr >= n.
 * @param[in] mm      The number of columns in VL and/or VR. mm >= m.
 * @param[out] m      The number of columns actually used to store eigenvectors.
 * @param[out] work   Double precision array, dimension (3*n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void strevc(const char* side, const char* howmny, INT* select, const INT n,
            const f32* T, const INT ldt, f32* VL, const INT ldvl,
            f32* VR, const INT ldvr, const INT mm, INT* m,
            f32* work, INT* info)
{
    /* Local scalars */
    INT allv, bothv, leftv, over, pair, rightv, somev;
    INT i, ierr, ii, ip, is, j, j1, j2, jnxt, k, ki, n2;
    f32 beta, bignum, emax, rec, remax, scale;
    f32 smin, smlnum, ulp, unfl, vcrit, vmax, wi, wr, xnorm;

    /* Local array */
    f32 x[4];  /* 2x2 stored column-major: x[0]=X(1,1), x[1]=X(2,1), x[2]=X(1,2), x[3]=X(2,2) */

    /* Decode and test the input parameters */
    bothv = (side[0] == 'B' || side[0] == 'b');
    rightv = (side[0] == 'R' || side[0] == 'r') || bothv;
    leftv = (side[0] == 'L' || side[0] == 'l') || bothv;

    allv = (howmny[0] == 'A' || howmny[0] == 'a');
    over = (howmny[0] == 'B' || howmny[0] == 'b');
    somev = (howmny[0] == 'S' || howmny[0] == 's');

    *info = 0;
    if (!rightv && !leftv) {
        *info = -1;
    } else if (!allv && !over && !somev) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        *info = -8;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        *info = -10;
    } else {
        /* Set M to the number of columns required to store the selected
         * eigenvectors, standardize the array SELECT if necessary, and
         * test MM. */
        if (somev) {
            *m = 0;
            pair = 0;
            for (j = 0; j < n; j++) {
                if (pair) {
                    pair = 0;
                    select[j] = 0;
                } else {
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] == 0.0f) {
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
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Set the constants to control overflow */
    unfl = FLT_MIN;
    ulp = FLT_EPSILON;
    smlnum = unfl * ((f32)n / ulp);
    bignum = (1.0f - ulp) / smlnum;

    /* Compute 1-norm of each column of strictly upper triangular
     * part of T to control overflow in triangular solver. */
    work[0] = 0.0f;
    for (j = 1; j < n; j++) {
        work[j] = 0.0f;
        for (i = 0; i < j; i++) {
            work[j] += fabsf(T[i + j * ldt]);
        }
    }

    /* Index IP is used to specify the real or complex eigenvalue:
     *   IP = 0, real eigenvalue,
     *        1, first of conjugate complex pair: (wr,wi)
     *       -1, second of conjugate complex pair: (wr,wi) */
    n2 = 2 * n;

    if (rightv) {
        /* Compute right eigenvectors. */
        ip = 0;
        is = *m - 1;  /* 0-based column index */

        for (ki = n - 1; ki >= 0; ki--) {
            if (ip == 1)
                goto L130;
            if (ki > 0 && T[ki + (ki - 1) * ldt] != 0.0f)
                ip = -1;

            if (somev) {
                if (ip == 0) {
                    if (!select[ki])
                        goto L130;
                } else {
                    if (!select[ki - 1])
                        goto L130;
                }
            }

            /* Compute the KI-th eigenvalue (WR,WI). */
            wr = T[ki + ki * ldt];
            wi = 0.0f;
            if (ip != 0)
                wi = sqrtf(fabsf(T[ki + (ki - 1) * ldt])) *
                     sqrtf(fabsf(T[(ki - 1) + ki * ldt]));
            smin = ulp * (fabsf(wr) + fabsf(wi));
            if (smin < smlnum) smin = smlnum;

            if (ip == 0) {
                /* Real right eigenvector */
                work[ki + n] = 1.0f;

                /* Form right-hand side */
                for (k = 0; k < ki; k++) {
                    work[k + n] = -T[k + ki * ldt];
                }

                /* Solve the upper quasi-triangular system:
                 *   (T(0:ki-1,0:ki-1) - WR)*X = SCALE*WORK. */
                jnxt = ki - 1;
                for (j = ki - 1; j >= 0; j--) {
                    if (j > jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j - 1;
                    if (j > 0) {
                        if (T[j + (j - 1) * ldt] != 0.0f) {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */
                        slaln2(0, 1, 1, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr,
                               0.0f, x, 2, &scale, &xnorm, &ierr);

                        /* Scale X(0,0) to avoid overflow when updating
                         * the right-hand side. */
                        if (xnorm > 1.0f) {
                            if (work[j] > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != 1.0f)
                            cblas_sscal(ki + 1, scale, &work[n], 1);
                        work[j + n] = x[0];

                        /* Update right-hand side */
                        cblas_saxpy(j, -x[0], &T[j * ldt], 1, &work[n], 1);

                    } else {
                        /* 2-by-2 diagonal block */
                        slaln2(0, 2, 1, smin, 1.0f,
                               &T[(j - 1) + (j - 1) * ldt], ldt, 1.0f, 1.0f,
                               &work[(j - 1) + n], n, wr, 0.0f, x, 2,
                               &scale, &xnorm, &ierr);

                        /* Scale X(0,0) and X(1,0) to avoid overflow when
                         * updating the right-hand side. */
                        if (xnorm > 1.0f) {
                            beta = work[j - 1];
                            if (work[j] > beta) beta = work[j];
                            if (beta > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                x[1] = x[1] / xnorm;
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != 1.0f)
                            cblas_sscal(ki + 1, scale, &work[n], 1);
                        work[(j - 1) + n] = x[0];
                        work[j + n] = x[1];

                        /* Update right-hand side */
                        cblas_saxpy(j - 1, -x[0], &T[(j - 1) * ldt], 1, &work[n], 1);
                        cblas_saxpy(j - 1, -x[1], &T[j * ldt], 1, &work[n], 1);
                    }
                }

                /* Copy the vector x or Q*x to VR and normalize. */
                if (!over) {
                    cblas_scopy(ki + 1, &work[n], 1, &VR[is * ldvr], 1);

                    ii = cblas_isamax(ki + 1, &VR[is * ldvr], 1);
                    remax = 1.0f / fabsf(VR[ii + is * ldvr]);
                    cblas_sscal(ki + 1, remax, &VR[is * ldvr], 1);

                    for (k = ki + 1; k < n; k++) {
                        VR[k + is * ldvr] = 0.0f;
                    }
                } else {
                    if (ki > 0)
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, ki, 1.0f, VR, ldvr,
                                    &work[n], 1, work[ki + n], &VR[ki * ldvr], 1);

                    ii = cblas_isamax(n, &VR[ki * ldvr], 1);
                    remax = 1.0f / fabsf(VR[ii + ki * ldvr]);
                    cblas_sscal(n, remax, &VR[ki * ldvr], 1);
                }

            } else {
                /* Complex right eigenvector.
                 *
                 * Initial solve
                 *   [ (T(ki-1,ki-1) T(ki-1,ki) ) - (WR + I* WI)]*X = 0.
                 *   [ (T(ki,ki-1)   T(ki,ki)   )               ] */
                if (fabsf(T[(ki - 1) + ki * ldt]) >= fabsf(T[ki + (ki - 1) * ldt])) {
                    work[(ki - 1) + n] = 1.0f;
                    work[ki + n2] = wi / T[(ki - 1) + ki * ldt];
                } else {
                    work[(ki - 1) + n] = -wi / T[ki + (ki - 1) * ldt];
                    work[ki + n2] = 1.0f;
                }
                work[ki + n] = 0.0f;
                work[(ki - 1) + n2] = 0.0f;

                /* Form right-hand side */
                for (k = 0; k < ki - 1; k++) {
                    work[k + n] = -work[(ki - 1) + n] * T[k + (ki - 1) * ldt];
                    work[k + n2] = -work[ki + n2] * T[k + ki * ldt];
                }

                /* Solve upper quasi-triangular system:
                 * (T(0:ki-2,0:ki-2) - (WR+i*WI))*X = SCALE*(WORK+i*WORK2) */
                jnxt = ki - 2;
                for (j = ki - 2; j >= 0; j--) {
                    if (j > jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j - 1;
                    if (j > 0) {
                        if (T[j + (j - 1) * ldt] != 0.0f) {
                            j1 = j - 1;
                            jnxt = j - 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block */
                        slaln2(0, 1, 2, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr, wi,
                               x, 2, &scale, &xnorm, &ierr);

                        /* Scale X(0,0) and X(0,1) to avoid overflow when
                         * updating the right-hand side. */
                        if (xnorm > 1.0f) {
                            if (work[j] > bignum / xnorm) {
                                x[0] = x[0] / xnorm;
                                x[2] = x[2] / xnorm;
                                scale = scale / xnorm;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != 1.0f) {
                            cblas_sscal(ki + 1, scale, &work[n], 1);
                            cblas_sscal(ki + 1, scale, &work[n2], 1);
                        }
                        work[j + n] = x[0];
                        work[j + n2] = x[2];

                        /* Update the right-hand side */
                        cblas_saxpy(j, -x[0], &T[j * ldt], 1, &work[n], 1);
                        cblas_saxpy(j, -x[2], &T[j * ldt], 1, &work[n2], 1);

                    } else {
                        /* 2-by-2 diagonal block */
                        slaln2(0, 2, 2, smin, 1.0f,
                               &T[(j - 1) + (j - 1) * ldt], ldt, 1.0f, 1.0f,
                               &work[(j - 1) + n], n, wr, wi, x, 2,
                               &scale, &xnorm, &ierr);

                        /* Scale X to avoid overflow when updating
                         * the right-hand side. */
                        if (xnorm > 1.0f) {
                            beta = work[j - 1];
                            if (work[j] > beta) beta = work[j];
                            if (beta > bignum / xnorm) {
                                rec = 1.0f / xnorm;
                                x[0] = x[0] * rec;
                                x[2] = x[2] * rec;
                                x[1] = x[1] * rec;
                                x[3] = x[3] * rec;
                                scale = scale * rec;
                            }
                        }

                        /* Scale if necessary */
                        if (scale != 1.0f) {
                            cblas_sscal(ki + 1, scale, &work[n], 1);
                            cblas_sscal(ki + 1, scale, &work[n2], 1);
                        }
                        work[(j - 1) + n] = x[0];
                        work[j + n] = x[1];
                        work[(j - 1) + n2] = x[2];
                        work[j + n2] = x[3];

                        /* Update the right-hand side */
                        cblas_saxpy(j - 1, -x[0], &T[(j - 1) * ldt], 1, &work[n], 1);
                        cblas_saxpy(j - 1, -x[1], &T[j * ldt], 1, &work[n], 1);
                        cblas_saxpy(j - 1, -x[2], &T[(j - 1) * ldt], 1, &work[n2], 1);
                        cblas_saxpy(j - 1, -x[3], &T[j * ldt], 1, &work[n2], 1);
                    }
                }

                /* Copy the vector x or Q*x to VR and normalize. */
                if (!over) {
                    cblas_scopy(ki + 1, &work[n], 1, &VR[(is - 1) * ldvr], 1);
                    cblas_scopy(ki + 1, &work[n2], 1, &VR[is * ldvr], 1);

                    emax = 0.0f;
                    for (k = 0; k <= ki; k++) {
                        f32 temp = fabsf(VR[k + (is - 1) * ldvr]) + fabsf(VR[k + is * ldvr]);
                        if (temp > emax) emax = temp;
                    }

                    remax = 1.0f / emax;
                    cblas_sscal(ki + 1, remax, &VR[(is - 1) * ldvr], 1);
                    cblas_sscal(ki + 1, remax, &VR[is * ldvr], 1);

                    for (k = ki + 1; k < n; k++) {
                        VR[k + (is - 1) * ldvr] = 0.0f;
                        VR[k + is * ldvr] = 0.0f;
                    }

                } else {
                    if (ki > 1) {
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, ki - 1, 1.0f, VR, ldvr,
                                    &work[n], 1, work[(ki - 1) + n],
                                    &VR[(ki - 1) * ldvr], 1);
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, ki - 1, 1.0f, VR, ldvr,
                                    &work[n2], 1, work[ki + n2],
                                    &VR[ki * ldvr], 1);
                    } else {
                        cblas_sscal(n, work[(ki - 1) + n], &VR[(ki - 1) * ldvr], 1);
                        cblas_sscal(n, work[ki + n2], &VR[ki * ldvr], 1);
                    }

                    emax = 0.0f;
                    for (k = 0; k < n; k++) {
                        f32 temp = fabsf(VR[k + (ki - 1) * ldvr]) + fabsf(VR[k + ki * ldvr]);
                        if (temp > emax) emax = temp;
                    }
                    remax = 1.0f / emax;
                    cblas_sscal(n, remax, &VR[(ki - 1) * ldvr], 1);
                    cblas_sscal(n, remax, &VR[ki * ldvr], 1);
                }
            }

            is--;
            if (ip != 0)
                is--;
L130:
            if (ip == 1)
                ip = 0;
            if (ip == -1)
                ip = 1;
        }
    }

    if (leftv) {
        /* Compute left eigenvectors. */
        ip = 0;
        is = 0;  /* 0-based column index */

        for (ki = 0; ki < n; ki++) {
            if (ip == -1)
                goto L250;
            if (ki < n - 1 && T[(ki + 1) + ki * ldt] != 0.0f)
                ip = 1;

            if (somev) {
                if (!select[ki])
                    goto L250;
            }

            /* Compute the KI-th eigenvalue (WR,WI). */
            wr = T[ki + ki * ldt];
            wi = 0.0f;
            if (ip != 0)
                wi = sqrtf(fabsf(T[ki + (ki + 1) * ldt])) *
                     sqrtf(fabsf(T[(ki + 1) + ki * ldt]));
            smin = ulp * (fabsf(wr) + fabsf(wi));
            if (smin < smlnum) smin = smlnum;

            if (ip == 0) {
                /* Real left eigenvector. */
                work[ki + n] = 1.0f;

                /* Form right-hand side */
                for (k = ki + 1; k < n; k++) {
                    work[k + n] = -T[ki + k * ldt];
                }

                /* Solve the quasi-triangular system:
                 *   (T(ki+1:n-1,ki+1:n-1) - WR)**T*X = SCALE*WORK */
                vmax = 1.0f;
                vcrit = bignum;

                jnxt = ki + 1;
                for (j = ki + 1; j < n; j++) {
                    if (j < jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j + 1;
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] != 0.0f) {
                            j2 = j + 1;
                            jnxt = j + 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block
                         *
                         * Scale if necessary to avoid overflow when forming
                         * the right-hand side. */
                        if (work[j] > vcrit) {
                            rec = 1.0f / vmax;
                            cblas_sscal(n - ki, rec, &work[ki + n], 1);
                        }

                        work[j + n] -= cblas_sdot(j - ki - 1, &T[(ki + 1) + j * ldt], 1,
                                                  &work[(ki + 1) + n], 1);

                        /* Solve (T(J,J)-WR)**T*X = WORK */
                        slaln2(0, 1, 1, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr,
                               0.0f, x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != 1.0f)
                            cblas_sscal(n - ki, scale, &work[ki + n], 1);
                        work[j + n] = x[0];
                        vmax = fabsf(work[j + n]);
                        vcrit = bignum / vmax;

                    } else {
                        /* 2-by-2 diagonal block
                         *
                         * Scale if necessary to avoid overflow when forming
                         * the right-hand side. */
                        beta = work[j];
                        if (work[j + 1] > beta) beta = work[j + 1];
                        if (beta > vcrit) {
                            rec = 1.0f / vmax;
                            cblas_sscal(n - ki, rec, &work[ki + n], 1);
                        }

                        work[j + n] -= cblas_sdot(j - ki - 1, &T[(ki + 1) + j * ldt], 1,
                                                  &work[(ki + 1) + n], 1);

                        work[(j + 1) + n] -= cblas_sdot(j - ki - 1, &T[(ki + 1) + (j + 1) * ldt], 1,
                                                        &work[(ki + 1) + n], 1);

                        /* Solve
                         *   [T(J,J)-WR   T(J,J+1)     ]**T * X = SCALE*( WORK1 )
                         *   [T(J+1,J)    T(J+1,J+1)-WR]                ( WORK2 ) */
                        slaln2(1, 2, 1, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr,
                               0.0f, x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != 1.0f)
                            cblas_sscal(n - ki, scale, &work[ki + n], 1);
                        work[j + n] = x[0];
                        work[(j + 1) + n] = x[1];

                        vmax = fabsf(work[j + n]);
                        if (fabsf(work[(j + 1) + n]) > vmax) vmax = fabsf(work[(j + 1) + n]);
                        vcrit = bignum / vmax;
                    }
                }

                /* Copy the vector x or Q*x to VL and normalize. */
                if (!over) {
                    cblas_scopy(n - ki, &work[ki + n], 1, &VL[ki + is * ldvl], 1);

                    ii = cblas_isamax(n - ki, &VL[ki + is * ldvl], 1) + ki;
                    remax = 1.0f / fabsf(VL[ii + is * ldvl]);
                    cblas_sscal(n - ki, remax, &VL[ki + is * ldvl], 1);

                    for (k = 0; k < ki; k++) {
                        VL[k + is * ldvl] = 0.0f;
                    }

                } else {
                    if (ki < n - 1)
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, n - ki - 1, 1.0f,
                                    &VL[(ki + 1) * ldvl], ldvl,
                                    &work[(ki + 1) + n], 1, work[ki + n],
                                    &VL[ki * ldvl], 1);

                    ii = cblas_isamax(n, &VL[ki * ldvl], 1);
                    remax = 1.0f / fabsf(VL[ii + ki * ldvl]);
                    cblas_sscal(n, remax, &VL[ki * ldvl], 1);
                }

            } else {
                /* Complex left eigenvector.
                 *
                 *  Initial solve:
                 *    ((T(KI,KI)    T(KI,KI+1) )**T - (WR - I* WI))*X = 0.
                 *    ((T(KI+1,KI) T(KI+1,KI+1))                ) */
                if (fabsf(T[ki + (ki + 1) * ldt]) >= fabsf(T[(ki + 1) + ki * ldt])) {
                    work[ki + n] = wi / T[ki + (ki + 1) * ldt];
                    work[(ki + 1) + n2] = 1.0f;
                } else {
                    work[ki + n] = 1.0f;
                    work[(ki + 1) + n2] = -wi / T[(ki + 1) + ki * ldt];
                }
                work[(ki + 1) + n] = 0.0f;
                work[ki + n2] = 0.0f;

                /* Form right-hand side */
                for (k = ki + 2; k < n; k++) {
                    work[k + n] = -work[ki + n] * T[ki + k * ldt];
                    work[k + n2] = -work[(ki + 1) + n2] * T[(ki + 1) + k * ldt];
                }

                /* Solve complex quasi-triangular system:
                 * ( T(KI+2:N-1,KI+2:N-1) - (WR-i*WI) )*X = WORK1+i*WORK2 */
                vmax = 1.0f;
                vcrit = bignum;

                jnxt = ki + 2;
                for (j = ki + 2; j < n; j++) {
                    if (j < jnxt)
                        continue;
                    j1 = j;
                    j2 = j;
                    jnxt = j + 1;
                    if (j < n - 1) {
                        if (T[(j + 1) + j * ldt] != 0.0f) {
                            j2 = j + 1;
                            jnxt = j + 2;
                        }
                    }

                    if (j1 == j2) {
                        /* 1-by-1 diagonal block
                         *
                         * Scale if necessary to avoid overflow when
                         * forming the right-hand side elements. */
                        if (work[j] > vcrit) {
                            rec = 1.0f / vmax;
                            cblas_sscal(n - ki, rec, &work[ki + n], 1);
                            cblas_sscal(n - ki, rec, &work[ki + n2], 1);
                        }

                        work[j + n] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                                  &work[(ki + 2) + n], 1);
                        work[j + n2] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                                   &work[(ki + 2) + n2], 1);

                        /* Solve (T(J,J)-(WR-i*WI))*(X11+i*X12)= WK+I*WK2 */
                        slaln2(0, 1, 2, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr,
                               -wi, x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != 1.0f) {
                            cblas_sscal(n - ki, scale, &work[ki + n], 1);
                            cblas_sscal(n - ki, scale, &work[ki + n2], 1);
                        }
                        work[j + n] = x[0];
                        work[j + n2] = x[2];
                        vmax = fabsf(work[j + n]);
                        if (fabsf(work[j + n2]) > vmax) vmax = fabsf(work[j + n2]);
                        vcrit = bignum / vmax;

                    } else {
                        /* 2-by-2 diagonal block
                         *
                         * Scale if necessary to avoid overflow when forming
                         * the right-hand side elements. */
                        beta = work[j];
                        if (work[j + 1] > beta) beta = work[j + 1];
                        if (beta > vcrit) {
                            rec = 1.0f / vmax;
                            cblas_sscal(n - ki, rec, &work[ki + n], 1);
                            cblas_sscal(n - ki, rec, &work[ki + n2], 1);
                        }

                        work[j + n] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                                  &work[(ki + 2) + n], 1);

                        work[j + n2] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + j * ldt], 1,
                                                   &work[(ki + 2) + n2], 1);

                        work[(j + 1) + n] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + (j + 1) * ldt], 1,
                                                        &work[(ki + 2) + n], 1);

                        work[(j + 1) + n2] -= cblas_sdot(j - ki - 2, &T[(ki + 2) + (j + 1) * ldt], 1,
                                                         &work[(ki + 2) + n2], 1);

                        /* Solve 2-by-2 complex linear equation
                         *   ([T(j,j)   T(j,j+1)  ]**T-(wr-i*wi)*I)*X = SCALE*B
                         *   ([T(j+1,j) T(j+1,j+1)]               ) */
                        slaln2(1, 2, 2, smin, 1.0f, &T[j + j * ldt],
                               ldt, 1.0f, 1.0f, &work[j + n], n, wr,
                               -wi, x, 2, &scale, &xnorm, &ierr);

                        /* Scale if necessary */
                        if (scale != 1.0f) {
                            cblas_sscal(n - ki, scale, &work[ki + n], 1);
                            cblas_sscal(n - ki, scale, &work[ki + n2], 1);
                        }
                        work[j + n] = x[0];
                        work[j + n2] = x[2];
                        work[(j + 1) + n] = x[1];
                        work[(j + 1) + n2] = x[3];
                        vmax = fabsf(x[0]);
                        if (fabsf(x[2]) > vmax) vmax = fabsf(x[2]);
                        if (fabsf(x[1]) > vmax) vmax = fabsf(x[1]);
                        if (fabsf(x[3]) > vmax) vmax = fabsf(x[3]);
                        vcrit = bignum / vmax;
                    }
                }

                /* Copy the vector x or Q*x to VL and normalize. */
                if (!over) {
                    cblas_scopy(n - ki, &work[ki + n], 1, &VL[ki + is * ldvl], 1);
                    cblas_scopy(n - ki, &work[ki + n2], 1, &VL[ki + (is + 1) * ldvl], 1);

                    emax = 0.0f;
                    for (k = ki; k < n; k++) {
                        f32 temp = fabsf(VL[k + is * ldvl]) + fabsf(VL[k + (is + 1) * ldvl]);
                        if (temp > emax) emax = temp;
                    }
                    remax = 1.0f / emax;
                    cblas_sscal(n - ki, remax, &VL[ki + is * ldvl], 1);
                    cblas_sscal(n - ki, remax, &VL[ki + (is + 1) * ldvl], 1);

                    for (k = 0; k < ki; k++) {
                        VL[k + is * ldvl] = 0.0f;
                        VL[k + (is + 1) * ldvl] = 0.0f;
                    }
                } else {
                    if (ki < n - 2) {
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, n - ki - 2, 1.0f,
                                    &VL[(ki + 2) * ldvl], ldvl,
                                    &work[(ki + 2) + n], 1, work[ki + n],
                                    &VL[ki * ldvl], 1);
                        cblas_sgemv(CblasColMajor, CblasNoTrans, n, n - ki - 2, 1.0f,
                                    &VL[(ki + 2) * ldvl], ldvl,
                                    &work[(ki + 2) + n2], 1, work[(ki + 1) + n2],
                                    &VL[(ki + 1) * ldvl], 1);
                    } else {
                        cblas_sscal(n, work[ki + n], &VL[ki * ldvl], 1);
                        cblas_sscal(n, work[(ki + 1) + n2], &VL[(ki + 1) * ldvl], 1);
                    }

                    emax = 0.0f;
                    for (k = 0; k < n; k++) {
                        f32 temp = fabsf(VL[k + ki * ldvl]) + fabsf(VL[k + (ki + 1) * ldvl]);
                        if (temp > emax) emax = temp;
                    }
                    remax = 1.0f / emax;
                    cblas_sscal(n, remax, &VL[ki * ldvl], 1);
                    cblas_sscal(n, remax, &VL[(ki + 1) * ldvl], 1);
                }
            }

            is++;
            if (ip != 0)
                is++;
L250:
            if (ip == -1)
                ip = 0;
            if (ip == 1)
                ip = -1;
        }
    }
}
