/**
 * @file dlaein.c
 * @brief DLAEIN computes a specified right or left eigenvector of an upper
 *        Hessenberg matrix by inverse iteration.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAEIN uses inverse iteration to find a right or left eigenvector
 * corresponding to the eigenvalue (wr, wi) of a real upper Hessenberg
 * matrix H.
 *
 * @param[in]     rightv  If nonzero, compute right eigenvector;
 *                        if zero, compute left eigenvector.
 * @param[in]     noinit  If nonzero, no initial vector supplied in (vr,vi);
 *                        if zero, initial vector supplied in (vr,vi).
 * @param[in]     n       The order of the matrix H (n >= 0).
 * @param[in]     H       Upper Hessenberg matrix H. Array of dimension (ldh, n).
 * @param[in]     ldh     The leading dimension of H (ldh >= max(1,n)).
 * @param[in]     wr      The real part of the eigenvalue.
 * @param[in]     wi      The imaginary part of the eigenvalue.
 * @param[in,out] vr      Real part of eigenvector. Array of dimension n.
 *                        On exit, if wi = 0.0, vr contains the computed real
 *                        eigenvector; if wi != 0.0, vr contains the real part
 *                        of the computed complex eigenvector.
 * @param[in,out] vi      Imaginary part of eigenvector. Array of dimension n.
 *                        Not referenced if wi = 0.0.
 * @param[out]    B       Workspace array of dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of B (ldb >= n+1).
 * @param[out]    work    Workspace array of dimension n.
 * @param[in]     eps3    A small machine-dependent value used to perturb
 *                        close eigenvalues and replace zero pivots.
 * @param[in]     smlnum  A machine-dependent value close to underflow threshold.
 * @param[in]     bignum  A machine-dependent value close to overflow threshold.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - = 1: inverse iteration did not converge; vr is set
 *                           to the last iterate.
 */
void dlaein(
    const INT rightv,
    const INT noinit,
    const INT n,
    const f64* restrict H,
    const INT ldh,
    const f64 wr,
    const f64 wi,
    f64* restrict vr,
    f64* restrict vi,
    f64* restrict B,
    const INT ldb,
    f64* restrict work,
    const f64 eps3,
    const f64 smlnum,
    const f64 bignum,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TENTH = 0.1;

    INT i, i1, i2, i3, ierr, its, j;
    f64 absbii, absbjj, ei, ej, growto, norm, nrmsml;
    f64 rec, rootn, scale, temp, vcrit, vmax, vnorm, w, w1, x, xi, xr, y;

    *info = 0;

    /* GROWTO is the threshold used in the acceptance test for an eigenvector */
    rootn = sqrt((f64)n);
    growto = TENTH / rootn;
    nrmsml = ONE > eps3 * rootn ? ONE : eps3 * rootn;
    nrmsml *= smlnum;

    /* Form B = H - (wr,wi)*I (except that the subdiagonal elements and
     * the imaginary parts of the diagonal elements are not stored). */
    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++) {
            B[i + j * ldb] = H[i + j * ldh];
        }
        B[j + j * ldb] = H[j + j * ldh] - wr;
    }

    if (wi == ZERO) {
        /* Real eigenvalue */

        if (noinit) {
            /* Set initial vector */
            for (i = 0; i < n; i++) {
                vr[i] = eps3;
            }
        } else {
            /* Scale supplied initial vector */
            vnorm = cblas_dnrm2(n, vr, 1);
            f64 denom = vnorm > nrmsml ? vnorm : nrmsml;
            cblas_dscal(n, (eps3 * rootn) / denom, vr, 1);
        }

        if (rightv) {
            /* LU decomposition with partial pivoting of B, replacing zero
             * pivots by eps3. */
            for (i = 0; i < n - 1; i++) {
                ei = H[i + 1 + i * ldh];
                if (fabs(B[i + i * ldb]) < fabs(ei)) {
                    /* Interchange rows and eliminate */
                    x = B[i + i * ldb] / ei;
                    B[i + i * ldb] = ei;
                    for (j = i + 1; j < n; j++) {
                        temp = B[i + 1 + j * ldb];
                        B[i + 1 + j * ldb] = B[i + j * ldb] - x * temp;
                        B[i + j * ldb] = temp;
                    }
                } else {
                    /* Eliminate without interchange */
                    if (B[i + i * ldb] == ZERO) {
                        B[i + i * ldb] = eps3;
                    }
                    x = ei / B[i + i * ldb];
                    if (x != ZERO) {
                        for (j = i + 1; j < n; j++) {
                            B[i + 1 + j * ldb] -= x * B[i + j * ldb];
                        }
                    }
                }
            }
            if (B[n - 1 + (n - 1) * ldb] == ZERO) {
                B[n - 1 + (n - 1) * ldb] = eps3;
            }

            /* Solve using DLATRS */
            char normin = 'N';
            for (its = 0; its < n; its++) {
                /* Solve U*x = scale*v for a right eigenvector */
                dlatrs("U", "N", "N", &normin, n, B, ldb, vr, &scale, work, &ierr);
                normin = 'Y';

                /* Test for sufficient growth in the norm of v */
                vnorm = cblas_dasum(n, vr, 1);
                if (vnorm >= growto * scale) {
                    goto L120;
                }

                /* Choose new orthogonal starting vector and try again */
                temp = eps3 / (rootn + ONE);
                vr[0] = eps3;
                for (i = 1; i < n; i++) {
                    vr[i] = temp;
                }
                vr[n - its - 1] -= eps3 * rootn;
            }

            /* Failure to find eigenvector in n iterations */
            *info = 1;

        L120:
            /* Normalize eigenvector */
            i = cblas_idamax(n, vr, 1);
            cblas_dscal(n, ONE / fabs(vr[i]), vr, 1);

        } else {
            /* UL decomposition with partial pivoting of B, replacing zero
             * pivots by eps3. */
            for (j = n - 1; j >= 1; j--) {
                ej = H[j + (j - 1) * ldh];
                if (fabs(B[j + j * ldb]) < fabs(ej)) {
                    /* Interchange columns and eliminate */
                    x = B[j + j * ldb] / ej;
                    B[j + j * ldb] = ej;
                    for (i = 0; i < j; i++) {
                        temp = B[i + (j - 1) * ldb];
                        B[i + (j - 1) * ldb] = B[i + j * ldb] - x * temp;
                        B[i + j * ldb] = temp;
                    }
                } else {
                    /* Eliminate without interchange */
                    if (B[j + j * ldb] == ZERO) {
                        B[j + j * ldb] = eps3;
                    }
                    x = ej / B[j + j * ldb];
                    if (x != ZERO) {
                        for (i = 0; i < j; i++) {
                            B[i + (j - 1) * ldb] -= x * B[i + j * ldb];
                        }
                    }
                }
            }
            if (B[0] == ZERO) {
                B[0] = eps3;
            }

            /* Solve using DLATRS */
            char normin = 'N';
            for (its = 0; its < n; its++) {
                /* Solve U**T*x = scale*v for a left eigenvector */
                dlatrs("U", "T", "N", &normin, n, B, ldb, vr, &scale, work, &ierr);
                normin = 'Y';

                /* Test for sufficient growth in the norm of v */
                vnorm = cblas_dasum(n, vr, 1);
                if (vnorm >= growto * scale) {
                    goto L120b;
                }

                /* Choose new orthogonal starting vector and try again */
                temp = eps3 / (rootn + ONE);
                vr[0] = eps3;
                for (i = 1; i < n; i++) {
                    vr[i] = temp;
                }
                vr[n - its - 1] -= eps3 * rootn;
            }

            /* Failure to find eigenvector in n iterations */
            *info = 1;

        L120b:
            /* Normalize eigenvector */
            i = cblas_idamax(n, vr, 1);
            cblas_dscal(n, ONE / fabs(vr[i]), vr, 1);
        }

    } else {
        /* Complex eigenvalue */

        if (noinit) {
            /* Set initial vector */
            for (i = 0; i < n; i++) {
                vr[i] = eps3;
                vi[i] = ZERO;
            }
        } else {
            /* Scale supplied initial vector */
            norm = dlapy2(cblas_dnrm2(n, vr, 1), cblas_dnrm2(n, vi, 1));
            f64 denom = norm > nrmsml ? norm : nrmsml;
            rec = (eps3 * rootn) / denom;
            cblas_dscal(n, rec, vr, 1);
            cblas_dscal(n, rec, vi, 1);
        }

        if (rightv) {
            /* LU decomposition with partial pivoting of B, replacing zero
             * pivots by eps3.
             * The imaginary part of the (i,j)-th element of U is stored in B(j+1,i). */
            B[1 + 0 * ldb] = -wi;
            for (i = 1; i < n; i++) {
                B[i + 1 + 0 * ldb] = ZERO;
            }

            for (i = 0; i < n - 1; i++) {
                absbii = dlapy2(B[i + i * ldb], B[i + 1 + i * ldb]);
                ei = H[i + 1 + i * ldh];
                if (absbii < fabs(ei)) {
                    /* Interchange rows and eliminate */
                    xr = B[i + i * ldb] / ei;
                    xi = B[i + 1 + i * ldb] / ei;
                    B[i + i * ldb] = ei;
                    B[i + 1 + i * ldb] = ZERO;
                    for (j = i + 1; j < n; j++) {
                        temp = B[i + 1 + j * ldb];
                        B[i + 1 + j * ldb] = B[i + j * ldb] - xr * temp;
                        B[j + 1 + (i + 1) * ldb] = B[j + 1 + i * ldb] - xi * temp;
                        B[i + j * ldb] = temp;
                        B[j + 1 + i * ldb] = ZERO;
                    }
                    B[i + 2 + i * ldb] = -wi;
                    B[i + 1 + (i + 1) * ldb] -= xi * wi;
                    B[i + 2 + (i + 1) * ldb] += xr * wi;
                } else {
                    /* Eliminate without interchanging rows */
                    if (absbii == ZERO) {
                        B[i + i * ldb] = eps3;
                        B[i + 1 + i * ldb] = ZERO;
                        absbii = eps3;
                    }
                    ei = (ei / absbii) / absbii;
                    xr = B[i + i * ldb] * ei;
                    xi = -B[i + 1 + i * ldb] * ei;
                    for (j = i + 1; j < n; j++) {
                        B[i + 1 + j * ldb] = B[i + 1 + j * ldb] - xr * B[i + j * ldb] +
                                             xi * B[j + 1 + i * ldb];
                        B[j + 1 + (i + 1) * ldb] = -xr * B[j + 1 + i * ldb] - xi * B[i + j * ldb];
                    }
                    B[i + 2 + (i + 1) * ldb] -= wi;
                }

                /* Compute 1-norm of offdiagonal elements of i-th row */
                work[i] = cblas_dasum(n - i - 1, &B[i + (i + 1) * ldb], ldb) +
                          cblas_dasum(n - i - 1, &B[i + 2 + i * ldb], 1);
            }
            if (B[n - 1 + (n - 1) * ldb] == ZERO && B[n + (n - 1) * ldb] == ZERO) {
                B[n - 1 + (n - 1) * ldb] = eps3;
            }
            work[n - 1] = ZERO;

            i1 = n - 1;
            i2 = 0;
            i3 = -1;

        } else {
            /* UL decomposition with partial pivoting of conjg(B),
             * replacing zero pivots by eps3.
             * The imaginary part of the (i,j)-th element of U is stored in B(j+1,i). */
            B[n + (n - 1) * ldb] = wi;
            for (j = 0; j < n - 1; j++) {
                B[n + j * ldb] = ZERO;
            }

            for (j = n - 1; j >= 1; j--) {
                ej = H[j + (j - 1) * ldh];
                absbjj = dlapy2(B[j + j * ldb], B[j + 1 + j * ldb]);
                if (absbjj < fabs(ej)) {
                    /* Interchange columns and eliminate */
                    xr = B[j + j * ldb] / ej;
                    xi = B[j + 1 + j * ldb] / ej;
                    B[j + j * ldb] = ej;
                    B[j + 1 + j * ldb] = ZERO;
                    for (i = 0; i < j; i++) {
                        temp = B[i + (j - 1) * ldb];
                        B[i + (j - 1) * ldb] = B[i + j * ldb] - xr * temp;
                        B[j + i * ldb] = B[j + 1 + i * ldb] - xi * temp;
                        B[i + j * ldb] = temp;
                        B[j + 1 + i * ldb] = ZERO;
                    }
                    B[j + 1 + (j - 1) * ldb] = wi;
                    B[j - 1 + (j - 1) * ldb] += xi * wi;
                    B[j + (j - 1) * ldb] -= xr * wi;
                } else {
                    /* Eliminate without interchange */
                    if (absbjj == ZERO) {
                        B[j + j * ldb] = eps3;
                        B[j + 1 + j * ldb] = ZERO;
                        absbjj = eps3;
                    }
                    ej = (ej / absbjj) / absbjj;
                    xr = B[j + j * ldb] * ej;
                    xi = -B[j + 1 + j * ldb] * ej;
                    for (i = 0; i < j; i++) {
                        B[i + (j - 1) * ldb] = B[i + (j - 1) * ldb] - xr * B[i + j * ldb] +
                                               xi * B[j + 1 + i * ldb];
                        B[j + i * ldb] = -xr * B[j + 1 + i * ldb] - xi * B[i + j * ldb];
                    }
                    B[j + (j - 1) * ldb] += wi;
                }

                /* Compute 1-norm of offdiagonal elements of j-th column */
                work[j] = cblas_dasum(j, &B[j * ldb], 1) +
                          cblas_dasum(j, &B[j + 1], ldb);
            }
            if (B[0] == ZERO && B[1] == ZERO) {
                B[0] = eps3;
            }
            work[0] = ZERO;

            i1 = 0;
            i2 = n - 1;
            i3 = 1;
        }

        for (its = 0; its < n; its++) {
            scale = ONE;
            vmax = ONE;
            vcrit = bignum;

            /* Solve U*(xr,xi) = scale*(vr,vi) for a right eigenvector,
             *    or U**T*(xr,xi) = scale*(vr,vi) for a left eigenvector,
             * overwriting (xr,xi) on (vr,vi). */
            for (i = i1; (i3 > 0) ? (i <= i2) : (i >= i2); i += i3) {
                if (work[i] > vcrit) {
                    rec = ONE / vmax;
                    cblas_dscal(n, rec, vr, 1);
                    cblas_dscal(n, rec, vi, 1);
                    scale *= rec;
                    vmax = ONE;
                }

                xr = vr[i];
                xi = vi[i];
                if (rightv) {
                    for (j = i + 1; j < n; j++) {
                        xr = xr - B[i + j * ldb] * vr[j] + B[j + 1 + i * ldb] * vi[j];
                        xi = xi - B[i + j * ldb] * vi[j] - B[j + 1 + i * ldb] * vr[j];
                    }
                } else {
                    for (j = 0; j < i; j++) {
                        xr = xr - B[j + i * ldb] * vr[j] + B[i + 1 + j * ldb] * vi[j];
                        xi = xi - B[j + i * ldb] * vi[j] - B[i + 1 + j * ldb] * vr[j];
                    }
                }

                w = fabs(B[i + i * ldb]) + fabs(B[i + 1 + i * ldb]);
                if (w > smlnum) {
                    if (w < ONE) {
                        w1 = fabs(xr) + fabs(xi);
                        if (w1 > w * bignum) {
                            rec = ONE / w1;
                            cblas_dscal(n, rec, vr, 1);
                            cblas_dscal(n, rec, vi, 1);
                            xr = vr[i];
                            xi = vi[i];
                            scale *= rec;
                            vmax *= rec;
                        }
                    }

                    /* Divide by diagonal element of B */
                    dladiv(xr, xi, B[i + i * ldb], B[i + 1 + i * ldb], &vr[i], &vi[i]);
                    f64 absvr = fabs(vr[i]) + fabs(vi[i]);
                    if (absvr > vmax) {
                        vmax = absvr;
                    }
                    vcrit = bignum / vmax;
                } else {
                    for (j = 0; j < n; j++) {
                        vr[j] = ZERO;
                        vi[j] = ZERO;
                    }
                    vr[i] = ONE;
                    vi[i] = ONE;
                    scale = ZERO;
                    vmax = ONE;
                    vcrit = bignum;
                }
            }

            /* Test for sufficient growth in the norm of (vr,vi) */
            vnorm = cblas_dasum(n, vr, 1) + cblas_dasum(n, vi, 1);
            if (vnorm >= growto * scale) {
                goto L280;
            }

            /* Choose a new orthogonal starting vector and try again */
            y = eps3 / (rootn + ONE);
            vr[0] = eps3;
            vi[0] = ZERO;
            for (i = 1; i < n; i++) {
                vr[i] = y;
                vi[i] = ZERO;
            }
            vr[n - its - 1] -= eps3 * rootn;
        }

        /* Failure to find eigenvector in n iterations */
        *info = 1;

    L280:
        /* Normalize eigenvector */
        vnorm = ZERO;
        for (i = 0; i < n; i++) {
            f64 mag = fabs(vr[i]) + fabs(vi[i]);
            if (mag > vnorm) {
                vnorm = mag;
            }
        }
        cblas_dscal(n, ONE / vnorm, vr, 1);
        cblas_dscal(n, ONE / vnorm, vi, 1);
    }
}
