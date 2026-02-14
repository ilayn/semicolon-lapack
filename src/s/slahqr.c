/**
 * @file slahqr.c
 * @brief SLAHQR computes the eigenvalues and Schur factorization of an
 *        upper Hessenberg matrix, using the double-shift/single-shift QR
 *        algorithm.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

/**
 * SLAHQR is an auxiliary routine called by SHSEQR to update the eigenvalues
 * and Schur decomposition already computed by SHSEQR, by dealing with the
 * Hessenberg submatrix in rows and columns ilo to ihi.
 *
 * This is a modified version that is (1) more robust against overflow and
 * underflow and (2) adopts the more conservative Ahues & Tisseur stopping
 * criterion (LAWN 122, 1997).
 *
 * @param[in] wantt   If nonzero, the full Schur form T is required;
 *                    if zero, only eigenvalues are required.
 * @param[in] wantz   If nonzero, the matrix of Schur vectors Z is required;
 *                    if zero, Schur vectors are not required.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First row/column of the active submatrix (0-based).
 * @param[in] ihi     Last row/column of the active submatrix (0-based).
 *                    It is assumed that H is already upper quasi-triangular
 *                    in rows and columns ihi+1:n-1.
 * @param[in,out] H   Double precision array, dimension (ldh, n).
 *                    On entry, the upper Hessenberg matrix H.
 *                    On exit, if info == 0 and wantt is nonzero, H is upper
 *                    quasi-triangular in rows and columns ilo:ihi.
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] wr     Double precision array, dimension (n).
 *                    Real parts of computed eigenvalues ilo to ihi.
 * @param[out] wi     Double precision array, dimension (n).
 *                    Imaginary parts of computed eigenvalues ilo to ihi.
 * @param[in] iloz    First row of Z to which transformations must be applied.
 * @param[in] ihiz    Last row of Z to which transformations must be applied.
 * @param[in,out] Z   Double precision array, dimension (ldz, n).
 *                    If wantz is nonzero, on entry Z must contain the current
 *                    matrix Z of transformations, and on exit Z has been
 *                    updated; transformations applied only to Z(iloz:ihiz, ilo:ihi).
 * @param[in] ldz     Leading dimension of Z. ldz >= max(1, n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - > 0: If info = i (1-based), SLAHQR failed to compute all
 *                           eigenvalues; elements i:ihi of wr and wi contain
 *                           those eigenvalues which have been successfully computed.
 */
SEMICOLON_API void slahqr(const int wantt, const int wantz, const int n,
                          const int ilo, const int ihi,
                          f32* H, const int ldh,
                          f32* wr, f32* wi,
                          const int iloz, const int ihiz,
                          f32* Z, const int ldz,
                          int* info)
{
    /* Parameters */
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 two = 2.0f;
    const f32 dat1 = 3.0f / 4.0f;
    const f32 dat2 = -0.4375f;
    const int kexsh = 10;

    /* Local scalars */
    f32 aa, ab, ba, bb, cs, det, h11, h12, h21, h21s, h22;
    f32 rt1i, rt1r, rt2i, rt2r, rtdisc, s, safmin;
    f32 smlnum, sn, sum, t1, t2, t3, tr, tst, ulp, v2, v3;
    int i, i1, i2, its, itmax, j, k, l, m, nh, nr, nz, kdefl;

    /* Local array */
    f32 v[3];

    *info = 0;

    /* Quick return if possible */
    if (n == 0)
        return;

    if (ilo == ihi) {
        wr[ilo] = H[ilo + ilo * ldh];
        wi[ilo] = zero;
        return;
    }

    /* Clear out the trash below the subdiagonal */
    for (j = ilo; j <= ihi - 3; j++) {
        H[(j + 2) + j * ldh] = zero;
        H[(j + 3) + j * ldh] = zero;
    }
    if (ilo <= ihi - 2)
        H[ihi + (ihi - 2) * ldh] = zero;

    nh = ihi - ilo + 1;
    nz = ihiz - iloz + 1;

    /* Set machine-dependent constants for the stopping criterion */
    safmin = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = safmin * ((f32)nh / ulp);

    /* I1 and I2 are the indices of the first row and last column of H
     * to which transformations must be applied. If eigenvalues only are
     * being computed, I1 and I2 are set inside the main loop. */
    i1 = 0;  /* Initialize to avoid warning */
    i2 = 0;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    }

    /* ITMAX is the total number of QR iterations allowed */
    itmax = 30 * (10 > nh ? 10 : nh);

    /* KDEFL counts the number of iterations since a deflation */
    kdefl = 0;

    /* The main loop begins here. I is the loop index and decreases from
     * ihi to ilo in steps of 1 or 2. Each iteration of the loop works
     * with the active submatrix in rows and columns l to i.
     * Eigenvalues i+1 to ihi have already converged. */
    i = ihi;

    while (i >= ilo) {
        l = ilo;

        /* Perform QR iterations on rows and columns ilo to i until a
         * submatrix of order 1 or 2 splits off at the bottom */
        for (its = 0; its <= itmax; its++) {

            /* Look for a single small subdiagonal element */
            for (k = i; k >= l + 1; k--) {
                if (fabsf(H[k + (k - 1) * ldh]) <= smlnum)
                    break;
                tst = fabsf(H[(k - 1) + (k - 1) * ldh]) + fabsf(H[k + k * ldh]);
                if (tst == zero) {
                    if (k - 2 >= ilo)
                        tst = tst + fabsf(H[(k - 1) + (k - 2) * ldh]);
                    if (k + 1 <= ihi)
                        tst = tst + fabsf(H[(k + 1) + k * ldh]);
                }
                /* Conservative small subdiagonal deflation criterion
                 * (Ahues & Tisseur, LAWN 122, 1997) */
                if (fabsf(H[k + (k - 1) * ldh]) <= ulp * tst) {
                    ab = fabsf(H[k + (k - 1) * ldh]) > fabsf(H[(k - 1) + k * ldh]) ?
                         fabsf(H[k + (k - 1) * ldh]) : fabsf(H[(k - 1) + k * ldh]);
                    ba = fabsf(H[k + (k - 1) * ldh]) < fabsf(H[(k - 1) + k * ldh]) ?
                         fabsf(H[k + (k - 1) * ldh]) : fabsf(H[(k - 1) + k * ldh]);
                    aa = fabsf(H[k + k * ldh]) > fabsf(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         fabsf(H[k + k * ldh]) : fabsf(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
                    bb = fabsf(H[k + k * ldh]) < fabsf(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         fabsf(H[k + k * ldh]) : fabsf(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
                    s = aa + ab;
                    if (ba * (ab / s) <= (smlnum > ulp * (bb * (aa / s)) ?
                                          smlnum : ulp * (bb * (aa / s))))
                        break;
                }
            }
            l = k;

            if (l > ilo) {
                /* H(l, l-1) is negligible */
                H[l + (l - 1) * ldh] = zero;
            }

            /* Exit from loop if a submatrix of order 1 or 2 has split off */
            if (l >= i - 1)
                goto converged;

            kdefl = kdefl + 1;

            /* Now the active submatrix is in rows and columns l to i. If
             * eigenvalues only are being computed, only the active submatrix
             * need be transformed. */
            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            if ((kdefl % (2 * kexsh)) == 0) {
                /* Exceptional shift */
                s = fabsf(H[i + (i - 1) * ldh]) + fabsf(H[(i - 1) + (i - 2) * ldh]);
                h11 = dat1 * s + H[i + i * ldh];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else if ((kdefl % kexsh) == 0) {
                /* Exceptional shift */
                s = fabsf(H[(l + 1) + l * ldh]) + fabsf(H[(l + 2) + (l + 1) * ldh]);
                h11 = dat1 * s + H[l + l * ldh];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else {
                /* Prepare to use Francis' double shift
                 * (i.e. 2nd degree generalized Rayleigh quotient) */
                h11 = H[(i - 1) + (i - 1) * ldh];
                h21 = H[i + (i - 1) * ldh];
                h12 = H[(i - 1) + i * ldh];
                h22 = H[i + i * ldh];
            }

            s = fabsf(h11) + fabsf(h12) + fabsf(h21) + fabsf(h22);
            if (s == zero) {
                rt1r = zero;
                rt1i = zero;
                rt2r = zero;
                rt2i = zero;
            } else {
                h11 = h11 / s;
                h21 = h21 / s;
                h12 = h12 / s;
                h22 = h22 / s;
                tr = (h11 + h22) / two;
                det = (h11 - tr) * (h22 - tr) - h12 * h21;
                rtdisc = sqrtf(fabsf(det));
                if (det >= zero) {
                    /* Complex conjugate shifts */
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    /* Real shifts (use only one of them) */
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (fabsf(rt1r - h22) <= fabsf(rt2r - h22)) {
                        rt1r = rt1r * s;
                        rt2r = rt1r;
                    } else {
                        rt2r = rt2r * s;
                        rt1r = rt2r;
                    }
                    rt1i = zero;
                    rt2i = zero;
                }
            }

            /* Look for two consecutive small subdiagonal elements */
            for (m = i - 2; m >= l; m--) {
                /* Determine the effect of starting the double-shift QR
                 * iteration at row m, and see if this would make H(m, m-1)
                 * negligible. */
                h21s = H[(m + 1) + m * ldh];
                s = fabsf(H[m + m * ldh] - rt2r) + fabsf(rt2i) + fabsf(h21s);
                h21s = H[(m + 1) + m * ldh] / s;
                v[0] = h21s * H[m + (m + 1) * ldh] + (H[m + m * ldh] - rt1r) *
                       ((H[m + m * ldh] - rt2r) / s) - rt1i * (rt2i / s);
                v[1] = h21s * (H[m + m * ldh] + H[(m + 1) + (m + 1) * ldh] - rt1r - rt2r);
                v[2] = h21s * H[(m + 2) + (m + 1) * ldh];
                s = fabsf(v[0]) + fabsf(v[1]) + fabsf(v[2]);
                v[0] = v[0] / s;
                v[1] = v[1] / s;
                v[2] = v[2] / s;
                if (m == l)
                    break;
                if (fabsf(H[m + (m - 1) * ldh]) * (fabsf(v[1]) + fabsf(v[2])) <=
                    ulp * fabsf(v[0]) * (fabsf(H[(m - 1) + (m - 1) * ldh]) +
                                        fabsf(H[m + m * ldh]) +
                                        fabsf(H[(m + 1) + (m + 1) * ldh])))
                    break;
            }

            /* Double-shift QR step */
            for (k = m; k <= i - 1; k++) {
                /* The first iteration of this loop determines a reflection G
                 * from the vector v and applies it from left and right to H,
                 * thus creating a nonzero bulge below the subdiagonal.
                 *
                 * Each subsequent iteration determines a reflection G to
                 * restore the Hessenberg form in the (k-1)th column, and thus
                 * chases the bulge one step toward the bottom of the active
                 * submatrix. nr is the order of G. */
                nr = 3 < (i - k + 1) ? 3 : (i - k + 1);
                if (k > m)
                    cblas_scopy(nr, &H[k + (k - 1) * ldh], 1, v, 1);

                slarfg(nr, &v[0], &v[1], 1, &t1);

                if (k > m) {
                    H[k + (k - 1) * ldh] = v[0];
                    H[(k + 1) + (k - 1) * ldh] = zero;
                    if (k < i - 1)
                        H[(k + 2) + (k - 1) * ldh] = zero;
                } else if (m > l) {
                    /* Use the following instead of H(k,k-1) = -H(k,k-1) to
                     * avoid a bug when v[1] and v[2] underflow. */
                    H[k + (k - 1) * ldh] = H[k + (k - 1) * ldh] * (one - t1);
                }

                v2 = v[1];
                t2 = t1 * v2;

                if (nr == 3) {
                    v3 = v[2];
                    t3 = t1 * v3;

                    /* Apply G from the left to transform the rows of the matrix
                     * in columns k to i2 */
                    for (j = k; j <= i2; j++) {
                        sum = H[k + j * ldh] + v2 * H[(k + 1) + j * ldh] +
                              v3 * H[(k + 2) + j * ldh];
                        H[k + j * ldh] = H[k + j * ldh] - sum * t1;
                        H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - sum * t2;
                        H[(k + 2) + j * ldh] = H[(k + 2) + j * ldh] - sum * t3;
                    }

                    /* Apply G from the right to transform the columns of the
                     * matrix in rows i1 to min(k+3, i) */
                    for (j = i1; j <= (k + 3 < i ? k + 3 : i); j++) {
                        sum = H[j + k * ldh] + v2 * H[j + (k + 1) * ldh] +
                              v3 * H[j + (k + 2) * ldh];
                        H[j + k * ldh] = H[j + k * ldh] - sum * t1;
                        H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - sum * t2;
                        H[j + (k + 2) * ldh] = H[j + (k + 2) * ldh] - sum * t3;
                    }

                    if (wantz) {
                        /* Accumulate transformations in the matrix Z */
                        for (j = iloz; j <= ihiz; j++) {
                            sum = Z[j + k * ldz] + v2 * Z[j + (k + 1) * ldz] +
                                  v3 * Z[j + (k + 2) * ldz];
                            Z[j + k * ldz] = Z[j + k * ldz] - sum * t1;
                            Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - sum * t2;
                            Z[j + (k + 2) * ldz] = Z[j + (k + 2) * ldz] - sum * t3;
                        }
                    }
                } else if (nr == 2) {
                    /* Apply G from the left to transform the rows of the matrix
                     * in columns k to i2 */
                    for (j = k; j <= i2; j++) {
                        sum = H[k + j * ldh] + v2 * H[(k + 1) + j * ldh];
                        H[k + j * ldh] = H[k + j * ldh] - sum * t1;
                        H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - sum * t2;
                    }

                    /* Apply G from the right to transform the columns of the
                     * matrix in rows i1 to i */
                    for (j = i1; j <= i; j++) {
                        sum = H[j + k * ldh] + v2 * H[j + (k + 1) * ldh];
                        H[j + k * ldh] = H[j + k * ldh] - sum * t1;
                        H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - sum * t2;
                    }

                    if (wantz) {
                        /* Accumulate transformations in the matrix Z */
                        for (j = iloz; j <= ihiz; j++) {
                            sum = Z[j + k * ldz] + v2 * Z[j + (k + 1) * ldz];
                            Z[j + k * ldz] = Z[j + k * ldz] - sum * t1;
                            Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - sum * t2;
                        }
                    }
                }
            }
        }

        /* Failure to converge in remaining number of iterations */
        *info = i + 1;  /* Return 1-based index for compatibility */
        return;

    converged:
        if (l == i) {
            /* H(i, i-1) is negligible: one eigenvalue has converged */
            wr[i] = H[i + i * ldh];
            wi[i] = zero;
        } else if (l == i - 1) {
            /* H(i-1, i-2) is negligible: a pair of eigenvalues have converged.
             * Transform the 2-by-2 submatrix to standard Schur form,
             * and compute and store the eigenvalues. */
            slanv2(&H[(i - 1) + (i - 1) * ldh], &H[(i - 1) + i * ldh],
                   &H[i + (i - 1) * ldh], &H[i + i * ldh],
                   &wr[i - 1], &wi[i - 1], &wr[i], &wi[i], &cs, &sn);

            if (wantt) {
                /* Apply the transformation to the rest of H */
                if (i2 > i)
                    cblas_srot(i2 - i, &H[(i - 1) + (i + 1) * ldh], ldh,
                               &H[i + (i + 1) * ldh], ldh, cs, sn);
                cblas_srot(i - i1 - 1, &H[i1 + (i - 1) * ldh], 1,
                           &H[i1 + i * ldh], 1, cs, sn);
            }
            if (wantz) {
                /* Apply the transformation to Z */
                cblas_srot(nz, &Z[iloz + (i - 1) * ldz], 1,
                           &Z[iloz + i * ldz], 1, cs, sn);
            }
        }

        /* Reset deflation counter */
        kdefl = 0;

        /* Return to start of the main loop with new value of i */
        i = l - 1;
    }
}
