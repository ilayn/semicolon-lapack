/**
 * @file zlahqr.c
 * @brief ZLAHQR computes the eigenvalues and Schur factorization of an
 *        upper Hessenberg matrix, using the single-shift QR algorithm.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * ZLAHQR is an auxiliary routine called by ZHSEQR to update the eigenvalues
 * and Schur decomposition already computed by ZHSEQR, by dealing with the
 * Hessenberg submatrix in rows and columns ilo to ihi.
 *
 * @param[in] wantt   If nonzero, the full Schur form T is required;
 *                    if zero, only eigenvalues are required.
 * @param[in] wantz   If nonzero, the matrix of Schur vectors Z is required;
 *                    if zero, Schur vectors are not required.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First row/column of the active submatrix (0-based).
 * @param[in] ihi     Last row/column of the active submatrix (0-based).
 * @param[in,out] H   Complex array, dimension (ldh, n).
 *                    On entry, the upper Hessenberg matrix H.
 *                    On exit, if info == 0 and wantt is nonzero, H is upper
 *                    triangular in rows and columns ilo:ihi.
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] W      Complex array, dimension (n).
 *                    The computed eigenvalues ilo to ihi.
 * @param[in] iloz    First row of Z to which transformations must be applied.
 * @param[in] ihiz    Last row of Z to which transformations must be applied.
 * @param[in,out] Z   Complex array, dimension (ldz, n).
 * @param[in] ldz     Leading dimension of Z. ldz >= max(1, n).
 * @param[out] info   = 0: successful exit
 *                    > 0: if info = i (1-based), ZLAHQR failed to compute
 *                         all eigenvalues.
 */
void zlahqr(const INT wantt, const INT wantz, const INT n,
            const INT ilo, const INT ihi,
            c128* H, const INT ldh,
            c128* W,
            const INT iloz, const INT ihiz,
            c128* Z, const INT ldz,
            INT* info)
{
    const c128 czero = 0.0;
    const c128 cone = 1.0;
    const f64 rzero = 0.0;
    const f64 half = 0.5;
    const f64 dat1 = 3.0 / 4.0;
    const INT kexsh = 10;

    c128 sc, sum, t, t1, temp, u, v2, x, y;
    c128 h11, h11s, h22;
    f64 aa, ab, ba, bb, h10, h21, rtemp, s, safmin;
    f64 smlnum, sx, t2, tst, ulp;
    INT i, i1, i2, its, itmax, j, jhi, jlo, k, l, m, nh, nz, kdefl;

    c128 v[2];

    *info = 0;

    if (n == 0)
        return;

    if (ilo == ihi) {
        W[ilo] = H[ilo + ilo * ldh];
        return;
    }

    /* Clear out the trash */
    for (j = ilo; j <= ihi - 3; j++) {
        H[(j + 2) + j * ldh] = czero;
        H[(j + 3) + j * ldh] = czero;
    }
    if (ilo <= ihi - 2)
        H[ihi + (ihi - 2) * ldh] = czero;

    /* Ensure that subdiagonal entries are real */
    if (wantt) {
        jlo = 0;
        jhi = n - 1;
    } else {
        jlo = ilo;
        jhi = ihi;
    }
    for (i = ilo + 1; i <= ihi; i++) {
        if (cimag(H[i + (i - 1) * ldh]) != rzero) {
            sc = H[i + (i - 1) * ldh] / cabs1(H[i + (i - 1) * ldh]);
            sc = conj(sc) / cabs(sc);
            H[i + (i - 1) * ldh] = cabs(H[i + (i - 1) * ldh]);
            cblas_zscal(jhi - i + 1, &sc, &H[i + i * ldh], ldh);
            {
                c128 sc_conj = conj(sc);
                INT cnt = ((jhi < i + 1) ? jhi : i + 1) - jlo + 1;
                cblas_zscal(cnt, &sc_conj, &H[jlo + i * ldh], 1);
            }
            if (wantz) {
                c128 sc_conj = conj(sc);
                cblas_zscal(ihiz - iloz + 1, &sc_conj, &Z[iloz + i * ldz], 1);
            }
        }
    }

    nh = ihi - ilo + 1;
    nz = ihiz - iloz + 1;

    safmin = dlamch("Safe minimum");
    ulp = dlamch("Precision");
    smlnum = safmin * ((f64)nh / ulp);

    i1 = 0;
    i2 = 0;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    }

    itmax = 30 * (10 > nh ? 10 : nh);

    kdefl = 0;

    i = ihi;

    while (i >= ilo) {
        l = ilo;

        for (its = 0; its <= itmax; its++) {

            /* Look for a single small subdiagonal element */
            for (k = i; k >= l + 1; k--) {
                if (cabs1(H[k + (k - 1) * ldh]) <= smlnum)
                    break;
                tst = cabs1(H[(k - 1) + (k - 1) * ldh]) + cabs1(H[k + k * ldh]);
                if (tst == rzero) {
                    if (k - 2 >= ilo)
                        tst = tst + fabs(creal(H[(k - 1) + (k - 2) * ldh]));
                    if (k + 1 <= ihi)
                        tst = tst + fabs(creal(H[(k + 1) + k * ldh]));
                }
                if (fabs(creal(H[k + (k - 1) * ldh])) <= ulp * tst) {
                    ab = cabs1(H[k + (k - 1) * ldh]) > cabs1(H[(k - 1) + k * ldh]) ?
                         cabs1(H[k + (k - 1) * ldh]) : cabs1(H[(k - 1) + k * ldh]);
                    ba = cabs1(H[k + (k - 1) * ldh]) < cabs1(H[(k - 1) + k * ldh]) ?
                         cabs1(H[k + (k - 1) * ldh]) : cabs1(H[(k - 1) + k * ldh]);
                    aa = cabs1(H[k + k * ldh]) > cabs1(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         cabs1(H[k + k * ldh]) : cabs1(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
                    bb = cabs1(H[k + k * ldh]) < cabs1(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         cabs1(H[k + k * ldh]) : cabs1(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
                    s = aa + ab;
                    if (ba * (ab / s) <= (smlnum > ulp * (bb * (aa / s)) ?
                                          smlnum : ulp * (bb * (aa / s))))
                        break;
                }
            }
            l = k;

            if (l > ilo) {
                H[l + (l - 1) * ldh] = czero;
            }

            /* Exit from loop if a submatrix of order 1 has split off */
            if (l >= i)
                goto converged;

            kdefl = kdefl + 1;

            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            if ((kdefl % (2 * kexsh)) == 0) {
                /* Exceptional shift */
                s = dat1 * fabs(creal(H[i + (i - 1) * ldh]));
                t = s + H[i + i * ldh];
            } else if ((kdefl % kexsh) == 0) {
                /* Exceptional shift */
                s = dat1 * fabs(creal(H[(l + 1) + l * ldh]));
                t = s + H[l + l * ldh];
            } else {
                /* Wilkinson's shift */
                t = H[i + i * ldh];
                u = csqrt(H[(i - 1) + i * ldh]) * csqrt(H[i + (i - 1) * ldh]);
                s = cabs1(u);
                if (s != rzero) {
                    x = half * (H[(i - 1) + (i - 1) * ldh] - t);
                    sx = cabs1(x);
                    s = s > cabs1(x) ? s : cabs1(x);
                    y = s * csqrt((x / s) * (x / s) + (u / s) * (u / s));
                    if (sx > rzero) {
                        if (creal(x / sx) * creal(y) + cimag(x / sx) * cimag(y) < rzero)
                            y = -y;
                    }
                    t = t - u * zladiv(u, (x + y));
                }
            }

            /* Look for two consecutive small subdiagonal elements */
            for (m = i - 1; m >= l + 1; m--) {
                h11 = H[m + m * ldh];
                h22 = H[(m + 1) + (m + 1) * ldh];
                h11s = h11 - t;
                h21 = creal(H[(m + 1) + m * ldh]);
                s = cabs1(h11s) + fabs(h21);
                h11s = h11s / s;
                h21 = h21 / s;
                v[0] = h11s;
                v[1] = h21;
                h10 = creal(H[m + (m - 1) * ldh]);
                if (fabs(h10) * fabs(h21) <=
                    ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22))))
                    break;
            }
            if (m == l) {
                h11 = H[l + l * ldh];
                h11s = h11 - t;
                h21 = creal(H[(l + 1) + l * ldh]);
                s = cabs1(h11s) + fabs(h21);
                h11s = h11s / s;
                h21 = h21 / s;
                v[0] = h11s;
                v[1] = h21;
            }

            /* Single-shift QR step */
            for (k = m; k <= i - 1; k++) {
                if (k > m)
                    cblas_zcopy(2, &H[k + (k - 1) * ldh], 1, v, 1);

                zlarfg(2, &v[0], &v[1], 1, &t1);

                if (k > m) {
                    H[k + (k - 1) * ldh] = v[0];
                    H[(k + 1) + (k - 1) * ldh] = czero;
                }
                v2 = v[1];
                t2 = creal(t1 * v2);

                /* Apply G from the left to transform the rows of the matrix
                 * in columns k to i2 */
                for (j = k; j <= i2; j++) {
                    sum = conj(t1) * H[k + j * ldh] + t2 * H[(k + 1) + j * ldh];
                    H[k + j * ldh] = H[k + j * ldh] - sum;
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - sum * v2;
                }

                /* Apply G from the right to transform the columns of the
                 * matrix in rows i1 to min(k+2, i) */
                for (j = i1; j <= (k + 2 < i ? k + 2 : i); j++) {
                    sum = t1 * H[j + k * ldh] + t2 * H[j + (k + 1) * ldh];
                    H[j + k * ldh] = H[j + k * ldh] - sum;
                    H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - sum * conj(v2);
                }

                if (wantz) {
                    for (j = iloz; j <= ihiz; j++) {
                        sum = t1 * Z[j + k * ldz] + t2 * Z[j + (k + 1) * ldz];
                        Z[j + k * ldz] = Z[j + k * ldz] - sum;
                        Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - sum * conj(v2);
                    }
                }

                if (k == m && m > l) {
                    temp = cone - t1;
                    temp = temp / cabs(temp);
                    H[(m + 1) + m * ldh] = H[(m + 1) + m * ldh] * conj(temp);
                    if (m + 2 <= i)
                        H[(m + 2) + (m + 1) * ldh] = H[(m + 2) + (m + 1) * ldh] * temp;
                    for (j = m; j <= i; j++) {
                        if (j != m + 1) {
                            if (i2 > j) {
                                cblas_zscal(i2 - j, &temp, &H[j + (j + 1) * ldh], ldh);
                            }
                            {
                                c128 ct = conj(temp);
                                cblas_zscal(j - i1, &ct, &H[i1 + j * ldh], 1);
                            }
                            if (wantz) {
                                c128 ct = conj(temp);
                                cblas_zscal(nz, &ct, &Z[iloz + j * ldz], 1);
                            }
                        }
                    }
                }
            }

            /* Ensure that H(I,I-1) is real */
            temp = H[i + (i - 1) * ldh];
            if (cimag(temp) != rzero) {
                rtemp = cabs(temp);
                H[i + (i - 1) * ldh] = rtemp;
                temp = temp / rtemp;
                if (i2 > i) {
                    c128 ct = conj(temp);
                    cblas_zscal(i2 - i, &ct, &H[i + (i + 1) * ldh], ldh);
                }
                cblas_zscal(i - i1, &temp, &H[i1 + i * ldh], 1);
                if (wantz) {
                    cblas_zscal(nz, &temp, &Z[iloz + i * ldz], 1);
                }
            }
        }

        /* Failure to converge in remaining number of iterations */
        *info = i + 1;
        return;

    converged:
        W[i] = H[i + i * ldh];

        kdefl = 0;
        i = l - 1;
    }
}
