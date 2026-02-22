/**
 * @file clahqr.c
 * @brief CLAHQR computes the eigenvalues and Schur factorization of an
 *        upper Hessenberg matrix, using the single-shift QR algorithm.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>

/**
 * CLAHQR is an auxiliary routine called by CHSEQR to update the eigenvalues
 * and Schur decomposition already computed by CHSEQR, by dealing with the
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
 *                    > 0: if info = i (1-based), CLAHQR failed to compute
 *                         all eigenvalues.
 */
void clahqr(const INT wantt, const INT wantz, const INT n,
            const INT ilo, const INT ihi,
            c64* H, const INT ldh,
            c64* W,
            const INT iloz, const INT ihiz,
            c64* Z, const INT ldz,
            INT* info)
{
    const c64 czero = 0.0f;
    const c64 cone = 1.0f;
    const f32 rzero = 0.0f;
    const f32 half = 0.5f;
    const f32 dat1 = 3.0f / 4.0f;
    const INT kexsh = 10;

    c64 sc, sum, t, t1, temp, u, v2, x, y;
    c64 h11, h11s, h22;
    f32 aa, ab, ba, bb, h10, h21, rtemp, s, safmin;
    f32 smlnum, sx, t2, tst, ulp;
    INT i, i1, i2, its, itmax, j, jhi, jlo, k, l, m, nh, nz, kdefl;

    c64 v[2];

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
        if (cimagf(H[i + (i - 1) * ldh]) != rzero) {
            sc = H[i + (i - 1) * ldh] / cabs1f(H[i + (i - 1) * ldh]);
            sc = conjf(sc) / cabsf(sc);
            H[i + (i - 1) * ldh] = cabsf(H[i + (i - 1) * ldh]);
            cblas_cscal(jhi - i + 1, &sc, &H[i + i * ldh], ldh);
            {
                c64 sc_conj = conjf(sc);
                INT cnt = ((jhi < i + 1) ? jhi : i + 1) - jlo + 1;
                cblas_cscal(cnt, &sc_conj, &H[jlo + i * ldh], 1);
            }
            if (wantz) {
                c64 sc_conj = conjf(sc);
                cblas_cscal(ihiz - iloz + 1, &sc_conj, &Z[iloz + i * ldz], 1);
            }
        }
    }

    nh = ihi - ilo + 1;
    nz = ihiz - iloz + 1;

    safmin = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = safmin * ((f32)nh / ulp);

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
                if (cabs1f(H[k + (k - 1) * ldh]) <= smlnum)
                    break;
                tst = cabs1f(H[(k - 1) + (k - 1) * ldh]) + cabs1f(H[k + k * ldh]);
                if (tst == rzero) {
                    if (k - 2 >= ilo)
                        tst = tst + fabsf(crealf(H[(k - 1) + (k - 2) * ldh]));
                    if (k + 1 <= ihi)
                        tst = tst + fabsf(crealf(H[(k + 1) + k * ldh]));
                }
                if (fabsf(crealf(H[k + (k - 1) * ldh])) <= ulp * tst) {
                    ab = cabs1f(H[k + (k - 1) * ldh]) > cabs1f(H[(k - 1) + k * ldh]) ?
                         cabs1f(H[k + (k - 1) * ldh]) : cabs1f(H[(k - 1) + k * ldh]);
                    ba = cabs1f(H[k + (k - 1) * ldh]) < cabs1f(H[(k - 1) + k * ldh]) ?
                         cabs1f(H[k + (k - 1) * ldh]) : cabs1f(H[(k - 1) + k * ldh]);
                    aa = cabs1f(H[k + k * ldh]) > cabs1f(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         cabs1f(H[k + k * ldh]) : cabs1f(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
                    bb = cabs1f(H[k + k * ldh]) < cabs1f(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]) ?
                         cabs1f(H[k + k * ldh]) : cabs1f(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]);
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
                s = dat1 * fabsf(crealf(H[i + (i - 1) * ldh]));
                t = s + H[i + i * ldh];
            } else if ((kdefl % kexsh) == 0) {
                /* Exceptional shift */
                s = dat1 * fabsf(crealf(H[(l + 1) + l * ldh]));
                t = s + H[l + l * ldh];
            } else {
                /* Wilkinson's shift */
                t = H[i + i * ldh];
                u = csqrtf(H[(i - 1) + i * ldh]) * csqrtf(H[i + (i - 1) * ldh]);
                s = cabs1f(u);
                if (s != rzero) {
                    x = half * (H[(i - 1) + (i - 1) * ldh] - t);
                    sx = cabs1f(x);
                    s = s > cabs1f(x) ? s : cabs1f(x);
                    y = s * csqrtf((x / s) * (x / s) + (u / s) * (u / s));
                    if (sx > rzero) {
                        if (crealf(x / sx) * crealf(y) + cimagf(x / sx) * cimagf(y) < rzero)
                            y = -y;
                    }
                    t = t - u * cladiv(u, (x + y));
                }
            }

            /* Look for two consecutive small subdiagonal elements */
            for (m = i - 1; m >= l + 1; m--) {
                h11 = H[m + m * ldh];
                h22 = H[(m + 1) + (m + 1) * ldh];
                h11s = h11 - t;
                h21 = crealf(H[(m + 1) + m * ldh]);
                s = cabs1f(h11s) + fabsf(h21);
                h11s = h11s / s;
                h21 = h21 / s;
                v[0] = h11s;
                v[1] = h21;
                h10 = crealf(H[m + (m - 1) * ldh]);
                if (fabsf(h10) * fabsf(h21) <=
                    ulp * (cabs1f(h11s) * (cabs1f(h11) + cabs1f(h22))))
                    break;
            }
            if (m == l) {
                h11 = H[l + l * ldh];
                h11s = h11 - t;
                h21 = crealf(H[(l + 1) + l * ldh]);
                s = cabs1f(h11s) + fabsf(h21);
                h11s = h11s / s;
                h21 = h21 / s;
                v[0] = h11s;
                v[1] = h21;
            }

            /* Single-shift QR step */
            for (k = m; k <= i - 1; k++) {
                if (k > m)
                    cblas_ccopy(2, &H[k + (k - 1) * ldh], 1, v, 1);

                clarfg(2, &v[0], &v[1], 1, &t1);

                if (k > m) {
                    H[k + (k - 1) * ldh] = v[0];
                    H[(k + 1) + (k - 1) * ldh] = czero;
                }
                v2 = v[1];
                t2 = crealf(t1 * v2);

                /* Apply G from the left to transform the rows of the matrix
                 * in columns k to i2 */
                for (j = k; j <= i2; j++) {
                    sum = conjf(t1) * H[k + j * ldh] + t2 * H[(k + 1) + j * ldh];
                    H[k + j * ldh] = H[k + j * ldh] - sum;
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - sum * v2;
                }

                /* Apply G from the right to transform the columns of the
                 * matrix in rows i1 to min(k+2, i) */
                for (j = i1; j <= (k + 2 < i ? k + 2 : i); j++) {
                    sum = t1 * H[j + k * ldh] + t2 * H[j + (k + 1) * ldh];
                    H[j + k * ldh] = H[j + k * ldh] - sum;
                    H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - sum * conjf(v2);
                }

                if (wantz) {
                    for (j = iloz; j <= ihiz; j++) {
                        sum = t1 * Z[j + k * ldz] + t2 * Z[j + (k + 1) * ldz];
                        Z[j + k * ldz] = Z[j + k * ldz] - sum;
                        Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - sum * conjf(v2);
                    }
                }

                if (k == m && m > l) {
                    temp = cone - t1;
                    temp = temp / cabsf(temp);
                    H[(m + 1) + m * ldh] = H[(m + 1) + m * ldh] * conjf(temp);
                    if (m + 2 <= i)
                        H[(m + 2) + (m + 1) * ldh] = H[(m + 2) + (m + 1) * ldh] * temp;
                    for (j = m; j <= i; j++) {
                        if (j != m + 1) {
                            if (i2 > j) {
                                cblas_cscal(i2 - j, &temp, &H[j + (j + 1) * ldh], ldh);
                            }
                            {
                                c64 ct = conjf(temp);
                                cblas_cscal(j - i1, &ct, &H[i1 + j * ldh], 1);
                            }
                            if (wantz) {
                                c64 ct = conjf(temp);
                                cblas_cscal(nz, &ct, &Z[iloz + j * ldz], 1);
                            }
                        }
                    }
                }
            }

            /* Ensure that H(I,I-1) is real */
            temp = H[i + (i - 1) * ldh];
            if (cimagf(temp) != rzero) {
                rtemp = cabsf(temp);
                H[i + (i - 1) * ldh] = rtemp;
                temp = temp / rtemp;
                if (i2 > i) {
                    c64 ct = conjf(temp);
                    cblas_cscal(i2 - i, &ct, &H[i + (i + 1) * ldh], ldh);
                }
                cblas_cscal(i - i1, &temp, &H[i1 + i * ldh], 1);
                if (wantz) {
                    cblas_cscal(nz, &temp, &Z[iloz + i * ldz], 1);
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
