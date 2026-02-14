/**
 * @file zbbcsd.c
 * @brief ZBBCSD computes the CS decomposition of a unitary matrix in bidiagonal-block form.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZBBCSD computes the CS decomposition of a unitary matrix in
 * bidiagonal-block form.
 *
 * X is M-by-M, its top-left block is P-by-Q, and Q must be no larger
 * than P, M-P, or M-Q.
 *
 * @param[in] jobu1
 *          = 'Y': U1 is updated; otherwise: U1 is not updated.
 *
 * @param[in] jobu2
 *          = 'Y': U2 is updated; otherwise: U2 is not updated.
 *
 * @param[in] jobv1t
 *          = 'Y': V1T is updated; otherwise: V1T is not updated.
 *
 * @param[in] jobv2t
 *          = 'Y': V2T is updated; otherwise: V2T is not updated.
 *
 * @param[in] trans
 *          = 'T': X, U1, U2, V1T, and V2T are stored in row-major order;
 *          otherwise: they are stored in column-major order.
 *
 * @param[in] m
 *          The number of rows and columns in X.
 *
 * @param[in] p
 *          The number of rows in the top-left block of X. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in the top-left block of X.
 *          0 <= q <= min(p, m-p, m-q).
 *
 * @param[in,out] theta
 *          Double precision array, dimension (q).
 *
 * @param[in,out] phi
 *          Double precision array, dimension (q-1).
 *
 * @param[in,out] U1
 *          Complex*16 array, dimension (ldu1, p).
 *
 * @param[in] ldu1
 *          The leading dimension of U1. ldu1 >= max(1, p).
 *
 * @param[in,out] U2
 *          Complex*16 array, dimension (ldu2, m-p).
 *
 * @param[in] ldu2
 *          The leading dimension of U2. ldu2 >= max(1, m-p).
 *
 * @param[in,out] V1T
 *          Complex*16 array, dimension (ldv1t, q).
 *
 * @param[in] ldv1t
 *          The leading dimension of V1T. ldv1t >= max(1, q).
 *
 * @param[in,out] V2T
 *          Complex*16 array, dimension (ldv2t, m-q).
 *
 * @param[in] ldv2t
 *          The leading dimension of V2T. ldv2t >= max(1, m-q).
 *
 * @param[out] B11D
 *          Double precision array, dimension (q).
 *
 * @param[out] B11E
 *          Double precision array, dimension (q-1).
 *
 * @param[out] B12D
 *          Double precision array, dimension (q).
 *
 * @param[out] B12E
 *          Double precision array, dimension (q-1).
 *
 * @param[out] B21D
 *          Double precision array, dimension (q).
 *
 * @param[out] B21E
 *          Double precision array, dimension (q-1).
 *
 * @param[out] B22D
 *          Double precision array, dimension (q).
 *
 * @param[out] B22E
 *          Double precision array, dimension (q-1).
 *
 * @param[out] rwork
 *          Double precision array, dimension (lrwork).
 *
 * @param[in] lrwork
 *          The dimension of the array rwork. lrwork >= max(1, 8*q).
 *          If lrwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if ZBBCSD did not converge, info specifies the number
 *                           of nonzero entries in PHI.
 */
void zbbcsd(
    const char* jobu1,
    const char* jobu2,
    const char* jobv1t,
    const char* jobv2t,
    const char* trans,
    const int m,
    const int p,
    const int q,
    f64* restrict theta,
    f64* restrict phi,
    c128* restrict U1,
    const int ldu1,
    c128* restrict U2,
    const int ldu2,
    c128* restrict V1T,
    const int ldv1t,
    c128* restrict V2T,
    const int ldv2t,
    f64* restrict B11D,
    f64* restrict B11E,
    f64* restrict B12D,
    f64* restrict B12E,
    f64* restrict B21D,
    f64* restrict B21E,
    f64* restrict B22D,
    f64* restrict B22E,
    f64* restrict rwork,
    const int lrwork,
    int* info)
{
    const int maxitr = 6;
    const f64 hundred = 100.0;
    const f64 meighth = -0.125;
    const f64 one = 1.0;
    const f64 ten = 10.0;
    const f64 zero = 0.0;
    const f64 piover2 = 1.57079632679489661923132169163975144210;
    const c128 negonecomplex = CMPLX(-1.0, 0.0);

    int colmajor, lquery, restart11, restart12, restart21, restart22;
    int wantu1, wantu2, wantv1t, wantv2t;
    int i, imin, imax, iter, iu1cs, iu1sn, iu2cs, iu2sn;
    int iv1tcs, iv1tsn, iv2tcs, iv2tsn, j, lrworkmin, lrworkopt, maxit, mini;
    f64 b11bulge, b12bulge, b21bulge, b22bulge, dummy;
    f64 eps, mu, nu, r, sigma11, sigma21;
    f64 temp, thetamax, thetamin, thresh, tol, tolmul, unfl;
    f64 x1, x2, y1, y2;

    *info = 0;
    lquery = (lrwork == -1);
    wantu1 = (jobu1[0] == 'Y' || jobu1[0] == 'y');
    wantu2 = (jobu2[0] == 'Y' || jobu2[0] == 'y');
    wantv1t = (jobv1t[0] == 'Y' || jobv1t[0] == 'y');
    wantv2t = (jobv2t[0] == 'Y' || jobv2t[0] == 'y');
    colmajor = !(trans[0] == 'T' || trans[0] == 't');

    if (m < 0) {
        *info = -6;
    } else if (p < 0 || p > m) {
        *info = -7;
    } else if (q < 0 || q > m) {
        *info = -8;
    } else if (q > p || q > m - p || q > m - q) {
        *info = -8;
    } else if (wantu1 && ldu1 < p) {
        *info = -12;
    } else if (wantu2 && ldu2 < m - p) {
        *info = -14;
    } else if (wantv1t && ldv1t < q) {
        *info = -16;
    } else if (wantv2t && ldv2t < m - q) {
        *info = -18;
    }

    if (*info == 0 && q == 0) {
        lrworkmin = 1;
        rwork[0] = (f64)lrworkmin;
        return;
    }

    if (*info == 0) {
        iu1cs = 0;
        iu1sn = iu1cs + q;
        iu2cs = iu1sn + q;
        iu2sn = iu2cs + q;
        iv1tcs = iu2sn + q;
        iv1tsn = iv1tcs + q;
        iv2tcs = iv1tsn + q;
        iv2tsn = iv2tcs + q;
        lrworkopt = iv2tsn + q;
        lrworkmin = lrworkopt;
        rwork[0] = (f64)lrworkopt;
        if (lrwork < lrworkmin && !lquery) {
            *info = -28;
        }
    }

    if (*info != 0) {
        xerbla("ZBBCSD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    tolmul = (ten > (hundred < pow(eps, meighth) ? hundred : pow(eps, meighth))) ?
              ten : (hundred < pow(eps, meighth) ? hundred : pow(eps, meighth));
    tol = tolmul * eps;
    thresh = (tol > maxitr * q * q * unfl) ? tol : (maxitr * q * q * unfl);

    for (i = 0; i < q; i++) {
        if (theta[i] < thresh) {
            theta[i] = zero;
        } else if (theta[i] > piover2 - thresh) {
            theta[i] = piover2;
        }
    }
    for (i = 0; i < q - 1; i++) {
        if (phi[i] < thresh) {
            phi[i] = zero;
        } else if (phi[i] > piover2 - thresh) {
            phi[i] = piover2;
        }
    }

    imax = q;
    while (imax > 1) {
        if (phi[imax - 2] != zero) {
            break;
        }
        imax = imax - 1;
    }
    imin = imax - 1;
    if (imin > 1) {
        while (phi[imin - 2] != zero) {
            imin = imin - 1;
            if (imin <= 1) break;
        }
    }

    maxit = maxitr * q * q;
    iter = 0;

    while (imax > 1) {

        B11D[imin - 1] = cos(theta[imin - 1]);
        B21D[imin - 1] = -sin(theta[imin - 1]);
        for (i = imin - 1; i < imax - 1; i++) {
            B11E[i] = -sin(theta[i]) * sin(phi[i]);
            B11D[i + 1] = cos(theta[i + 1]) * cos(phi[i]);
            B12D[i] = sin(theta[i]) * cos(phi[i]);
            B12E[i] = cos(theta[i + 1]) * sin(phi[i]);
            B21E[i] = -cos(theta[i]) * sin(phi[i]);
            B21D[i + 1] = -sin(theta[i + 1]) * cos(phi[i]);
            B22D[i] = cos(theta[i]) * cos(phi[i]);
            B22E[i] = -sin(theta[i + 1]) * sin(phi[i]);
        }
        B12D[imax - 1] = sin(theta[imax - 1]);
        B22D[imax - 1] = cos(theta[imax - 1]);

        if (iter > maxit) {
            *info = 0;
            for (i = 0; i < q; i++) {
                if (phi[i] != zero)
                    (*info)++;
            }
            return;
        }

        iter = iter + imax - imin;

        thetamax = theta[imin - 1];
        thetamin = theta[imin - 1];
        for (i = imin; i < imax; i++) {
            if (theta[i] > thetamax)
                thetamax = theta[i];
            if (theta[i] < thetamin)
                thetamin = theta[i];
        }

        if (thetamax > piover2 - thresh) {
            mu = zero;
            nu = one;
        } else if (thetamin < thresh) {
            mu = one;
            nu = zero;
        } else {
            dlas2(B11D[imax - 2], B11E[imax - 2], B11D[imax - 1], &sigma11, &dummy);
            dlas2(B21D[imax - 2], B21E[imax - 2], B21D[imax - 1], &sigma21, &dummy);

            if (sigma11 <= sigma21) {
                mu = sigma11;
                nu = sqrt(one - mu * mu);
                if (mu < thresh) {
                    mu = zero;
                    nu = one;
                }
            } else {
                nu = sigma21;
                mu = sqrt(1.0 - nu * nu);
                if (nu < thresh) {
                    mu = one;
                    nu = zero;
                }
            }
        }

        if (mu <= nu) {
            dlartgs(B11D[imin - 1], B11E[imin - 1], mu,
                    &rwork[iv1tcs + imin - 1], &rwork[iv1tsn + imin - 1]);
        } else {
            dlartgs(B21D[imin - 1], B21E[imin - 1], nu,
                    &rwork[iv1tcs + imin - 1], &rwork[iv1tsn + imin - 1]);
        }

        temp = rwork[iv1tcs + imin - 1] * B11D[imin - 1] +
               rwork[iv1tsn + imin - 1] * B11E[imin - 1];
        B11E[imin - 1] = rwork[iv1tcs + imin - 1] * B11E[imin - 1] -
                         rwork[iv1tsn + imin - 1] * B11D[imin - 1];
        B11D[imin - 1] = temp;
        b11bulge = rwork[iv1tsn + imin - 1] * B11D[imin];
        B11D[imin] = rwork[iv1tcs + imin - 1] * B11D[imin];
        temp = rwork[iv1tcs + imin - 1] * B21D[imin - 1] +
               rwork[iv1tsn + imin - 1] * B21E[imin - 1];
        B21E[imin - 1] = rwork[iv1tcs + imin - 1] * B21E[imin - 1] -
                         rwork[iv1tsn + imin - 1] * B21D[imin - 1];
        B21D[imin - 1] = temp;
        b21bulge = rwork[iv1tsn + imin - 1] * B21D[imin];
        B21D[imin] = rwork[iv1tcs + imin - 1] * B21D[imin];

        theta[imin - 1] = atan2(sqrt(B21D[imin - 1] * B21D[imin - 1] + b21bulge * b21bulge),
                                sqrt(B11D[imin - 1] * B11D[imin - 1] + b11bulge * b11bulge));

        if (B11D[imin - 1] * B11D[imin - 1] + b11bulge * b11bulge > thresh * thresh) {
            dlartgp(b11bulge, B11D[imin - 1], &rwork[iu1sn + imin - 1],
                    &rwork[iu1cs + imin - 1], &r);
        } else if (mu <= nu) {
            dlartgs(B11E[imin - 1], B11D[imin], mu,
                    &rwork[iu1cs + imin - 1], &rwork[iu1sn + imin - 1]);
        } else {
            dlartgs(B12D[imin - 1], B12E[imin - 1], nu,
                    &rwork[iu1cs + imin - 1], &rwork[iu1sn + imin - 1]);
        }
        if (B21D[imin - 1] * B21D[imin - 1] + b21bulge * b21bulge > thresh * thresh) {
            dlartgp(b21bulge, B21D[imin - 1], &rwork[iu2sn + imin - 1],
                    &rwork[iu2cs + imin - 1], &r);
        } else if (nu < mu) {
            dlartgs(B21E[imin - 1], B21D[imin], nu,
                    &rwork[iu2cs + imin - 1], &rwork[iu2sn + imin - 1]);
        } else {
            dlartgs(B22D[imin - 1], B22E[imin - 1], mu,
                    &rwork[iu2cs + imin - 1], &rwork[iu2sn + imin - 1]);
        }
        rwork[iu2cs + imin - 1] = -rwork[iu2cs + imin - 1];
        rwork[iu2sn + imin - 1] = -rwork[iu2sn + imin - 1];

        temp = rwork[iu1cs + imin - 1] * B11E[imin - 1] +
               rwork[iu1sn + imin - 1] * B11D[imin];
        B11D[imin] = rwork[iu1cs + imin - 1] * B11D[imin] -
                     rwork[iu1sn + imin - 1] * B11E[imin - 1];
        B11E[imin - 1] = temp;
        if (imax > imin + 1) {
            b11bulge = rwork[iu1sn + imin - 1] * B11E[imin];
            B11E[imin] = rwork[iu1cs + imin - 1] * B11E[imin];
        }
        temp = rwork[iu1cs + imin - 1] * B12D[imin - 1] +
               rwork[iu1sn + imin - 1] * B12E[imin - 1];
        B12E[imin - 1] = rwork[iu1cs + imin - 1] * B12E[imin - 1] -
                         rwork[iu1sn + imin - 1] * B12D[imin - 1];
        B12D[imin - 1] = temp;
        b12bulge = rwork[iu1sn + imin - 1] * B12D[imin];
        B12D[imin] = rwork[iu1cs + imin - 1] * B12D[imin];
        temp = rwork[iu2cs + imin - 1] * B21E[imin - 1] +
               rwork[iu2sn + imin - 1] * B21D[imin];
        B21D[imin] = rwork[iu2cs + imin - 1] * B21D[imin] -
                     rwork[iu2sn + imin - 1] * B21E[imin - 1];
        B21E[imin - 1] = temp;
        if (imax > imin + 1) {
            b21bulge = rwork[iu2sn + imin - 1] * B21E[imin];
            B21E[imin] = rwork[iu2cs + imin - 1] * B21E[imin];
        }
        temp = rwork[iu2cs + imin - 1] * B22D[imin - 1] +
               rwork[iu2sn + imin - 1] * B22E[imin - 1];
        B22E[imin - 1] = rwork[iu2cs + imin - 1] * B22E[imin - 1] -
                         rwork[iu2sn + imin - 1] * B22D[imin - 1];
        B22D[imin - 1] = temp;
        b22bulge = rwork[iu2sn + imin - 1] * B22D[imin];
        B22D[imin] = rwork[iu2cs + imin - 1] * B22D[imin];

        for (i = imin; i < imax - 1; i++) {

            x1 = sin(theta[i - 1]) * B11E[i - 1] + cos(theta[i - 1]) * B21E[i - 1];
            x2 = sin(theta[i - 1]) * b11bulge + cos(theta[i - 1]) * b21bulge;
            y1 = sin(theta[i - 1]) * B12D[i - 1] + cos(theta[i - 1]) * B22D[i - 1];
            y2 = sin(theta[i - 1]) * b12bulge + cos(theta[i - 1]) * b22bulge;

            phi[i - 1] = atan2(sqrt(x1 * x1 + x2 * x2), sqrt(y1 * y1 + y2 * y2));

            restart11 = (B11E[i - 1] * B11E[i - 1] + b11bulge * b11bulge <= thresh * thresh);
            restart21 = (B21E[i - 1] * B21E[i - 1] + b21bulge * b21bulge <= thresh * thresh);
            restart12 = (B12D[i - 1] * B12D[i - 1] + b12bulge * b12bulge <= thresh * thresh);
            restart22 = (B22D[i - 1] * B22D[i - 1] + b22bulge * b22bulge <= thresh * thresh);

            if (!restart11 && !restart21) {
                dlartgp(x2, x1, &rwork[iv1tsn + i], &rwork[iv1tcs + i], &r);
            } else if (!restart11 && restart21) {
                dlartgp(b11bulge, B11E[i - 1], &rwork[iv1tsn + i], &rwork[iv1tcs + i], &r);
            } else if (restart11 && !restart21) {
                dlartgp(b21bulge, B21E[i - 1], &rwork[iv1tsn + i], &rwork[iv1tcs + i], &r);
            } else if (mu <= nu) {
                dlartgs(B11D[i], B11E[i], mu, &rwork[iv1tcs + i], &rwork[iv1tsn + i]);
            } else {
                dlartgs(B21D[i], B21E[i], nu, &rwork[iv1tcs + i], &rwork[iv1tsn + i]);
            }
            rwork[iv1tcs + i] = -rwork[iv1tcs + i];
            rwork[iv1tsn + i] = -rwork[iv1tsn + i];
            if (!restart12 && !restart22) {
                dlartgp(y2, y1, &rwork[iv2tsn + i - 1], &rwork[iv2tcs + i - 1], &r);
            } else if (!restart12 && restart22) {
                dlartgp(b12bulge, B12D[i - 1], &rwork[iv2tsn + i - 1], &rwork[iv2tcs + i - 1], &r);
            } else if (restart12 && !restart22) {
                dlartgp(b22bulge, B22D[i - 1], &rwork[iv2tsn + i - 1], &rwork[iv2tcs + i - 1], &r);
            } else if (nu < mu) {
                dlartgs(B12E[i - 1], B12D[i], nu, &rwork[iv2tcs + i - 1], &rwork[iv2tsn + i - 1]);
            } else {
                dlartgs(B22E[i - 1], B22D[i], mu, &rwork[iv2tcs + i - 1], &rwork[iv2tsn + i - 1]);
            }

            temp = rwork[iv1tcs + i] * B11D[i] + rwork[iv1tsn + i] * B11E[i];
            B11E[i] = rwork[iv1tcs + i] * B11E[i] - rwork[iv1tsn + i] * B11D[i];
            B11D[i] = temp;
            b11bulge = rwork[iv1tsn + i] * B11D[i + 1];
            B11D[i + 1] = rwork[iv1tcs + i] * B11D[i + 1];
            temp = rwork[iv1tcs + i] * B21D[i] + rwork[iv1tsn + i] * B21E[i];
            B21E[i] = rwork[iv1tcs + i] * B21E[i] - rwork[iv1tsn + i] * B21D[i];
            B21D[i] = temp;
            b21bulge = rwork[iv1tsn + i] * B21D[i + 1];
            B21D[i + 1] = rwork[iv1tcs + i] * B21D[i + 1];
            temp = rwork[iv2tcs + i - 1] * B12E[i - 1] + rwork[iv2tsn + i - 1] * B12D[i];
            B12D[i] = rwork[iv2tcs + i - 1] * B12D[i] - rwork[iv2tsn + i - 1] * B12E[i - 1];
            B12E[i - 1] = temp;
            b12bulge = rwork[iv2tsn + i - 1] * B12E[i];
            B12E[i] = rwork[iv2tcs + i - 1] * B12E[i];
            temp = rwork[iv2tcs + i - 1] * B22E[i - 1] + rwork[iv2tsn + i - 1] * B22D[i];
            B22D[i] = rwork[iv2tcs + i - 1] * B22D[i] - rwork[iv2tsn + i - 1] * B22E[i - 1];
            B22E[i - 1] = temp;
            b22bulge = rwork[iv2tsn + i - 1] * B22E[i];
            B22E[i] = rwork[iv2tcs + i - 1] * B22E[i];

            x1 = cos(phi[i - 1]) * B11D[i] + sin(phi[i - 1]) * B12E[i - 1];
            x2 = cos(phi[i - 1]) * b11bulge + sin(phi[i - 1]) * b12bulge;
            y1 = cos(phi[i - 1]) * B21D[i] + sin(phi[i - 1]) * B22E[i - 1];
            y2 = cos(phi[i - 1]) * b21bulge + sin(phi[i - 1]) * b22bulge;

            theta[i] = atan2(sqrt(y1 * y1 + y2 * y2), sqrt(x1 * x1 + x2 * x2));

            restart11 = (B11D[i] * B11D[i] + b11bulge * b11bulge <= thresh * thresh);
            restart12 = (B12E[i - 1] * B12E[i - 1] + b12bulge * b12bulge <= thresh * thresh);
            restart21 = (B21D[i] * B21D[i] + b21bulge * b21bulge <= thresh * thresh);
            restart22 = (B22E[i - 1] * B22E[i - 1] + b22bulge * b22bulge <= thresh * thresh);

            if (!restart11 && !restart12) {
                dlartgp(x2, x1, &rwork[iu1sn + i], &rwork[iu1cs + i], &r);
            } else if (!restart11 && restart12) {
                dlartgp(b11bulge, B11D[i], &rwork[iu1sn + i], &rwork[iu1cs + i], &r);
            } else if (restart11 && !restart12) {
                dlartgp(b12bulge, B12E[i - 1], &rwork[iu1sn + i], &rwork[iu1cs + i], &r);
            } else if (mu <= nu) {
                dlartgs(B11E[i], B11D[i + 1], mu, &rwork[iu1cs + i], &rwork[iu1sn + i]);
            } else {
                dlartgs(B12D[i], B12E[i], nu, &rwork[iu1cs + i], &rwork[iu1sn + i]);
            }
            if (!restart21 && !restart22) {
                dlartgp(y2, y1, &rwork[iu2sn + i], &rwork[iu2cs + i], &r);
            } else if (!restart21 && restart22) {
                dlartgp(b21bulge, B21D[i], &rwork[iu2sn + i], &rwork[iu2cs + i], &r);
            } else if (restart21 && !restart22) {
                dlartgp(b22bulge, B22E[i - 1], &rwork[iu2sn + i], &rwork[iu2cs + i], &r);
            } else if (nu < mu) {
                dlartgs(B21E[i], B21E[i + 1], nu, &rwork[iu2cs + i], &rwork[iu2sn + i]);
            } else {
                dlartgs(B22D[i], B22E[i], mu, &rwork[iu2cs + i], &rwork[iu2sn + i]);
            }
            rwork[iu2cs + i] = -rwork[iu2cs + i];
            rwork[iu2sn + i] = -rwork[iu2sn + i];

            temp = rwork[iu1cs + i] * B11E[i] + rwork[iu1sn + i] * B11D[i + 1];
            B11D[i + 1] = rwork[iu1cs + i] * B11D[i + 1] - rwork[iu1sn + i] * B11E[i];
            B11E[i] = temp;
            if (i < imax - 2) {
                b11bulge = rwork[iu1sn + i] * B11E[i + 1];
                B11E[i + 1] = rwork[iu1cs + i] * B11E[i + 1];
            }
            temp = rwork[iu2cs + i] * B21E[i] + rwork[iu2sn + i] * B21D[i + 1];
            B21D[i + 1] = rwork[iu2cs + i] * B21D[i + 1] - rwork[iu2sn + i] * B21E[i];
            B21E[i] = temp;
            if (i < imax - 2) {
                b21bulge = rwork[iu2sn + i] * B21E[i + 1];
                B21E[i + 1] = rwork[iu2cs + i] * B21E[i + 1];
            }
            temp = rwork[iu1cs + i] * B12D[i] + rwork[iu1sn + i] * B12E[i];
            B12E[i] = rwork[iu1cs + i] * B12E[i] - rwork[iu1sn + i] * B12D[i];
            B12D[i] = temp;
            b12bulge = rwork[iu1sn + i] * B12D[i + 1];
            B12D[i + 1] = rwork[iu1cs + i] * B12D[i + 1];
            temp = rwork[iu2cs + i] * B22D[i] + rwork[iu2sn + i] * B22E[i];
            B22E[i] = rwork[iu2cs + i] * B22E[i] - rwork[iu2sn + i] * B22D[i];
            B22D[i] = temp;
            b22bulge = rwork[iu2sn + i] * B22D[i + 1];
            B22D[i + 1] = rwork[iu2cs + i] * B22D[i + 1];

        }

        x1 = sin(theta[imax - 2]) * B11E[imax - 2] +
             cos(theta[imax - 2]) * B21E[imax - 2];
        y1 = sin(theta[imax - 2]) * B12D[imax - 2] +
             cos(theta[imax - 2]) * B22D[imax - 2];
        y2 = sin(theta[imax - 2]) * b12bulge + cos(theta[imax - 2]) * b22bulge;

        phi[imax - 2] = atan2(fabs(x1), sqrt(y1 * y1 + y2 * y2));

        restart12 = (B12D[imax - 2] * B12D[imax - 2] + b12bulge * b12bulge <= thresh * thresh);
        restart22 = (B22D[imax - 2] * B22D[imax - 2] + b22bulge * b22bulge <= thresh * thresh);

        if (!restart12 && !restart22) {
            dlartgp(y2, y1, &rwork[iv2tsn + imax - 2], &rwork[iv2tcs + imax - 2], &r);
        } else if (!restart12 && restart22) {
            dlartgp(b12bulge, B12D[imax - 2], &rwork[iv2tsn + imax - 2],
                    &rwork[iv2tcs + imax - 2], &r);
        } else if (restart12 && !restart22) {
            dlartgp(b22bulge, B22D[imax - 2], &rwork[iv2tsn + imax - 2],
                    &rwork[iv2tcs + imax - 2], &r);
        } else if (nu < mu) {
            dlartgs(B12E[imax - 2], B12D[imax - 1], nu,
                    &rwork[iv2tcs + imax - 2], &rwork[iv2tsn + imax - 2]);
        } else {
            dlartgs(B22E[imax - 2], B22D[imax - 1], mu,
                    &rwork[iv2tcs + imax - 2], &rwork[iv2tsn + imax - 2]);
        }

        temp = rwork[iv2tcs + imax - 2] * B12E[imax - 2] +
               rwork[iv2tsn + imax - 2] * B12D[imax - 1];
        B12D[imax - 1] = rwork[iv2tcs + imax - 2] * B12D[imax - 1] -
                         rwork[iv2tsn + imax - 2] * B12E[imax - 2];
        B12E[imax - 2] = temp;
        temp = rwork[iv2tcs + imax - 2] * B22E[imax - 2] +
               rwork[iv2tsn + imax - 2] * B22D[imax - 1];
        B22D[imax - 1] = rwork[iv2tcs + imax - 2] * B22D[imax - 1] -
                         rwork[iv2tsn + imax - 2] * B22E[imax - 2];
        B22E[imax - 2] = temp;

        if (wantu1) {
            if (colmajor) {
                zlasr("R", "V", "F", p, imax - imin + 1,
                      &rwork[iu1cs + imin - 1], &rwork[iu1sn + imin - 1],
                      &U1[0 + (imin - 1) * ldu1], ldu1);
            } else {
                zlasr("L", "V", "F", imax - imin + 1, p,
                      &rwork[iu1cs + imin - 1], &rwork[iu1sn + imin - 1],
                      &U1[(imin - 1) + 0 * ldu1], ldu1);
            }
        }
        if (wantu2) {
            if (colmajor) {
                zlasr("R", "V", "F", m - p, imax - imin + 1,
                      &rwork[iu2cs + imin - 1], &rwork[iu2sn + imin - 1],
                      &U2[0 + (imin - 1) * ldu2], ldu2);
            } else {
                zlasr("L", "V", "F", imax - imin + 1, m - p,
                      &rwork[iu2cs + imin - 1], &rwork[iu2sn + imin - 1],
                      &U2[(imin - 1) + 0 * ldu2], ldu2);
            }
        }
        if (wantv1t) {
            if (colmajor) {
                zlasr("L", "V", "F", imax - imin + 1, q,
                      &rwork[iv1tcs + imin - 1], &rwork[iv1tsn + imin - 1],
                      &V1T[(imin - 1) + 0 * ldv1t], ldv1t);
            } else {
                zlasr("R", "V", "F", q, imax - imin + 1,
                      &rwork[iv1tcs + imin - 1], &rwork[iv1tsn + imin - 1],
                      &V1T[0 + (imin - 1) * ldv1t], ldv1t);
            }
        }
        if (wantv2t) {
            if (colmajor) {
                zlasr("L", "V", "F", imax - imin + 1, m - q,
                      &rwork[iv2tcs + imin - 1], &rwork[iv2tsn + imin - 1],
                      &V2T[(imin - 1) + 0 * ldv2t], ldv2t);
            } else {
                zlasr("R", "V", "F", m - q, imax - imin + 1,
                      &rwork[iv2tcs + imin - 1], &rwork[iv2tsn + imin - 1],
                      &V2T[0 + (imin - 1) * ldv2t], ldv2t);
            }
        }

        if (B11E[imax - 2] + B21E[imax - 2] > 0) {
            B11D[imax - 1] = -B11D[imax - 1];
            B21D[imax - 1] = -B21D[imax - 1];
            if (wantv1t) {
                if (colmajor) {
                    cblas_zscal(q, &negonecomplex, &V1T[(imax - 1) + 0 * ldv1t], ldv1t);
                } else {
                    cblas_zscal(q, &negonecomplex, &V1T[0 + (imax - 1) * ldv1t], 1);
                }
            }
        }

        x1 = cos(phi[imax - 2]) * B11D[imax - 1] +
             sin(phi[imax - 2]) * B12E[imax - 2];
        y1 = cos(phi[imax - 2]) * B21D[imax - 1] +
             sin(phi[imax - 2]) * B22E[imax - 2];

        theta[imax - 1] = atan2(fabs(y1), fabs(x1));

        if (B11D[imax - 1] + B12E[imax - 2] < 0) {
            B12D[imax - 1] = -B12D[imax - 1];
            if (wantu1) {
                if (colmajor) {
                    cblas_zscal(p, &negonecomplex, &U1[0 + (imax - 1) * ldu1], 1);
                } else {
                    cblas_zscal(p, &negonecomplex, &U1[(imax - 1) + 0 * ldu1], ldu1);
                }
            }
        }
        if (B21D[imax - 1] + B22E[imax - 2] > 0) {
            B22D[imax - 1] = -B22D[imax - 1];
            if (wantu2) {
                if (colmajor) {
                    cblas_zscal(m - p, &negonecomplex, &U2[0 + (imax - 1) * ldu2], 1);
                } else {
                    cblas_zscal(m - p, &negonecomplex, &U2[(imax - 1) + 0 * ldu2], ldu2);
                }
            }
        }

        if (B12D[imax - 1] + B22D[imax - 1] < 0) {
            if (wantv2t) {
                if (colmajor) {
                    cblas_zscal(m - q, &negonecomplex, &V2T[(imax - 1) + 0 * ldv2t], ldv2t);
                } else {
                    cblas_zscal(m - q, &negonecomplex, &V2T[0 + (imax - 1) * ldv2t], 1);
                }
            }
        }

        for (i = imin - 1; i < imax; i++) {
            if (theta[i] < thresh) {
                theta[i] = zero;
            } else if (theta[i] > piover2 - thresh) {
                theta[i] = piover2;
            }
        }
        for (i = imin - 1; i < imax - 1; i++) {
            if (phi[i] < thresh) {
                phi[i] = zero;
            } else if (phi[i] > piover2 - thresh) {
                phi[i] = piover2;
            }
        }

        if (imax > 1) {
            while (phi[imax - 2] == zero) {
                imax = imax - 1;
                if (imax <= 1) break;
            }
        }
        if (imin > imax - 1)
            imin = imax - 1;
        if (imin > 1) {
            while (phi[imin - 2] != zero) {
                imin = imin - 1;
                if (imin <= 1) break;
            }
        }

    }

    for (i = 0; i < q; i++) {

        mini = i;
        thetamin = theta[i];
        for (j = i + 1; j < q; j++) {
            if (theta[j] < thetamin) {
                mini = j;
                thetamin = theta[j];
            }
        }

        if (mini != i) {
            theta[mini] = theta[i];
            theta[i] = thetamin;
            if (colmajor) {
                if (wantu1)
                    cblas_zswap(p, &U1[0 + i * ldu1], 1, &U1[0 + mini * ldu1], 1);
                if (wantu2)
                    cblas_zswap(m - p, &U2[0 + i * ldu2], 1, &U2[0 + mini * ldu2], 1);
                if (wantv1t)
                    cblas_zswap(q, &V1T[i + 0 * ldv1t], ldv1t, &V1T[mini + 0 * ldv1t], ldv1t);
                if (wantv2t)
                    cblas_zswap(m - q, &V2T[i + 0 * ldv2t], ldv2t, &V2T[mini + 0 * ldv2t], ldv2t);
            } else {
                if (wantu1)
                    cblas_zswap(p, &U1[i + 0 * ldu1], ldu1, &U1[mini + 0 * ldu1], ldu1);
                if (wantu2)
                    cblas_zswap(m - p, &U2[i + 0 * ldu2], ldu2, &U2[mini + 0 * ldu2], ldu2);
                if (wantv1t)
                    cblas_zswap(q, &V1T[0 + i * ldv1t], 1, &V1T[0 + mini * ldv1t], 1);
                if (wantv2t)
                    cblas_zswap(m - q, &V2T[0 + i * ldv2t], 1, &V2T[0 + mini * ldv2t], 1);
            }
        }

    }
}
