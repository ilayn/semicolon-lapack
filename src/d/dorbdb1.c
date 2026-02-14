/**
 * @file dorbdb1.c
 * @brief DORBDB1 simultaneously bidiagonalizes the blocks of a tall and skinny matrix with orthonormal columns.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORBDB1 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. Q must be no larger than P,
 * M-P, or M-Q.
 *
 * @param[in] m
 *          The number of rows X11 plus the number of rows in X21.
 *
 * @param[in] p
 *          The number of rows in X11. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in X11 and X21. 0 <= q <= min(p, m-p, m-q).
 *
 * @param[in,out] X11
 *          Double precision array, dimension (ldx11, q).
 *          On entry, the top block of the matrix X to be reduced.
 *          On exit, the columns of tril(X11) specify reflectors for P1
 *          and the rows of triu(X11,1) specify reflectors for Q1.
 *
 * @param[in] ldx11
 *          The leading dimension of X11. ldx11 >= p.
 *
 * @param[in,out] X21
 *          Double precision array, dimension (ldx21, q).
 *          On entry, the bottom block of the matrix X to be reduced.
 *          On exit, the columns of tril(X21) specify reflectors for P2.
 *
 * @param[in] ldx21
 *          The leading dimension of X21. ldx21 >= m-p.
 *
 * @param[out] theta
 *          Double precision array, dimension (q).
 *          The entries of the bidiagonal blocks B11, B21 are defined by
 *          theta and phi.
 *
 * @param[out] phi
 *          Double precision array, dimension (q-1).
 *          The entries of the bidiagonal blocks B11, B21 are defined by
 *          theta and phi.
 *
 * @param[out] taup1
 *          Double precision array, dimension (p).
 *          The scalar factors of the elementary reflectors that define P1.
 *
 * @param[out] taup2
 *          Double precision array, dimension (m-p).
 *          The scalar factors of the elementary reflectors that define P2.
 *
 * @param[out] tauq1
 *          Double precision array, dimension (q).
 *          The scalar factors of the elementary reflectors that define Q1.
 *
 * @param[out] work
 *          Double precision array, dimension (lwork).
 *
 * @param[in] lwork
 *          The dimension of the array work. lwork >= m-q.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dorbdb1(
    const int m,
    const int p,
    const int q,
    f64* restrict X11,
    const int ldx11,
    f64* restrict X21,
    const int ldx21,
    f64* restrict theta,
    f64* restrict phi,
    f64* restrict taup1,
    f64* restrict taup2,
    f64* restrict tauq1,
    f64* restrict work,
    const int lwork,
    int* info)
{
    f64 c, s;
    int childinfo, i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt;
    int lquery;
    int max_val;

    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (p < q || m - p < q) {
        *info = -2;
    } else if (q < 0 || m - q < q) {
        *info = -3;
    } else if (ldx11 < (1 > p ? 1 : p)) {
        *info = -5;
    } else if (ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -7;
    }

    if (*info == 0) {
        ilarf = 1;
        max_val = p - 1;
        if (m - p - 1 > max_val) max_val = m - p - 1;
        if (q - 1 > max_val) max_val = q - 1;
        llarf = max_val;
        iorbdb5 = 1;
        lorbdb5 = q - 2;
        lworkopt = (ilarf + llarf > iorbdb5 + lorbdb5) ? (ilarf + llarf) : (iorbdb5 + lorbdb5);
        lworkmin = lworkopt;
        work[0] = (f64)lworkopt;
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("DORBDB1", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    for (i = 0; i < q; i++) {

        dlarfgp(p - i, &X11[i + i * ldx11], &X11[(i + 1) + i * ldx11], 1, &taup1[i]);
        dlarfgp(m - p - i, &X21[i + i * ldx21], &X21[(i + 1) + i * ldx21], 1, &taup2[i]);
        theta[i] = atan2(X21[i + i * ldx21], X11[i + i * ldx11]);
        c = cos(theta[i]);
        s = sin(theta[i]);
        dlarf1f("L", p - i, q - i - 1, &X11[i + i * ldx11], 1, taup1[i],
                &X11[i + (i + 1) * ldx11], ldx11, &work[ilarf]);
        dlarf1f("L", m - p - i, q - i - 1, &X21[i + i * ldx21], 1, taup2[i],
                &X21[i + (i + 1) * ldx21], ldx21, &work[ilarf]);

        if (i < q - 1) {
            cblas_drot(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                       &X21[i + (i + 1) * ldx21], ldx21, c, s);
            dlarfgp(q - i - 1, &X21[i + (i + 1) * ldx21], &X21[i + (i + 2) * ldx21],
                    ldx21, &tauq1[i]);
            s = X21[i + (i + 1) * ldx21];
            dlarf1f("R", p - i - 1, q - i - 1, &X21[i + (i + 1) * ldx21], ldx21,
                    tauq1[i], &X11[(i + 1) + (i + 1) * ldx11], ldx11, &work[ilarf]);
            dlarf1f("R", m - p - i - 1, q - i - 1, &X21[i + (i + 1) * ldx21], ldx21,
                    tauq1[i], &X21[(i + 1) + (i + 1) * ldx21], ldx21, &work[ilarf]);
            f64 nrm1 = cblas_dnrm2(p - i - 1, &X11[(i + 1) + (i + 1) * ldx11], 1);
            f64 nrm2 = cblas_dnrm2(m - p - i - 1, &X21[(i + 1) + (i + 1) * ldx21], 1);
            c = sqrt(nrm1 * nrm1 + nrm2 * nrm2);
            phi[i] = atan2(s, c);
            dorbdb5(p - i - 1, m - p - i - 1, q - i - 2,
                    &X11[(i + 1) + (i + 1) * ldx11], 1,
                    &X21[(i + 1) + (i + 1) * ldx21], 1,
                    &X11[(i + 1) + (i + 2) * ldx11], ldx11,
                    &X21[(i + 1) + (i + 2) * ldx21], ldx21,
                    &work[iorbdb5], lorbdb5, &childinfo);
        }

    }
}
