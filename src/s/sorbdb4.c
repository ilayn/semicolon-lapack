/**
 * @file sorbdb4.c
 * @brief SORBDB4 simultaneously bidiagonalizes the blocks of a tall and skinny matrix with orthonormal columns.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SORBDB4 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P,
 * M-P, or Q.
 *
 * @param[in] m
 *          The number of rows X11 plus the number of rows in X21.
 *
 * @param[in] p
 *          The number of rows in X11. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in X11 and X21. 0 <= q <= m and
 *          m-q <= min(p, m-p, q).
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
 *          Double precision array, dimension (m-q).
 *          The scalar factors of the elementary reflectors that define P1.
 *
 * @param[out] taup2
 *          Double precision array, dimension (m-q).
 *          The scalar factors of the elementary reflectors that define P2.
 *
 * @param[out] tauq1
 *          Double precision array, dimension (q).
 *          The scalar factors of the elementary reflectors that define Q1.
 *
 * @param[out] phantom
 *          Double precision array, dimension (m).
 *          The routine computes an M-by-1 column vector Y that is
 *          orthogonal to the columns of [ X11; X21 ]. PHANTOM(1:P) and
 *          PHANTOM(P+1:M) contain Householder vectors for Y(1:P) and
 *          Y(P+1:M), respectively.
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
void sorbdb4(
    const int m,
    const int p,
    const int q,
    f32* const restrict X11,
    const int ldx11,
    f32* const restrict X21,
    const int ldx21,
    f32* restrict theta,
    f32* restrict phi,
    f32* restrict taup1,
    f32* restrict taup2,
    f32* restrict tauq1,
    f32* restrict phantom,
    f32* restrict work,
    const int lwork,
    int* info)
{
    const f32 negone = -1.0f;
    const f32 zero = 0.0f;
    f32 c, s;
    int childinfo, i, ilarf, iorbdb5, j, llarf, lorbdb5, lworkmin, lworkopt;
    int lquery;
    int max_val;

    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (p < m - q || m - p < m - q) {
        *info = -2;
    } else if (q < m - q || q > m) {
        *info = -3;
    } else if (ldx11 < (1 > p ? 1 : p)) {
        *info = -5;
    } else if (ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -7;
    }

    if (*info == 0) {
        ilarf = 1;
        max_val = q - 1;
        if (p - 1 > max_val) max_val = p - 1;
        if (m - p - 1 > max_val) max_val = m - p - 1;
        llarf = max_val;
        iorbdb5 = 1;
        lorbdb5 = q;
        lworkopt = ilarf + llarf;
        if (iorbdb5 + lorbdb5 > lworkopt) lworkopt = iorbdb5 + lorbdb5;
        lworkmin = lworkopt;
        work[0] = (f32)lworkopt;
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("SORBDB4", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    for (i = 0; i < m - q; i++) {

        if (i == 0) {
            for (j = 0; j < m; j++) {
                phantom[j] = zero;
            }
            sorbdb5(p, m - p, q, &phantom[0], 1, &phantom[p], 1,
                    X11, ldx11, X21, ldx21, &work[iorbdb5], lorbdb5, &childinfo);
            cblas_sscal(p, negone, &phantom[0], 1);
            slarfgp(p, &phantom[0], &phantom[1], 1, &taup1[0]);
            slarfgp(m - p, &phantom[p], &phantom[p + 1], 1, &taup2[0]);
            theta[i] = atan2f(phantom[0], phantom[p]);
            c = cosf(theta[i]);
            s = sinf(theta[i]);
            slarf1f("L", p, q, &phantom[0], 1, taup1[0], X11, ldx11, &work[ilarf]);
            slarf1f("L", m - p, q, &phantom[p], 1, taup2[0], X21, ldx21, &work[ilarf]);
        } else {
            sorbdb5(p - i, m - p - i, q - i,
                    &X11[i + (i - 1) * ldx11], 1, &X21[i + (i - 1) * ldx21], 1,
                    &X11[i + i * ldx11], ldx11, &X21[i + i * ldx21], ldx21,
                    &work[iorbdb5], lorbdb5, &childinfo);
            cblas_sscal(p - i, negone, &X11[i + (i - 1) * ldx11], 1);
            slarfgp(p - i, &X11[i + (i - 1) * ldx11], &X11[(i + 1) + (i - 1) * ldx11], 1,
                    &taup1[i]);
            slarfgp(m - p - i, &X21[i + (i - 1) * ldx21], &X21[(i + 1) + (i - 1) * ldx21], 1,
                    &taup2[i]);
            theta[i] = atan2f(X11[i + (i - 1) * ldx11], X21[i + (i - 1) * ldx21]);
            c = cosf(theta[i]);
            s = sinf(theta[i]);
            slarf1f("L", p - i, q - i, &X11[i + (i - 1) * ldx11], 1, taup1[i],
                    &X11[i + i * ldx11], ldx11, &work[ilarf]);
            slarf1f("L", m - p - i, q - i, &X21[i + (i - 1) * ldx21], 1, taup2[i],
                    &X21[i + i * ldx21], ldx21, &work[ilarf]);
        }

        cblas_srot(q - i, &X11[i + i * ldx11], ldx11, &X21[i + i * ldx21], ldx21, s, -c);
        slarfgp(q - i, &X21[i + i * ldx21], &X21[i + (i + 1) * ldx21], ldx21, &tauq1[i]);
        c = X21[i + i * ldx21];
        slarf1f("R", p - i - 1, q - i, &X21[i + i * ldx21], ldx21, tauq1[i],
                &X11[(i + 1) + i * ldx11], ldx11, &work[ilarf]);
        slarf1f("R", m - p - i - 1, q - i, &X21[i + i * ldx21], ldx21, tauq1[i],
                &X21[(i + 1) + i * ldx21], ldx21, &work[ilarf]);
        if (i < m - q - 1) {
            f32 nrm1 = cblas_snrm2(p - i - 1, &X11[(i + 1) + i * ldx11], 1);
            f32 nrm2 = cblas_snrm2(m - p - i - 1, &X21[(i + 1) + i * ldx21], 1);
            s = sqrtf(nrm1 * nrm1 + nrm2 * nrm2);
            phi[i] = atan2f(s, c);
        }

    }

    for (i = m - q; i < p; i++) {
        slarfgp(q - i, &X11[i + i * ldx11], &X11[i + (i + 1) * ldx11], ldx11, &tauq1[i]);
        slarf1f("R", p - i - 1, q - i, &X11[i + i * ldx11], ldx11, tauq1[i],
                &X11[(i + 1) + i * ldx11], ldx11, &work[ilarf]);
        slarf1f("R", q - p, q - i, &X11[i + i * ldx11], ldx11, tauq1[i],
                &X21[(m - q) + i * ldx21], ldx21, &work[ilarf]);
    }

    for (i = p; i < q; i++) {
        slarfgp(q - i, &X21[(m - q + i - p) + i * ldx21],
                &X21[(m - q + i - p) + (i + 1) * ldx21], ldx21, &tauq1[i]);
        slarf1f("R", q - i - 1, q - i, &X21[(m - q + i - p) + i * ldx21], ldx21, tauq1[i],
                &X21[(m - q + i - p + 1) + i * ldx21], ldx21, &work[ilarf]);
    }
}
