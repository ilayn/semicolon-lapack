/**
 * @file zunbdb2.c
 * @brief ZUNBDB2 simultaneously bidiagonalizes the blocks of a tall and
 *        skinny matrix X with orthonormal columns.
 */

#include "semicolon_lapack_complex_double.h"
#include "semicolon_cblas.h"
#include <complex.h>
#include <math.h>

/**
 * ZUNBDB2 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. P must be no larger than M-P,
 * Q, or M-Q. Routines ZUNBDB1, ZUNBDB3, and ZUNBDB4 handle cases in
 * which P is not the minimum dimension.
 *
 * The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
 * and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
 * Householder vectors.
 *
 * B11 and B12 are P-by-P bidiagonal matrices represented implicitly by
 * angles THETA, PHI.
 *
 * @param[in]     m       The number of rows X11 plus the number of rows in X21.
 * @param[in]     p       The number of rows in X11. 0 <= P <= min(M-P,Q,M-Q).
 * @param[in]     q       The number of columns in X11 and X21. 0 <= Q <= M.
 * @param[in,out] X11     Complex array, dimension (ldx11, q).
 *                        On entry, the top block of the matrix X to be reduced.
 *                        On exit, the columns of tril(X11) specify reflectors
 *                        for P1 and the rows of triu(X11,1) specify reflectors
 *                        for Q1.
 * @param[in]     ldx11   The leading dimension of X11. ldx11 >= P.
 * @param[in,out] X21     Complex array, dimension (ldx21, q).
 *                        On entry, the bottom block of the matrix X to be
 *                        reduced. On exit, the columns of tril(X21) specify
 *                        reflectors for P2.
 * @param[in]     ldx21   The leading dimension of X21. ldx21 >= M-P.
 * @param[out]    theta   Double precision array, dimension (q).
 *                        The entries of the bidiagonal blocks B11, B21 are
 *                        defined by THETA and PHI. See Further Details.
 * @param[out]    phi     Double precision array, dimension (q-1).
 *                        The entries of the bidiagonal blocks B11, B21 are
 *                        defined by THETA and PHI. See Further Details.
 * @param[out]    taup1   Complex array, dimension (p-1).
 *                        The scalar factors of the elementary reflectors that
 *                        define P1.
 * @param[out]    taup2   Complex array, dimension (q).
 *                        The scalar factors of the elementary reflectors that
 *                        define P2.
 * @param[out]    tauq1   Complex array, dimension (q).
 *                        The scalar factors of the elementary reflectors that
 *                        define Q1.
 * @param[out]    work    Complex array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array WORK. LWORK >= M-Q.
 *                        If LWORK = -1, then a workspace query is assumed; the
 *                        routine only calculates the optimal size of the WORK
 *                        array, returns this value as the first entry of the
 *                        WORK array, and no error message related to LWORK is
 *                        issued by XERBLA.
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if INFO = -i, the i-th argument had an illegal
 *                        value.
 */
void zunbdb2(const INT m, const INT p, const INT q,
             c128* restrict X11, const INT ldx11,
             c128* restrict X21, const INT ldx21,
             f64* restrict theta,
             f64* restrict phi,
             c128* restrict taup1,
             c128* restrict taup2,
             c128* restrict tauq1,
             c128* restrict work, const INT lwork,
             INT* info)
{
    const c128 NEGONE = CMPLX(-1.0, 0.0);

    f64 c, s;
    INT childinfo, i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt;
    INT lquery;

    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (p < 0 || p > m - p) {
        *info = -2;
    } else if (q < 0 || q < p || m - q < p) {
        *info = -3;
    } else if (ldx11 < (1 > p ? 1 : p)) {
        *info = -5;
    } else if (ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -7;
    }

    if (*info == 0) {
        ilarf = 2;
        llarf = p - 1;
        if (m - p > llarf) llarf = m - p;
        if (q - 1 > llarf) llarf = q - 1;
        iorbdb5 = 2;
        lorbdb5 = q - 1;
        lworkopt = ilarf + llarf - 1;
        if (iorbdb5 + lorbdb5 - 1 > lworkopt)
            lworkopt = iorbdb5 + lorbdb5 - 1;
        lworkmin = lworkopt;
        work[0] = CMPLX((f64)lworkopt, 0.0);
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("ZUNBDB2", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Reduce rows 1, ..., P of X11 and X21 */

    for (i = 0; i < p; i++) {

        if (i > 0) {
            cblas_zdrot(q - i, &X11[i + i * ldx11], ldx11,
                       &X21[(i - 1) + i * ldx21], ldx21, c, s);
        }
        zlacgv(q - i, &X11[i + i * ldx11], ldx11);
        zlarfgp(q - i, &X11[i + i * ldx11],
                &X11[i + (i + 1) * ldx11], ldx11, &tauq1[i]);
        c = creal(X11[i + i * ldx11]);
        zlarf1f("R", p - i - 1, q - i, &X11[i + i * ldx11], ldx11,
                tauq1[i], &X11[(i + 1) + i * ldx11], ldx11,
                &work[ilarf - 1]);
        zlarf1f("R", m - p - i, q - i, &X11[i + i * ldx11], ldx11,
                tauq1[i], &X21[i + i * ldx21], ldx21,
                &work[ilarf - 1]);
        zlacgv(q - i, &X11[i + i * ldx11], ldx11);
        s = sqrt(pow(cblas_dznrm2(p - i - 1, &X11[(i + 1) + i * ldx11], 1), 2)
               + pow(cblas_dznrm2(m - p - i, &X21[i + i * ldx21], 1), 2));
        theta[i] = atan2(s, c);

        zunbdb5(p - i - 1, m - p - i, q - i - 1,
                &X11[(i + 1) + i * ldx11], 1,
                &X21[i + i * ldx21], 1,
                &X11[(i + 1) + (i + 1) * ldx11], ldx11,
                &X21[i + (i + 1) * ldx21], ldx21,
                &work[iorbdb5 - 1], lorbdb5, &childinfo);
        cblas_zscal(p - i - 1, &NEGONE, &X11[(i + 1) + i * ldx11], 1);
        zlarfgp(m - p - i, &X21[i + i * ldx21],
                &X21[(i + 1) + i * ldx21], 1, &taup2[i]);
        if (i < p - 1) {
            zlarfgp(p - i - 1, &X11[(i + 1) + i * ldx11],
                    &X11[(i + 2) + i * ldx11], 1, &taup1[i]);
            phi[i] = atan2(creal(X11[(i + 1) + i * ldx11]),
                           creal(X21[i + i * ldx21]));
            c = cos(phi[i]);
            s = sin(phi[i]);
            {
                c128 conjtaup1 = conj(taup1[i]);
                zlarf1f("L", p - i - 1, q - i - 1,
                        &X11[(i + 1) + i * ldx11], 1, conjtaup1,
                        &X11[(i + 1) + (i + 1) * ldx11], ldx11,
                        &work[ilarf - 1]);
            }
        }
        {
            c128 conjtaup2 = conj(taup2[i]);
            zlarf1f("L", m - p - i, q - i - 1,
                    &X21[i + i * ldx21], 1, conjtaup2,
                    &X21[i + (i + 1) * ldx21], ldx21,
                    &work[ilarf - 1]);
        }

    }

    /* Reduce the bottom-right portion of X21 to the identity matrix */

    for (i = p; i < q; i++) {
        zlarfgp(m - p - i, &X21[i + i * ldx21],
                &X21[(i + 1) + i * ldx21], 1, &taup2[i]);
        {
            c128 conjtaup2 = conj(taup2[i]);
            zlarf1f("L", m - p - i, q - i - 1,
                    &X21[i + i * ldx21], 1, conjtaup2,
                    &X21[i + (i + 1) * ldx21], ldx21,
                    &work[ilarf - 1]);
        }
    }
}
