/**
 * @file cunbdb3.c
 * @brief CUNBDB3 simultaneously bidiagonalizes the blocks of a tall and skinny
 *        matrix X with orthonormal columns.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CUNBDB3 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. M-P must be no larger than P,
 * Q, or M-Q. Routines CUNBDB1, CUNBDB2, and CUNBDB4 handle cases in
 * which M-P is not the minimum dimension.
 *
 * The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
 * and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
 * Householder vectors.
 *
 * B11 and B12 are (M-P)-by-(M-P) bidiagonal matrices represented
 * implicitly by angles THETA, PHI.
 *
 * @param[in]     m       The number of rows X11 plus the number of rows in X21.
 * @param[in]     p       The number of rows in X11. 0 <= P <= M. M-P <= min(P,Q,M-Q).
 * @param[in]     q       The number of columns in X11 and X21. 0 <= Q <= M.
 * @param[in,out] X11     Complex array, dimension (ldx11, q).
 *                        On entry, the top block of the matrix X to be reduced. On
 *                        exit, the columns of tril(X11) specify reflectors for P1 and
 *                        the rows of triu(X11,1) specify reflectors for Q1.
 * @param[in]     ldx11   The leading dimension of X11. ldx11 >= P.
 * @param[in,out] X21     Complex array, dimension (ldx21, q).
 *                        On entry, the bottom block of the matrix X to be reduced. On
 *                        exit, the columns of tril(X21) specify reflectors for P2.
 * @param[in]     ldx21   The leading dimension of X21. ldx21 >= M-P.
 * @param[out]    theta   Single precision array, dimension (q).
 *                        The entries of the bidiagonal blocks B11, B21 are defined by
 *                        THETA and PHI. See Further Details.
 * @param[out]    phi     Single precision array, dimension (q-1).
 *                        The entries of the bidiagonal blocks B11, B21 are defined by
 *                        THETA and PHI. See Further Details.
 * @param[out]    taup1   Complex array, dimension (p).
 *                        The scalar factors of the elementary reflectors that define P1.
 * @param[out]    taup2   Complex array, dimension (m-p).
 *                        The scalar factors of the elementary reflectors that define P2.
 * @param[out]    tauq1   Complex array, dimension (q).
 *                        The scalar factors of the elementary reflectors that define Q1.
 * @param[out]    work    Complex array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array WORK. lwork >= M-Q.
 *                        If lwork = -1, then a workspace query is assumed; the routine
 *                        only calculates the optimal size of the WORK array, returns
 *                        this value as the first entry of the WORK array, and no error
 *                        message related to lwork is issued by XERBLA.
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void cunbdb3(const int m, const int p, const int q,
             c64* restrict X11, const int ldx11,
             c64* restrict X21, const int ldx21,
             f32* restrict theta, f32* restrict phi,
             c64* restrict taup1,
             c64* restrict taup2,
             c64* restrict tauq1,
             c64* restrict work, const int lwork,
             int* info)
{
    f32 c, s;
    int childinfo, i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt;
    int lquery;

    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (2 * p < m || p > m) {
        *info = -2;
    } else if (q < m - p || m - q < m - p) {
        *info = -3;
    } else if (ldx11 < (1 > p ? 1 : p)) {
        *info = -5;
    } else if (ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -7;
    }

    if (*info == 0) {
        ilarf = 1;
        llarf = p;
        if (m - p - 1 > llarf) llarf = m - p - 1;
        if (q - 1 > llarf) llarf = q - 1;
        iorbdb5 = 1;
        lorbdb5 = q - 1;
        lworkopt = ilarf + llarf;
        if (iorbdb5 + lorbdb5 > lworkopt) lworkopt = iorbdb5 + lorbdb5;
        lworkmin = lworkopt;
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("CUNBDB3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Reduce rows 1, ..., M-P of X11 and X21 */

    for (i = 0; i < m - p; i++) {

        if (i > 0) {
            cblas_csrot(q - i, &X11[(i - 1) + i * ldx11], ldx11,
                        &X21[i + i * ldx21], ldx11, c, s);
        }

        clarfgp(q - i, &X21[i + i * ldx21], &X21[i + (i + 1) * ldx21],
                ldx21, &tauq1[i]);
        s = crealf(X21[i + i * ldx21]);
        clarf1f("R", p - i, q - i, &X21[i + i * ldx21], ldx21, tauq1[i],
                &X11[i + i * ldx11], ldx11, &work[ilarf]);
        clarf1f("R", m - p - i - 1, q - i, &X21[i + i * ldx21], ldx21,
                tauq1[i], &X21[(i + 1) + i * ldx21], ldx21, &work[ilarf]);
        clacgv(q - i, &X21[i + i * ldx21], ldx21);
        {
            f32 nrm1 = cblas_scnrm2(p - i, &X11[i + i * ldx11], 1);
            f32 nrm2 = cblas_scnrm2(m - p - i - 1, &X21[(i + 1) + i * ldx21], 1);
            c = sqrtf(nrm1 * nrm1 + nrm2 * nrm2);
        }
        theta[i] = atan2f(s, c);

        cunbdb5(p - i, m - p - i - 1, q - i - 1,
                &X11[i + i * ldx11], 1, &X21[(i + 1) + i * ldx21], 1,
                &X11[i + (i + 1) * ldx11], ldx11,
                &X21[(i + 1) + (i + 1) * ldx21], ldx21,
                &work[iorbdb5], lorbdb5, &childinfo);
        clarfgp(p - i, &X11[i + i * ldx11],
                &X11[((i + 1) < p ? (i + 1) : i) + i * ldx11], 1, &taup1[i]);
        if (i < m - p - 1) {
            clarfgp(m - p - i - 1, &X21[(i + 1) + i * ldx21],
                    &X21[((i + 2) < (m - p) ? (i + 2) : (i + 1)) + i * ldx21],
                    1, &taup2[i]);
            phi[i] = atan2f(crealf(X21[(i + 1) + i * ldx21]),
                           crealf(X11[i + i * ldx11]));
            c = cosf(phi[i]);
            s = sinf(phi[i]);
            {
                c64 conjtaup2 = conjf(taup2[i]);
                clarf1f("L", m - p - i - 1, q - i - 1,
                        &X21[(i + 1) + i * ldx21], 1, conjtaup2,
                        &X21[(i + 1) + (i + 1) * ldx21], ldx21, &work[ilarf]);
            }
        }
        {
            c64 conjtaup1 = conjf(taup1[i]);
            clarf1f("L", p - i, q - i - 1,
                    &X11[i + i * ldx11], 1, conjtaup1,
                    &X11[i + (i + 1) * ldx11], ldx11, &work[ilarf]);
        }
    }

    /* Reduce the bottom-right portion of X11 to the identity matrix */

    for (i = m - p; i < q; i++) {
        clarfgp(p - i, &X11[i + i * ldx11],
                &X11[((i + 1) < p ? (i + 1) : i) + i * ldx11], 1, &taup1[i]);
        {
            c64 conjtaup1 = conjf(taup1[i]);
            clarf1f("L", p - i, q - i - 1,
                    &X11[i + i * ldx11], 1, conjtaup1,
                    &X11[i + (i + 1) * ldx11], ldx11, &work[ilarf]);
        }
    }
}
