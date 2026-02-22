/**
 * @file cunbdb4.c
 * @brief CUNBDB4 simultaneously bidiagonalizes the blocks of a tall and skinny
 *        matrix X with orthonormal columns.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <cblas.h>
#include <complex.h>
#include <math.h>

/**
 * CUNBDB4 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P,
 * M-P, or Q. Routines CUNBDB1, CUNBDB2, and CUNBDB3 handle cases in
 * which M-Q is not the minimum dimension.
 *
 * The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
 * and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
 * Householder vectors.
 *
 * B11 and B12 are (M-Q)-by-(M-Q) bidiagonal matrices represented
 * implicitly by angles THETA, PHI.
 *
 * @param[in]     m        The number of rows X11 plus the number of rows in X21.
 * @param[in]     p        The number of rows in X11. 0 <= P <= M.
 * @param[in]     q        The number of columns in X11 and X21. 0 <= Q <= M and
 *                         M-Q <= min(P,M-P,Q).
 * @param[in,out] X11      Complex array, dimension (ldx11, q).
 *                         On entry, the top block of the matrix X to be reduced. On
 *                         exit, the columns of tril(X11) specify reflectors for P1 and
 *                         the rows of triu(X11,1) specify reflectors for Q1.
 * @param[in]     ldx11    The leading dimension of X11. ldx11 >= P.
 * @param[in,out] X21      Complex array, dimension (ldx21, q).
 *                         On entry, the bottom block of the matrix X to be reduced. On
 *                         exit, the columns of tril(X21) specify reflectors for P2.
 * @param[in]     ldx21    The leading dimension of X21. ldx21 >= M-P.
 * @param[out]    theta    Single precision array, dimension (q).
 *                         The entries of the bidiagonal blocks B11, B21 are defined by
 *                         THETA and PHI. See Further Details.
 * @param[out]    phi      Single precision array, dimension (q-1).
 *                         The entries of the bidiagonal blocks B11, B21 are defined by
 *                         THETA and PHI. See Further Details.
 * @param[out]    taup1    Complex array, dimension (m-q).
 *                         The scalar factors of the elementary reflectors that define P1.
 * @param[out]    taup2    Complex array, dimension (m-q).
 *                         The scalar factors of the elementary reflectors that define P2.
 * @param[out]    tauq1    Complex array, dimension (q).
 *                         The scalar factors of the elementary reflectors that define Q1.
 * @param[out]    phantom  Complex array, dimension (m).
 *                         The routine computes an M-by-1 column vector Y that is
 *                         orthogonal to the columns of [ X11; X21 ]. PHANTOM(0:P-1) and
 *                         PHANTOM(P:M-1) contain Householder vectors for Y(0:P-1) and
 *                         Y(P:M-1), respectively.
 * @param[out]    work     Complex array, dimension (lwork).
 * @param[in]     lwork    The dimension of the array WORK. lwork >= M-Q.
 *                         If lwork = -1, then a workspace query is assumed; the routine
 *                         only calculates the optimal size of the WORK array, returns
 *                         this value as the first entry of the WORK array, and no error
 *                         message related to lwork is issued by XERBLA.
 * @param[out]    info     = 0: successful exit.
 *                         < 0: if info = -i, the i-th argument had an illegal value.
 */
void cunbdb4(const INT m, const INT p, const INT q,
             c64* restrict X11, const INT ldx11,
             c64* restrict X21, const INT ldx21,
             f32* restrict theta,
             f32* restrict phi,
             c64* restrict taup1,
             c64* restrict taup2,
             c64* restrict tauq1,
             c64* restrict phantom,
             c64* restrict work, const INT lwork,
             INT* info)
{
    const c64 NEGONE = CMPLXF(-1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    f32 c, s;
    INT childinfo, i, ilarf, iorbdb5, j, llarf, lorbdb5, lworkmin, lworkopt;
    INT lquery;

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
        llarf = q - 1;
        if (p - 1 > llarf) llarf = p - 1;
        if (m - p - 1 > llarf) llarf = m - p - 1;
        iorbdb5 = 1;
        lorbdb5 = q;
        lworkopt = ilarf + llarf;
        if (iorbdb5 + lorbdb5 > lworkopt) lworkopt = iorbdb5 + lorbdb5;
        lworkmin = lworkopt;
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("CUNBDB4", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Reduce columns 1, ..., M-Q of X11 and X21 */

    for (i = 0; i < m - q; i++) {

        if (i == 0) {
            for (j = 0; j < m; j++) {
                phantom[j] = ZERO;
            }
            cunbdb5(p, m - p, q, &phantom[0], 1, &phantom[p], 1,
                    X11, ldx11, X21, ldx21, &work[iorbdb5],
                    lorbdb5, &childinfo);
            cblas_cscal(p, &NEGONE, &phantom[0], 1);
            clarfgp(p, &phantom[0], &phantom[1], 1, &taup1[0]);
            clarfgp(m - p, &phantom[p], &phantom[p + 1], 1,
                    &taup2[0]);
            theta[i] = atan2f(crealf(phantom[0]), crealf(phantom[p]));
            c = cosf(theta[i]);
            s = sinf(theta[i]);
            c64 conjtaup1 = conjf(taup1[0]);
            clarf1f("L", p, q, &phantom[0], 1, conjtaup1,
                    X11, ldx11, &work[ilarf]);
            c64 conjtaup2 = conjf(taup2[0]);
            clarf1f("L", m - p, q, &phantom[p], 1,
                    conjtaup2,
                    X21, ldx21, &work[ilarf]);
        } else {
            cunbdb5(p - i, m - p - i, q - i,
                    &X11[i + (i - 1) * ldx11], 1,
                    &X21[i + (i - 1) * ldx21], 1,
                    &X11[i + i * ldx11], ldx11,
                    &X21[i + i * ldx21], ldx21,
                    &work[iorbdb5], lorbdb5, &childinfo);
            cblas_cscal(p - i, &NEGONE, &X11[i + (i - 1) * ldx11], 1);
            clarfgp(p - i, &X11[i + (i - 1) * ldx11],
                    &X11[(i + 1) + (i - 1) * ldx11], 1,
                    &taup1[i]);
            clarfgp(m - p - i, &X21[i + (i - 1) * ldx21],
                    &X21[(i + 1) + (i - 1) * ldx21], 1,
                    &taup2[i]);
            theta[i] = atan2f(crealf(X11[i + (i - 1) * ldx11]),
                             crealf(X21[i + (i - 1) * ldx21]));
            c = cosf(theta[i]);
            s = sinf(theta[i]);
            c64 conjtaup1 = conjf(taup1[i]);
            clarf1f("L", p - i, q - i,
                    &X11[i + (i - 1) * ldx11], 1,
                    conjtaup1,
                    &X11[i + i * ldx11], ldx11,
                    &work[ilarf]);
            c64 conjtaup2 = conjf(taup2[i]);
            clarf1f("L", m - p - i, q - i,
                    &X21[i + (i - 1) * ldx21], 1,
                    conjtaup2,
                    &X21[i + i * ldx21], ldx21,
                    &work[ilarf]);
        }

        cblas_csrot(q - i, &X11[i + i * ldx11], ldx11,
                   &X21[i + i * ldx21], ldx21, s, -c);
        clacgv(q - i, &X21[i + i * ldx21], ldx21);
        clarfgp(q - i, &X21[i + i * ldx21],
                &X21[i + (i + 1) * ldx21], ldx21, &tauq1[i]);
        c = crealf(X21[i + i * ldx21]);
        clarf1f("R", p - i - 1, q - i, &X21[i + i * ldx21], ldx21,
                tauq1[i], &X11[(i + 1) + i * ldx11], ldx11,
                &work[ilarf]);
        clarf1f("R", m - p - i - 1, q - i, &X21[i + i * ldx21], ldx21,
                tauq1[i], &X21[(i + 1) + i * ldx21], ldx21,
                &work[ilarf]);
        clacgv(q - i, &X21[i + i * ldx21], ldx21);
        if (i < m - q - 1) {
            f32 nrm1 = cblas_scnrm2(p - i - 1, &X11[(i + 1) + i * ldx11], 1);
            f32 nrm2 = cblas_scnrm2(m - p - i - 1, &X21[(i + 1) + i * ldx21], 1);
            s = sqrtf(nrm1 * nrm1 + nrm2 * nrm2);
            phi[i] = atan2f(s, c);
        }

    }

    /* Reduce the bottom-right portion of X11 to [ I 0 ] */

    for (i = m - q; i < p; i++) {
        clacgv(q - i, &X11[i + i * ldx11], ldx11);
        clarfgp(q - i, &X11[i + i * ldx11],
                &X11[i + (i + 1) * ldx11], ldx11, &tauq1[i]);
        clarf1f("R", p - i - 1, q - i, &X11[i + i * ldx11], ldx11,
                tauq1[i], &X11[(i + 1) + i * ldx11], ldx11,
                &work[ilarf]);
        clarf1f("R", q - p, q - i, &X11[i + i * ldx11], ldx11,
                tauq1[i], &X21[(m - q) + i * ldx21], ldx21,
                &work[ilarf]);
        clacgv(q - i, &X11[i + i * ldx11], ldx11);
    }

    /* Reduce the bottom-right portion of X21 to [ 0 I ] */

    for (i = p; i < q; i++) {
        clacgv(q - i, &X21[(m - q + i - p) + i * ldx21], ldx21);
        clarfgp(q - i, &X21[(m - q + i - p) + i * ldx21],
                &X21[(m - q + i - p) + (i + 1) * ldx21], ldx21,
                &tauq1[i]);
        clarf1f("R", q - i - 1, q - i,
                &X21[(m - q + i - p) + i * ldx21], ldx21,
                tauq1[i],
                &X21[(m - q + i - p + 1) + i * ldx21], ldx21,
                &work[ilarf]);
        clacgv(q - i, &X21[(m - q + i - p) + i * ldx21], ldx21);
    }
}
