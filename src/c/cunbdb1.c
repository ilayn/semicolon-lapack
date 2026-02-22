/**
 * @file cunbdb1.c
 * @brief CUNBDB1 simultaneously bidiagonalizes the blocks of a tall and
 *        skinny matrix X with orthonormal columns.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>

/**
 * CUNBDB1 simultaneously bidiagonalizes the blocks of a tall and skinny
 * matrix X with orthonormal columns:
 *
 *                            [ B11 ]
 *      [ X11 ]   [ P1 |    ] [  0  ]
 *      [-----] = [---------] [-----] Q1**T .
 *      [ X21 ]   [    | P2 ] [ B21 ]
 *                            [  0  ]
 *
 * X11 is P-by-Q, and X21 is (M-P)-by-Q. Q must be no larger than P,
 * M-P, or M-Q. Routines CUNBDB2, CUNBDB3, and CUNBDB4 handle cases in
 * which Q is not the minimum dimension.
 *
 * The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
 * and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
 * Householder vectors.
 *
 * B11 and B12 are Q-by-Q bidiagonal matrices represented implicitly by
 * angles THETA, PHI.
 *
 * @param[in]     m       The number of rows X11 plus the number of rows in X21.
 * @param[in]     p       The number of rows in X11. 0 <= P <= M.
 * @param[in]     q       The number of columns in X11 and X21. 0 <= Q <=
 *                        MIN(P,M-P,M-Q).
 * @param[in,out] X11     Complex*16 array, dimension (LDX11,Q).
 *                        On entry, the top block of the matrix X to be reduced.
 *                        On exit, the columns of tril(X11) specify reflectors
 *                        for P1 and the rows of triu(X11,1) specify reflectors
 *                        for Q1.
 * @param[in]     ldx11   The leading dimension of X11. LDX11 >= P.
 * @param[in,out] X21     Complex*16 array, dimension (LDX21,Q).
 *                        On entry, the bottom block of the matrix X to be
 *                        reduced. On exit, the columns of tril(X21) specify
 *                        reflectors for P2.
 * @param[in]     ldx21   The leading dimension of X21. LDX21 >= M-P.
 * @param[out]    theta   Single precision array, dimension (Q).
 *                        The entries of the bidiagonal blocks B11, B21 are
 *                        defined by THETA and PHI. See Further Details.
 * @param[out]    phi     Single precision array, dimension (Q-1).
 *                        The entries of the bidiagonal blocks B11, B21 are
 *                        defined by THETA and PHI. See Further Details.
 * @param[out]    taup1   Complex*16 array, dimension (P).
 *                        The scalar factors of the elementary reflectors that
 *                        define P1.
 * @param[out]    taup2   Complex*16 array, dimension (M-P).
 *                        The scalar factors of the elementary reflectors that
 *                        define P2.
 * @param[out]    tauq1   Complex*16 array, dimension (Q).
 *                        The scalar factors of the elementary reflectors that
 *                        define Q1.
 * @param[out]    work    Complex*16 array, dimension (LWORK).
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
void cunbdb1(const INT m, const INT p, const INT q,
             c64* restrict X11, const INT ldx11,
             c64* restrict X21, const INT ldx21,
             f32* restrict theta, f32* restrict phi,
             c64* restrict taup1,
             c64* restrict taup2,
             c64* restrict tauq1,
             c64* restrict work, const INT lwork,
             INT* info)
{
    f32 c, s;
    INT childinfo, i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt;
    INT lquery;

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
        llarf = p - 1;
        if (m - p - 1 > llarf) llarf = m - p - 1;
        if (q - 1 > llarf) llarf = q - 1;
        iorbdb5 = 1;
        lorbdb5 = q - 2;
        lworkopt = ilarf + llarf;
        if (iorbdb5 + lorbdb5 > lworkopt) lworkopt = iorbdb5 + lorbdb5;
        lworkmin = lworkopt;
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        if (lwork < lworkmin && !lquery) {
            *info = -14;
        }
    }
    if (*info != 0) {
        xerbla("CUNBDB1", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    for (i = 0; i < q; i++) {

        clarfgp(p - i, &X11[i + i * ldx11],
                &X11[((i + 1) < p ? (i + 1) : i) + i * ldx11], 1,
                &taup1[i]);
        clarfgp(m - p - i, &X21[i + i * ldx21],
                &X21[((i + 1) < (m - p) ? (i + 1) : i) + i * ldx21], 1,
                &taup2[i]);
        theta[i] = atan2f(crealf(X21[i + i * ldx21]),
                         crealf(X11[i + i * ldx11]));
        c = cosf(theta[i]);
        s = sinf(theta[i]);
        {
            c64 conjtaup1 = conjf(taup1[i]);
            clarf1f("L", p - i, q - i - 1, &X11[i + i * ldx11], 1,
                    conjtaup1, &X11[i + (i + 1) * ldx11], ldx11,
                    &work[ilarf]);
        }
        {
            c64 conjtaup2 = conjf(taup2[i]);
            clarf1f("L", m - p - i, q - i - 1, &X21[i + i * ldx21], 1,
                    conjtaup2, &X21[i + (i + 1) * ldx21], ldx21,
                    &work[ilarf]);
        }

        if (i < q - 1) {
            cblas_csrot(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                        &X21[i + (i + 1) * ldx21], ldx21, c, s);
            clacgv(q - i - 1, &X21[i + (i + 1) * ldx21], ldx21);
            clarfgp(q - i - 1, &X21[i + (i + 1) * ldx21],
                    &X21[i + ((i + 2) < q ? (i + 2) : (i + 1)) * ldx21],
                    ldx21, &tauq1[i]);
            s = crealf(X21[i + (i + 1) * ldx21]);
            clarf1f("R", p - i - 1, q - i - 1,
                    &X21[i + (i + 1) * ldx21], ldx21, tauq1[i],
                    &X11[(i + 1) + (i + 1) * ldx11], ldx11, &work[ilarf]);
            clarf1f("R", m - p - i - 1, q - i - 1,
                    &X21[i + (i + 1) * ldx21], ldx21, tauq1[i],
                    &X21[(i + 1) + (i + 1) * ldx21], ldx21, &work[ilarf]);
            clacgv(q - i - 1, &X21[i + (i + 1) * ldx21], ldx21);
            c = sqrtf(powf(cblas_scnrm2(p - i - 1,
                        &X11[(i + 1) + (i + 1) * ldx11], 1), 2)
                   + powf(cblas_scnrm2(m - p - i - 1,
                        &X21[(i + 1) + (i + 1) * ldx21], 1), 2));
            phi[i] = atan2f(s, c);
            cunbdb5(p - i - 1, m - p - i - 1, q - i - 2,
                    &X11[(i + 1) + (i + 1) * ldx11], 1,
                    &X21[(i + 1) + (i + 1) * ldx21], 1,
                    &X11[(i + 1) + (i + 2) * ldx11], ldx11,
                    &X21[(i + 1) + (i + 2) * ldx21], ldx21,
                    &work[iorbdb5], lorbdb5, &childinfo);
        }
    }
}
