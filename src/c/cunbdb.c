/**
 * @file cunbdb.c
 * @brief CUNBDB simultaneously bidiagonalizes the blocks of an M-by-M
 *        partitioned unitary matrix X.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * CUNBDB simultaneously bidiagonalizes the blocks of an M-by-M
 * partitioned unitary matrix X:
 *
 *                                 [ B11 | B12 0  0 ]
 *     [ X11 | X12 ]   [ P1 |    ] [  0  |  0 -I  0 ] [ Q1 |    ]**H
 * X = [-----------] = [---------] [----------------] [---------]   .
 *     [ X21 | X22 ]   [    | P2 ] [ B21 | B22 0  0 ] [    | Q2 ]
 *                                 [  0  |  0  0  I ]
 *
 * X11 is P-by-Q. Q must be no larger than P, M-P, or M-Q. (If this is
 * not the case, then X must be transposed and/or permuted. This can be
 * done in constant time using the TRANS and SIGNS options. See CUNCSD
 * for details.)
 *
 * The unitary matrices P1, P2, Q1, and Q2 are P-by-P, (M-P)-by-
 * (M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. They are
 * represented implicitly by Householder vectors.
 *
 * B11, B12, B21, and B22 are Q-by-Q bidiagonal matrices represented
 * implicitly by angles THETA, PHI.
 *
 * @param[in]     trans   = 'T': X, U1, U2, V1T, and V2T are stored in row-major
 *                          order;
 *                        otherwise: X, U1, U2, V1T, and V2T are stored in column-
 *                          major order.
 * @param[in]     signs   = 'O': The lower-left block is made nonpositive (the
 *                          "other" convention);
 *                        otherwise: The upper-right block is made nonpositive (the
 *                          "default" convention).
 * @param[in]     m       The number of rows and columns in X.
 * @param[in]     p       The number of rows in X11 and X12. 0 <= P <= M.
 * @param[in]     q       The number of columns in X11 and X21. 0 <= Q <=
 *                        MIN(P,M-P,M-Q).
 * @param[in,out] X11     Complex*16 array, dimension (LDX11,Q).
 *                        On entry, the top-left block of the unitary matrix to be
 *                        reduced. On exit, the form depends on TRANS:
 *                        If TRANS = 'N', then
 *                           the columns of tril(X11) specify reflectors for P1,
 *                           the rows of triu(X11,1) specify reflectors for Q1;
 *                        else TRANS = 'T', and
 *                           the rows of triu(X11) specify reflectors for P1,
 *                           the columns of tril(X11,-1) specify reflectors for Q1.
 * @param[in]     ldx11   The leading dimension of X11. If TRANS = 'N', then
 *                        LDX11 >= P; else LDX11 >= Q.
 * @param[in,out] X12     Complex*16 array, dimension (LDX12,M-Q).
 *                        On entry, the top-right block of the unitary matrix to
 *                        be reduced. On exit, the form depends on TRANS:
 *                        If TRANS = 'N', then
 *                           the rows of triu(X12) specify the first P reflectors for
 *                           Q2;
 *                        else TRANS = 'T', and
 *                           the columns of tril(X12) specify the first P reflectors
 *                           for Q2.
 * @param[in]     ldx12   The leading dimension of X12. If TRANS = 'N', then
 *                        LDX12 >= P; else LDX12 >= M-Q.
 * @param[in,out] X21     Complex*16 array, dimension (LDX21,Q).
 *                        On entry, the bottom-left block of the unitary matrix to
 *                        be reduced. On exit, the form depends on TRANS:
 *                        If TRANS = 'N', then
 *                           the columns of tril(X21) specify reflectors for P2;
 *                        else TRANS = 'T', and
 *                           the rows of triu(X21) specify reflectors for P2.
 * @param[in]     ldx21   The leading dimension of X21. If TRANS = 'N', then
 *                        LDX21 >= M-P; else LDX21 >= Q.
 * @param[in,out] X22     Complex*16 array, dimension (LDX22,M-Q).
 *                        On entry, the bottom-right block of the unitary matrix to
 *                        be reduced. On exit, the form depends on TRANS:
 *                        If TRANS = 'N', then
 *                           the rows of triu(X22(Q+1:M-P,P+1:M-Q)) specify the last
 *                           M-P-Q reflectors for Q2,
 *                        else TRANS = 'T', and
 *                           the columns of tril(X22(P+1:M-Q,Q+1:M-P)) specify the last
 *                           M-P-Q reflectors for P2.
 * @param[in]     ldx22   The leading dimension of X22. If TRANS = 'N', then
 *                        LDX22 >= M-P; else LDX22 >= M-Q.
 * @param[out]    theta   Single precision array, dimension (Q).
 *                        The entries of the bidiagonal blocks B11, B12, B21, B22 can
 *                        be computed from the angles THETA and PHI. See Further
 *                        Details.
 * @param[out]    phi     Single precision array, dimension (Q-1).
 *                        The entries of the bidiagonal blocks B11, B12, B21, B22 can
 *                        be computed from the angles THETA and PHI. See Further
 *                        Details.
 * @param[out]    taup1   Complex*16 array, dimension (P).
 *                        The scalar factors of the elementary reflectors that define
 *                        P1.
 * @param[out]    taup2   Complex*16 array, dimension (M-P).
 *                        The scalar factors of the elementary reflectors that define
 *                        P2.
 * @param[out]    tauq1   Complex*16 array, dimension (Q).
 *                        The scalar factors of the elementary reflectors that define
 *                        Q1.
 * @param[out]    tauq2   Complex*16 array, dimension (M-Q).
 *                        The scalar factors of the elementary reflectors that define
 *                        Q2.
 * @param[out]    work    Complex*16 array, dimension (LWORK).
 * @param[in]     lwork   The dimension of the array WORK. LWORK >= M-Q.
 *                        If LWORK = -1, then a workspace query is assumed; the routine
 *                        only calculates the optimal size of the WORK array, returns
 *                        this value as the first entry of the WORK array, and no error
 *                        message related to LWORK is issued by XERBLA.
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if INFO = -i, the i-th argument had an illegal value.
 */
void cunbdb(const char* trans, const char* signs,
            const INT m, const INT p, const INT q,
            c64* restrict X11, const INT ldx11,
            c64* restrict X12, const INT ldx12,
            c64* restrict X21, const INT ldx21,
            c64* restrict X22, const INT ldx22,
            f32* restrict theta, f32* restrict phi,
            c64* restrict taup1,
            c64* restrict taup2,
            c64* restrict tauq1,
            c64* restrict tauq2,
            c64* restrict work, const INT lwork,
            INT* info)
{
    INT colmajor, lquery;
    INT i, lworkmin, lworkopt;
    f32 z1, z2, z3, z4;

    *info = 0;
    colmajor = !(trans[0] == 'T' || trans[0] == 't');
    if (!(signs[0] == 'O' || signs[0] == 'o')) {
        z1 = 1.0f;
        z2 = 1.0f;
        z3 = 1.0f;
        z4 = 1.0f;
    } else {
        z1 = 1.0f;
        z2 = -1.0f;
        z3 = 1.0f;
        z4 = -1.0f;
    }
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -3;
    } else if (p < 0 || p > m) {
        *info = -4;
    } else if (q < 0 || q > p || q > m - p || q > m - q) {
        *info = -5;
    } else if (colmajor && ldx11 < (1 > p ? 1 : p)) {
        *info = -7;
    } else if (!colmajor && ldx11 < (1 > q ? 1 : q)) {
        *info = -7;
    } else if (colmajor && ldx12 < (1 > p ? 1 : p)) {
        *info = -9;
    } else if (!colmajor && ldx12 < (1 > (m - q) ? 1 : (m - q))) {
        *info = -9;
    } else if (colmajor && ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -11;
    } else if (!colmajor && ldx21 < (1 > q ? 1 : q)) {
        *info = -11;
    } else if (colmajor && ldx22 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -13;
    } else if (!colmajor && ldx22 < (1 > (m - q) ? 1 : (m - q))) {
        *info = -13;
    }

    if (*info == 0) {
        lworkopt = m - q;
        lworkmin = m - q;
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        if (lwork < lworkmin && !lquery) {
            *info = -21;
        }
    }
    if (*info != 0) {
        xerbla("CUNBDB", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (colmajor) {

        /* Reduce columns 1, ..., Q of X11, X12, X21, and X22 */

        for (i = 0; i < q; i++) {

            if (i == 0) {
                c64 sc = CMPLXF(z1, 0.0f);
                cblas_cscal(p - i, &sc, &X11[i + i * ldx11], 1);
            } else {
                c64 sc = CMPLXF(z1 * cosf(phi[i - 1]), 0.0f);
                cblas_cscal(p - i, &sc, &X11[i + i * ldx11], 1);
                c64 alpha = CMPLXF(-z1 * z3 * z4 * sinf(phi[i - 1]), 0.0f);
                cblas_caxpy(p - i, &alpha, &X12[i + (i - 1) * ldx12], 1,
                            &X11[i + i * ldx11], 1);
            }
            if (i == 0) {
                c64 sc = CMPLXF(z2, 0.0f);
                cblas_cscal(m - p - i, &sc, &X21[i + i * ldx21], 1);
            } else {
                c64 sc = CMPLXF(z2 * cosf(phi[i - 1]), 0.0f);
                cblas_cscal(m - p - i, &sc, &X21[i + i * ldx21], 1);
                c64 alpha = CMPLXF(-z2 * z3 * z4 * sinf(phi[i - 1]), 0.0f);
                cblas_caxpy(m - p - i, &alpha, &X22[i + (i - 1) * ldx22], 1,
                            &X21[i + i * ldx21], 1);
            }

            theta[i] = atan2f(cblas_scnrm2(m - p - i, &X21[i + i * ldx21], 1),
                             cblas_scnrm2(p - i, &X11[i + i * ldx11], 1));

            if (p > i + 1) {
                clarfgp(p - i, &X11[i + i * ldx11],
                        &X11[(i + 1) + i * ldx11], 1, &taup1[i]);
            } else if (p == i + 1) {
                clarfgp(1, &X11[i + i * ldx11],
                        NULL, 1, &taup1[i]);
            }
            if (m - p > i + 1) {
                clarfgp(m - p - i, &X21[i + i * ldx21],
                        &X21[(i + 1) + i * ldx21], 1, &taup2[i]);
            } else if (m - p == i + 1) {
                clarfgp(1, &X21[i + i * ldx21],
                        NULL, 1, &taup2[i]);
            }

            if (q > i + 1) {
                c64 ct1 = conjf(taup1[i]);
                clarf1f("L", p - i, q - i - 1, &X11[i + i * ldx11], 1,
                        ct1, &X11[i + (i + 1) * ldx11], ldx11, work);
                c64 ct2 = conjf(taup2[i]);
                clarf1f("L", m - p - i, q - i - 1, &X21[i + i * ldx21], 1,
                        ct2, &X21[i + (i + 1) * ldx21], ldx21, work);
            }
            if (m - q >= i + 1) {
                c64 ct1 = conjf(taup1[i]);
                clarf1f("L", p - i, m - q - i, &X11[i + i * ldx11], 1,
                        ct1, &X12[i + i * ldx12], ldx12, work);
                c64 ct2 = conjf(taup2[i]);
                clarf1f("L", m - p - i, m - q - i, &X21[i + i * ldx21], 1,
                        ct2, &X22[i + i * ldx22], ldx22, work);
            }

            if (i < q - 1) {
                c64 sc = CMPLXF(-z1 * z3 * sinf(theta[i]), 0.0f);
                cblas_cscal(q - i - 1, &sc, &X11[i + (i + 1) * ldx11], ldx11);
                c64 alpha = CMPLXF(z2 * z3 * cosf(theta[i]), 0.0f);
                cblas_caxpy(q - i - 1, &alpha, &X21[i + (i + 1) * ldx21], ldx21,
                            &X11[i + (i + 1) * ldx11], ldx11);
            }
            {
                c64 sc = CMPLXF(-z1 * z4 * sinf(theta[i]), 0.0f);
                cblas_cscal(m - q - i, &sc, &X12[i + i * ldx12], ldx12);
                c64 alpha = CMPLXF(z2 * z4 * cosf(theta[i]), 0.0f);
                cblas_caxpy(m - q - i, &alpha, &X22[i + i * ldx22], ldx22,
                            &X12[i + i * ldx12], ldx12);
            }

            if (i < q - 1)
                phi[i] = atan2f(cblas_scnrm2(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11),
                               cblas_scnrm2(m - q - i, &X12[i + i * ldx12], ldx12));

            if (i < q - 1) {
                clacgv(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11);
                if (i == q - 2) {
                    clarfgp(1, &X11[i + (i + 1) * ldx11],
                            NULL, ldx11, &tauq1[i]);
                } else {
                    clarfgp(q - i - 1, &X11[i + (i + 1) * ldx11],
                            &X11[i + (i + 2) * ldx11], ldx11, &tauq1[i]);
                }
            }
            if (m - q >= i + 1) {
                clacgv(m - q - i, &X12[i + i * ldx12], ldx12);
                if (m - q == i + 1) {
                    clarfgp(1, &X12[i + i * ldx12],
                            NULL, ldx12, &tauq2[i]);
                } else {
                    clarfgp(m - q - i, &X12[i + i * ldx12],
                            &X12[i + (i + 1) * ldx12], ldx12, &tauq2[i]);
                }
            }

            if (i < q - 1) {
                clarf1f("R", p - i - 1, q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                        tauq1[i], &X11[(i + 1) + (i + 1) * ldx11], ldx11, work);
                clarf1f("R", m - p - i - 1, q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                        tauq1[i], &X21[(i + 1) + (i + 1) * ldx21], ldx21, work);
            }
            if (p > i + 1) {
                clarf1f("R", p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X12[(i + 1) + i * ldx12], ldx12, work);
            }
            if (m - p > i + 1) {
                clarf1f("R", m - p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X22[(i + 1) + i * ldx22], ldx22, work);
            }

            if (i < q - 1)
                clacgv(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11);
            clacgv(m - q - i, &X12[i + i * ldx12], ldx12);
        }

        /* Reduce columns Q + 1, ..., P of X12, X22 */

        for (i = q; i < p; i++) {

            {
                c64 sc = CMPLXF(-z1 * z4, 0.0f);
                cblas_cscal(m - q - i, &sc, &X12[i + i * ldx12], ldx12);
            }
            clacgv(m - q - i, &X12[i + i * ldx12], ldx12);
            if (i >= m - q - 1) {
                clarfgp(1, &X12[i + i * ldx12],
                        NULL, ldx12, &tauq2[i]);
            } else {
                clarfgp(m - q - i, &X12[i + i * ldx12],
                        &X12[i + (i + 1) * ldx12], ldx12, &tauq2[i]);
            }

            if (p > i + 1) {
                clarf1f("R", p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X12[(i + 1) + i * ldx12], ldx12, work);
            }
            if (m - p - q >= 1)
                clarf1f("R", m - p - q, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X22[q + i * ldx22], ldx22, work);

            clacgv(m - q - i, &X12[i + i * ldx12], ldx12);
        }

        /* Reduce columns P + 1, ..., M - Q of X12, X22 */

        for (i = 0; i < m - p - q; i++) {

            {
                c64 sc = CMPLXF(z2 * z4, 0.0f);
                cblas_cscal(m - p - q - i, &sc, &X22[(q + i) + (p + i) * ldx22], ldx22);
            }
            clacgv(m - p - q - i, &X22[(q + i) + (p + i) * ldx22], ldx22);
            clarfgp(m - p - q - i, &X22[(q + i) + (p + i) * ldx22],
                    &X22[(q + i) + (p + i + 1) * ldx22], ldx22, &tauq2[p + i]);
            clarf1f("R", m - p - q - i - 1, m - p - q - i,
                    &X22[(q + i) + (p + i) * ldx22], ldx22,
                    tauq2[p + i], &X22[(q + i + 1) + (p + i) * ldx22], ldx22, work);

            clacgv(m - p - q - i, &X22[(q + i) + (p + i) * ldx22], ldx22);
        }

    } else {

        /* Reduce columns 1, ..., Q of X11, X12, X21, and X22 */

        for (i = 0; i < q; i++) {

            if (i == 0) {
                c64 sc = CMPLXF(z1, 0.0f);
                cblas_cscal(p - i, &sc, &X11[i + i * ldx11], ldx11);
            } else {
                c64 sc = CMPLXF(z1 * cosf(phi[i - 1]), 0.0f);
                cblas_cscal(p - i, &sc, &X11[i + i * ldx11], ldx11);
                c64 alpha = CMPLXF(-z1 * z3 * z4 * sinf(phi[i - 1]), 0.0f);
                cblas_caxpy(p - i, &alpha, &X12[(i - 1) + i * ldx12], ldx12,
                            &X11[i + i * ldx11], ldx11);
            }
            if (i == 0) {
                c64 sc = CMPLXF(z2, 0.0f);
                cblas_cscal(m - p - i, &sc, &X21[i + i * ldx21], ldx21);
            } else {
                c64 sc = CMPLXF(z2 * cosf(phi[i - 1]), 0.0f);
                cblas_cscal(m - p - i, &sc, &X21[i + i * ldx21], ldx21);
                c64 alpha = CMPLXF(-z2 * z3 * z4 * sinf(phi[i - 1]), 0.0f);
                cblas_caxpy(m - p - i, &alpha, &X22[(i - 1) + i * ldx22], ldx22,
                            &X21[i + i * ldx21], ldx21);
            }

            theta[i] = atan2f(cblas_scnrm2(m - p - i, &X21[i + i * ldx21], ldx21),
                             cblas_scnrm2(p - i, &X11[i + i * ldx11], ldx11));

            clacgv(p - i, &X11[i + i * ldx11], ldx11);
            clacgv(m - p - i, &X21[i + i * ldx21], ldx21);

            clarfgp(p - i, &X11[i + i * ldx11],
                    &X11[i + (i + 1) * ldx11], ldx11, &taup1[i]);
            if (m - p == i + 1) {
                clarfgp(1, &X21[i + i * ldx21],
                        NULL, ldx21, &taup2[i]);
            } else {
                clarfgp(m - p - i, &X21[i + i * ldx21],
                        &X21[i + (i + 1) * ldx21], ldx21, &taup2[i]);
            }

            clarf1f("R", q - i - 1, p - i, &X11[i + i * ldx11], ldx11,
                    taup1[i], &X11[(i + 1) + i * ldx11], ldx11, work);
            clarf1f("R", m - q - i, p - i, &X11[i + i * ldx11], ldx11,
                    taup1[i], &X12[i + i * ldx12], ldx12, work);
            clarf1f("R", q - i - 1, m - p - i, &X21[i + i * ldx21], ldx21,
                    taup2[i], &X21[(i + 1) + i * ldx21], ldx21, work);
            clarf1f("R", m - q - i, m - p - i, &X21[i + i * ldx21], ldx21,
                    taup2[i], &X22[i + i * ldx22], ldx22, work);

            clacgv(p - i, &X11[i + i * ldx11], ldx11);
            clacgv(m - p - i, &X21[i + i * ldx21], ldx21);

            if (i < q - 1) {
                c64 sc = CMPLXF(-z1 * z3 * sinf(theta[i]), 0.0f);
                cblas_cscal(q - i - 1, &sc, &X11[(i + 1) + i * ldx11], 1);
                c64 alpha = CMPLXF(z2 * z3 * cosf(theta[i]), 0.0f);
                cblas_caxpy(q - i - 1, &alpha, &X21[(i + 1) + i * ldx21], 1,
                            &X11[(i + 1) + i * ldx11], 1);
            }
            {
                c64 sc = CMPLXF(-z1 * z4 * sinf(theta[i]), 0.0f);
                cblas_cscal(m - q - i, &sc, &X12[i + i * ldx12], 1);
                c64 alpha = CMPLXF(z2 * z4 * cosf(theta[i]), 0.0f);
                cblas_caxpy(m - q - i, &alpha, &X22[i + i * ldx22], 1,
                            &X12[i + i * ldx12], 1);
            }

            if (i < q - 1)
                phi[i] = atan2f(cblas_scnrm2(q - i - 1, &X11[(i + 1) + i * ldx11], 1),
                               cblas_scnrm2(m - q - i, &X12[i + i * ldx12], 1));

            if (i < q - 1) {
                clarfgp(q - i - 1, &X11[(i + 1) + i * ldx11],
                        &X11[(i + 2) + i * ldx11], 1, &tauq1[i]);
            }
            clarfgp(m - q - i, &X12[i + i * ldx12],
                    &X12[(i + 1) + i * ldx12], 1, &tauq2[i]);

            if (i < q - 1) {
                c64 ct1 = conjf(tauq1[i]);
                clarf1f("L", q - i - 1, p - i - 1, &X11[(i + 1) + i * ldx11], 1,
                        ct1, &X11[(i + 1) + (i + 1) * ldx11], ldx11, work);
                clarf1f("L", q - i - 1, m - p - i - 1, &X11[(i + 1) + i * ldx11], 1,
                        ct1, &X21[(i + 1) + (i + 1) * ldx21], ldx21, work);
            }
            {
                c64 ct2 = conjf(tauq2[i]);
                clarf1f("L", m - q - i, p - i - 1, &X12[i + i * ldx12], 1,
                        ct2, &X12[i + (i + 1) * ldx12], ldx12, work);
            }

            if (m - p > i + 1) {
                c64 ct2 = conjf(tauq2[i]);
                clarf1f("L", m - q - i, m - p - i - 1, &X12[i + i * ldx12], 1,
                        ct2, &X22[i + (i + 1) * ldx22], ldx22, work);
            }
        }

        /* Reduce columns Q + 1, ..., P of X12, X22 */

        for (i = q; i < p; i++) {

            {
                c64 sc = CMPLXF(-z1 * z4, 0.0f);
                cblas_cscal(m - q - i, &sc, &X12[i + i * ldx12], 1);
            }
            clarfgp(m - q - i, &X12[i + i * ldx12],
                    &X12[(i + 1) + i * ldx12], 1, &tauq2[i]);

            if (p > i + 1) {
                c64 ct = conjf(tauq2[i]);
                clarf1f("L", m - q - i, p - i - 1, &X12[i + i * ldx12], 1,
                        ct, &X12[i + (i + 1) * ldx12], ldx12, work);
            }
            if (m - p - q >= 1) {
                c64 ct = conjf(tauq2[i]);
                clarf1f("L", m - q - i, m - p - q, &X12[i + i * ldx12], 1,
                        ct, &X22[i + q * ldx22], ldx22, work);
            }
        }

        /* Reduce columns P + 1, ..., M - Q of X12, X22 */

        for (i = 0; i < m - p - q; i++) {

            {
                c64 sc = CMPLXF(z2 * z4, 0.0f);
                cblas_cscal(m - p - q - i, &sc, &X22[(p + i) + (q + i) * ldx22], 1);
            }
            clarfgp(m - p - q - i, &X22[(p + i) + (q + i) * ldx22],
                    &X22[(p + i + 1) + (q + i) * ldx22], 1, &tauq2[p + i]);
            if (m - p - q != i + 1) {
                c64 ct = conjf(tauq2[p + i]);
                clarf1f("L", m - p - q - i, m - p - q - i - 1,
                        &X22[(p + i) + (q + i) * ldx22], 1,
                        ct, &X22[(p + i) + (q + i + 1) * ldx22], ldx22, work);
            }
        }
    }
}
