/**
 * @file dorbdb.c
 * @brief DORBDB simultaneously bidiagonalizes the blocks of an M-by-M partitioned orthogonal matrix.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORBDB simultaneously bidiagonalizes the blocks of an M-by-M
 * partitioned orthogonal matrix X:
 *
 *                                 [ B11 | B12 0  0 ]
 *     [ X11 | X12 ]   [ P1 |    ] [  0  |  0 -I  0 ] [ Q1 |    ]**T
 * X = [-----------] = [---------] [----------------] [---------]   .
 *     [ X21 | X22 ]   [    | P2 ] [ B21 | B22 0  0 ] [    | Q2 ]
 *                                 [  0  |  0  0  I ]
 *
 * X11 is P-by-Q. Q must be no larger than P, M-P, or M-Q.
 *
 * @param[in] trans
 *          = 'T': X, U1, U2, V1T, and V2T are stored in row-major order;
 *          otherwise: X, U1, U2, V1T, and V2T are stored in column-major order.
 *
 * @param[in] signs
 *          = 'O': The lower-left block is made nonpositive (the "other" convention);
 *          otherwise: The upper-right block is made nonpositive (the "default" convention).
 *
 * @param[in] m
 *          The number of rows and columns in X.
 *
 * @param[in] p
 *          The number of rows in X11 and X12. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in X11 and X21. 0 <= q <= min(p, m-p, m-q).
 *
 * @param[in,out] X11
 *          Double precision array, dimension (ldx11, q).
 *
 * @param[in] ldx11
 *          The leading dimension of X11.
 *
 * @param[in,out] X12
 *          Double precision array, dimension (ldx12, m-q).
 *
 * @param[in] ldx12
 *          The leading dimension of X12.
 *
 * @param[in,out] X21
 *          Double precision array, dimension (ldx21, q).
 *
 * @param[in] ldx21
 *          The leading dimension of X21.
 *
 * @param[in,out] X22
 *          Double precision array, dimension (ldx22, m-q).
 *
 * @param[in] ldx22
 *          The leading dimension of X22.
 *
 * @param[out] theta
 *          Double precision array, dimension (q).
 *
 * @param[out] phi
 *          Double precision array, dimension (q-1).
 *
 * @param[out] taup1
 *          Double precision array, dimension (p).
 *
 * @param[out] taup2
 *          Double precision array, dimension (m-p).
 *
 * @param[out] tauq1
 *          Double precision array, dimension (q).
 *
 * @param[out] tauq2
 *          Double precision array, dimension (m-q).
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
void dorbdb(
    const char* trans,
    const char* signs,
    const int m,
    const int p,
    const int q,
    double* const restrict X11,
    const int ldx11,
    double* const restrict X12,
    const int ldx12,
    double* const restrict X21,
    const int ldx21,
    double* const restrict X22,
    const int ldx22,
    double* restrict theta,
    double* restrict phi,
    double* restrict taup1,
    double* restrict taup2,
    double* restrict tauq1,
    double* restrict tauq2,
    double* restrict work,
    const int lwork,
    int* info)
{
    const double one = 1.0;
    int colmajor, lquery;
    int i, lworkmin, lworkopt;
    double z1, z2, z3, z4;

    *info = 0;
    colmajor = !(trans[0] == 'T' || trans[0] == 't');
    if (!(signs[0] == 'O' || signs[0] == 'o')) {
        z1 = one;
        z2 = one;
        z3 = one;
        z4 = one;
    } else {
        z1 = one;
        z2 = -one;
        z3 = one;
        z4 = -one;
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
        work[0] = (double)lworkopt;
        if (lwork < lworkmin && !lquery) {
            *info = -21;
        }
    }
    if (*info != 0) {
        xerbla("xORBDB", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (colmajor) {

        for (i = 0; i < q; i++) {

            if (i == 0) {
                cblas_dscal(p - i, z1, &X11[i + i * ldx11], 1);
            } else {
                cblas_dscal(p - i, z1 * cos(phi[i - 1]), &X11[i + i * ldx11], 1);
                cblas_daxpy(p - i, -z1 * z3 * z4 * sin(phi[i - 1]),
                            &X12[i + (i - 1) * ldx12], 1, &X11[i + i * ldx11], 1);
            }
            if (i == 0) {
                cblas_dscal(m - p - i, z2, &X21[i + i * ldx21], 1);
            } else {
                cblas_dscal(m - p - i, z2 * cos(phi[i - 1]), &X21[i + i * ldx21], 1);
                cblas_daxpy(m - p - i, -z2 * z3 * z4 * sin(phi[i - 1]),
                            &X22[i + (i - 1) * ldx22], 1, &X21[i + i * ldx21], 1);
            }

            theta[i] = atan2(cblas_dnrm2(m - p - i, &X21[i + i * ldx21], 1),
                             cblas_dnrm2(p - i, &X11[i + i * ldx11], 1));

            if (p > i + 1) {
                dlarfgp(p - i, &X11[i + i * ldx11], &X11[(i + 1) + i * ldx11], 1,
                        &taup1[i]);
            } else if (p == i + 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X11[i + i * ldx11], NULL, 1, &taup1[i]);
            }
            if (m - p > i + 1) {
                dlarfgp(m - p - i, &X21[i + i * ldx21], &X21[(i + 1) + i * ldx21], 1,
                        &taup2[i]);
            } else if (m - p == i + 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X21[i + i * ldx21], NULL, 1, &taup2[i]);
            }

            if (q > i + 1) {
                dlarf1f("L", p - i, q - i - 1, &X11[i + i * ldx11], 1, taup1[i],
                        &X11[i + (i + 1) * ldx11], ldx11, work);
            }
            if (m - q >= i + 1) {
                dlarf1f("L", p - i, m - q - i, &X11[i + i * ldx11], 1, taup1[i],
                        &X12[i + i * ldx12], ldx12, work);
            }
            if (q > i + 1) {
                dlarf1f("L", m - p - i, q - i - 1, &X21[i + i * ldx21], 1, taup2[i],
                        &X21[i + (i + 1) * ldx21], ldx21, work);
            }
            if (m - q >= i + 1) {
                dlarf1f("L", m - p - i, m - q - i, &X21[i + i * ldx21], 1, taup2[i],
                        &X22[i + i * ldx22], ldx22, work);
            }

            if (i < q - 1) {
                cblas_dscal(q - i - 1, -z1 * z3 * sin(theta[i]), &X11[i + (i + 1) * ldx11],
                            ldx11);
                cblas_daxpy(q - i - 1, z2 * z3 * cos(theta[i]), &X21[i + (i + 1) * ldx21],
                            ldx21, &X11[i + (i + 1) * ldx11], ldx11);
            }
            cblas_dscal(m - q - i, -z1 * z4 * sin(theta[i]), &X12[i + i * ldx12],
                        ldx12);
            cblas_daxpy(m - q - i, z2 * z4 * cos(theta[i]), &X22[i + i * ldx22],
                        ldx22, &X12[i + i * ldx12], ldx12);

            if (i < q - 1) {
                phi[i] = atan2(cblas_dnrm2(q - i - 1, &X11[i + (i + 1) * ldx11], ldx11),
                               cblas_dnrm2(m - q - i, &X12[i + i * ldx12], ldx12));
            }

            if (i < q - 1) {
                if (q - i - 1 == 1) {
                    /* N=1: X not accessed */
                    dlarfgp(1, &X11[i + (i + 1) * ldx11], NULL, ldx11, &tauq1[i]);
                } else {
                    dlarfgp(q - i - 1, &X11[i + (i + 1) * ldx11],
                            &X11[i + (i + 2) * ldx11], ldx11, &tauq1[i]);
                }
            }
            if (q + i < m) {
                if (m - q == i + 1) {
                    /* N=1: X not accessed */
                    dlarfgp(1, &X12[i + i * ldx12], NULL, ldx12, &tauq2[i]);
                } else {
                    dlarfgp(m - q - i, &X12[i + i * ldx12], &X12[i + (i + 1) * ldx12], ldx12,
                            &tauq2[i]);
                }
            }

            if (i < q - 1) {
                dlarf1f("R", p - i - 1, q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                        tauq1[i], &X11[(i + 1) + (i + 1) * ldx11], ldx11, work);
                dlarf1f("R", m - p - i - 1, q - i - 1, &X11[i + (i + 1) * ldx11], ldx11,
                        tauq1[i], &X21[(i + 1) + (i + 1) * ldx21], ldx21, work);
            }
            if (p > i + 1) {
                dlarf1f("R", p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X12[(i + 1) + i * ldx12], ldx12, work);
            }
            if (m - p > i + 1) {
                dlarf1f("R", m - p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X22[(i + 1) + i * ldx22], ldx22, work);
            }

        }

        for (i = q; i < p; i++) {

            cblas_dscal(m - q - i, -z1 * z4, &X12[i + i * ldx12], ldx12);
            if (i >= m - q - 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X12[i + i * ldx12], NULL, ldx12, &tauq2[i]);
            } else {
                dlarfgp(m - q - i, &X12[i + i * ldx12], &X12[i + (i + 1) * ldx12], ldx12,
                        &tauq2[i]);
            }

            if (p > i + 1) {
                dlarf1f("R", p - i - 1, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X12[(i + 1) + i * ldx12], ldx12, work);
            }
            if (m - p - q >= 1) {
                dlarf1f("R", m - p - q, m - q - i, &X12[i + i * ldx12], ldx12,
                        tauq2[i], &X22[q + i * ldx22], ldx22, work);
            }

        }

        for (i = 0; i < m - p - q; i++) {

            cblas_dscal(m - p - q - i, z2 * z4, &X22[(q + i) + (p + i) * ldx22], ldx22);
            if (i == m - p - q - 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X22[(q + i) + (p + i) * ldx22], NULL, ldx22, &tauq2[p + i]);
            } else {
                dlarfgp(m - p - q - i, &X22[(q + i) + (p + i) * ldx22],
                        &X22[(q + i) + (p + i + 1) * ldx22], ldx22, &tauq2[p + i]);
            }
            if (i < m - p - q - 1) {
                dlarf1f("R", m - p - q - i - 1, m - p - q - i, &X22[(q + i) + (p + i) * ldx22],
                        ldx22, tauq2[p + i], &X22[(q + i + 1) + (p + i) * ldx22], ldx22, work);
            }

        }

    } else {

        for (i = 0; i < q; i++) {

            if (i == 0) {
                cblas_dscal(p - i, z1, &X11[i + i * ldx11], ldx11);
            } else {
                cblas_dscal(p - i, z1 * cos(phi[i - 1]), &X11[i + i * ldx11], ldx11);
                cblas_daxpy(p - i, -z1 * z3 * z4 * sin(phi[i - 1]),
                            &X12[(i - 1) + i * ldx12], ldx12, &X11[i + i * ldx11], ldx11);
            }
            if (i == 0) {
                cblas_dscal(m - p - i, z2, &X21[i + i * ldx21], ldx21);
            } else {
                cblas_dscal(m - p - i, z2 * cos(phi[i - 1]), &X21[i + i * ldx21], ldx21);
                cblas_daxpy(m - p - i, -z2 * z3 * z4 * sin(phi[i - 1]),
                            &X22[(i - 1) + i * ldx22], ldx22, &X21[i + i * ldx21], ldx21);
            }

            theta[i] = atan2(cblas_dnrm2(m - p - i, &X21[i + i * ldx21], ldx21),
                             cblas_dnrm2(p - i, &X11[i + i * ldx11], ldx11));

            dlarfgp(p - i, &X11[i + i * ldx11], &X11[i + (i + 1) * ldx11], ldx11,
                    &taup1[i]);
            if (i == m - p - 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X21[i + i * ldx21], NULL, ldx21, &taup2[i]);
            } else {
                dlarfgp(m - p - i, &X21[i + i * ldx21], &X21[i + (i + 1) * ldx21], ldx21,
                        &taup2[i]);
            }

            if (q > i + 1) {
                dlarf1f("R", q - i - 1, p - i, &X11[i + i * ldx11], ldx11, taup1[i],
                        &X11[(i + 1) + i * ldx11], ldx11, work);
            }
            if (m - q >= i + 1) {
                dlarf1f("R", m - q - i, p - i, &X11[i + i * ldx11], ldx11, taup1[i],
                        &X12[i + i * ldx12], ldx12, work);
            }
            if (q > i + 1) {
                dlarf1f("R", q - i - 1, m - p - i, &X21[i + i * ldx21], ldx21, taup2[i],
                        &X21[(i + 1) + i * ldx21], ldx21, work);
            }
            if (m - q >= i + 1) {
                dlarf1f("R", m - q - i, m - p - i, &X21[i + i * ldx21], ldx21, taup2[i],
                        &X22[i + i * ldx22], ldx22, work);
            }

            if (i < q - 1) {
                cblas_dscal(q - i - 1, -z1 * z3 * sin(theta[i]), &X11[(i + 1) + i * ldx11], 1);
                cblas_daxpy(q - i - 1, z2 * z3 * cos(theta[i]), &X21[(i + 1) + i * ldx21], 1,
                            &X11[(i + 1) + i * ldx11], 1);
            }
            cblas_dscal(m - q - i, -z1 * z4 * sin(theta[i]), &X12[i + i * ldx12], 1);
            cblas_daxpy(m - q - i, z2 * z4 * cos(theta[i]), &X22[i + i * ldx22], 1,
                        &X12[i + i * ldx12], 1);

            if (i < q - 1) {
                phi[i] = atan2(cblas_dnrm2(q - i - 1, &X11[(i + 1) + i * ldx11], 1),
                               cblas_dnrm2(m - q - i, &X12[i + i * ldx12], 1));
            }

            if (i < q - 1) {
                if (q - i - 1 == 1) {
                    /* N=1: X not accessed */
                    dlarfgp(1, &X11[(i + 1) + i * ldx11], NULL, 1, &tauq1[i]);
                } else {
                    dlarfgp(q - i - 1, &X11[(i + 1) + i * ldx11],
                            &X11[(i + 2) + i * ldx11], 1, &tauq1[i]);
                }
            }
            if (m - q > i + 1) {
                dlarfgp(m - q - i, &X12[i + i * ldx12], &X12[(i + 1) + i * ldx12], 1,
                        &tauq2[i]);
            } else {
                /* N=1: X not accessed */
                dlarfgp(1, &X12[i + i * ldx12], NULL, 1, &tauq2[i]);
            }

            if (i < q - 1) {
                dlarf1f("L", q - i - 1, p - i - 1, &X11[(i + 1) + i * ldx11], 1, tauq1[i],
                        &X11[(i + 1) + (i + 1) * ldx11], ldx11, work);
                dlarf1f("L", q - i - 1, m - p - i - 1, &X11[(i + 1) + i * ldx11], 1, tauq1[i],
                        &X21[(i + 1) + (i + 1) * ldx21], ldx21, work);
            }
            dlarf1f("L", m - q - i, p - i - 1, &X12[i + i * ldx12], 1, tauq2[i],
                    &X12[i + (i + 1) * ldx12], ldx12, work);
            if (m - p - i - 1 > 0) {
                dlarf1f("L", m - q - i, m - p - i - 1, &X12[i + i * ldx12], 1, tauq2[i],
                        &X22[i + (i + 1) * ldx22], ldx22, work);
            }

        }

        for (i = q; i < p; i++) {

            cblas_dscal(m - q - i, -z1 * z4, &X12[i + i * ldx12], 1);
            dlarfgp(m - q - i, &X12[i + i * ldx12], &X12[(i + 1) + i * ldx12], 1,
                    &tauq2[i]);

            if (p > i + 1) {
                dlarf1f("L", m - q - i, p - i - 1, &X12[i + i * ldx12], 1, tauq2[i],
                        &X12[i + (i + 1) * ldx12], ldx12, work);
            }
            if (m - p - q >= 1) {
                dlarf1f("L", m - q - i, m - p - q, &X12[i + i * ldx12], 1, tauq2[i],
                        &X22[i + q * ldx22], ldx22, work);
            }

        }

        for (i = 0; i < m - p - q; i++) {

            cblas_dscal(m - p - q - i, z2 * z4, &X22[(p + i) + (q + i) * ldx22], 1);
            if (m - p - q == i + 1) {
                /* N=1: X not accessed */
                dlarfgp(1, &X22[(p + i) + (q + i) * ldx22], NULL, 1, &tauq2[p + i]);
            } else {
                dlarfgp(m - p - q - i, &X22[(p + i) + (q + i) * ldx22],
                        &X22[(p + i + 1) + (q + i) * ldx22], 1, &tauq2[p + i]);
                dlarf1f("L", m - p - q - i, m - p - q - i - 1, &X22[(p + i) + (q + i) * ldx22],
                        1, tauq2[p + i], &X22[(p + i) + (q + i + 1) * ldx22], ldx22, work);
            }

        }

    }
}
