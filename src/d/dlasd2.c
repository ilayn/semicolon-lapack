/**
 * @file dlasd2.c
 * @brief DLASD2 merges the two sets of singular values together into a single
 *        sorted set and performs deflation.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/**
 * DLASD2 merges the two sets of singular values together into a single
 * sorted set. Then it tries to deflate the size of the problem.
 * There are two ways in which deflation can occur: when two or more
 * singular values are close together or if there is a tiny entry in the
 * Z vector. For each such occurrence the order of the related secular
 * equation problem is reduced by one.
 *
 * DLASD2 is called from DLASD1.
 *
 * @param[in]     nl      The row dimension of the upper block. nl >= 1.
 * @param[in]     nr      The row dimension of the lower block. nr >= 1.
 * @param[in]     sqre    = 0: lower block is nr-by-nr square matrix.
 *                         = 1: lower block is nr-by-(nr+1) rectangular.
 * @param[out]    k       Dimension of the non-deflated matrix. 1 <= k <= n.
 * @param[in,out] D       Array of dimension n. On entry, singular values of
 *                        two submatrices. On exit, trailing n-k deflated values.
 * @param[out]    Z       Array of dimension n. The updating row vector.
 * @param[in]     alpha   Diagonal element associated with added row.
 * @param[in]     beta    Off-diagonal element associated with added row.
 * @param[in,out] U       Array (ldu, n). Left singular vectors.
 * @param[in]     ldu     Leading dimension of U. ldu >= n.
 * @param[in,out] VT      Array (ldvt, m). Right singular vectors transposed.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= m.
 * @param[out]    DSIGMA  Array of dimension n. Copy of diagonal elements.
 * @param[out]    U2      Array (ldu2, n). Copy of left singular vectors.
 * @param[in]     ldu2    Leading dimension of U2. ldu2 >= n.
 * @param[out]    VT2     Array (ldvt2, n). Copy of right singular vectors.
 * @param[in]     ldvt2   Leading dimension of VT2. ldvt2 >= m.
 * @param[out]    IDXP    Integer array of dimension n. Permutation for deflation.
 * @param[out]    IDX     Integer array of dimension n. Sorting permutation.
 * @param[out]    IDXC    Integer array of dimension n. Column arrangement permutation.
 * @param[in,out] IDXQ    Integer array of dimension n. Sorting permutation for subproblems.
 * @param[out]    COLTYP  Integer array of dimension n. Column type labels.
 * @param[out]    info
 *                         - = 0: successful exit. < 0: illegal argument.
 */
void dlasd2(const INT nl, const INT nr, const INT sqre, INT* k,
            f64* restrict D, f64* restrict Z,
            const f64 alpha, const f64 beta,
            f64* restrict U, const INT ldu,
            f64* restrict VT, const INT ldvt,
            f64* restrict DSIGMA,
            f64* restrict U2, const INT ldu2,
            f64* restrict VT2, const INT ldvt2,
            INT* restrict IDXP, INT* restrict IDX,
            INT* restrict IDXC, INT* restrict IDXQ,
            INT* restrict COLTYP, INT* info)
{
    INT ctot[4], psm[4];
    INT ct, i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n;
    f64 c, eps, hlftol, s, tau, tol, z1;

    *info = 0;

    if (nl < 1) {
        *info = -1;
    } else if (nr < 1) {
        *info = -2;
    } else if (sqre != 1 && sqre != 0) {
        *info = -3;
    }

    n = nl + nr + 1;
    m = n + sqre;

    if (ldu < n) {
        *info = -10;
    } else if (ldvt < m) {
        *info = -12;
    } else if (ldu2 < n) {
        *info = -15;
    } else if (ldvt2 < m) {
        *info = -17;
    }
    if (*info != 0) {
        xerbla("DLASD2", -(*info));
        return;
    }

    z1 = alpha * VT[nl + nl * ldvt];
    Z[0] = z1;
    for (i = nl - 1; i >= 0; i--) {
        Z[i + 1] = alpha * VT[i + nl * ldvt];
        D[i + 1] = D[i];
        IDXQ[i + 1] = IDXQ[i] + 1;
    }

    for (i = nl + 1; i < m; i++) {
        Z[i] = beta * VT[i + (nl + 1) * ldvt];
    }

    for (i = 1; i <= nl; i++) {
        COLTYP[i] = 1;
    }
    for (i = nl + 1; i < n; i++) {
        COLTYP[i] = 2;
    }

    for (i = nl + 1; i < n; i++) {
        IDXQ[i] = IDXQ[i] + nl + 1;
    }

    for (i = 1; i < n; i++) {
        DSIGMA[i] = D[IDXQ[i]];
        U2[i] = Z[IDXQ[i]];
        IDXC[i] = COLTYP[IDXQ[i]];
    }

    dlamrg(nl, nr, &DSIGMA[1], 1, 1, &IDX[1]);

    for (i = 1; i < n; i++) {
        idxi = 1 + IDX[i];
        D[i] = DSIGMA[idxi];
        Z[i] = U2[idxi];
        COLTYP[i] = IDXC[idxi];
    }

    eps = dlamch("Epsilon");
    tol = fabs(alpha) > fabs(beta) ? fabs(alpha) : fabs(beta);
    tol = 8.0 * eps * (fabs(D[n - 1]) > tol ? fabs(D[n - 1]) : tol);

    *k = 1;
    k2 = n;
    jprev = -1;

    for (j = 1; j < n; j++) {
        if (fabs(Z[j]) <= tol) {
            k2--;
            IDXP[k2] = j;
            COLTYP[j] = 4;
            if (j == n - 1) {
                goto L120;
            }
        } else {
            jprev = j;
            break;
        }
    }

    j = jprev;
    while (1) {
        j++;
        if (j >= n) {
            break;
        }
        if (fabs(Z[j]) <= tol) {
            k2--;
            IDXP[k2] = j;
            COLTYP[j] = 4;
        } else {
            if (fabs(D[j] - D[jprev]) <= tol) {
                s = Z[jprev];
                c = Z[j];
                tau = dlapy2(c, s);
                c = c / tau;
                s = -s / tau;
                Z[j] = tau;
                Z[jprev] = 0.0;

                idxjp = IDXQ[IDX[jprev] + 1];
                idxj = IDXQ[IDX[j] + 1];
                if (idxjp <= nl) {
                    idxjp--;
                }
                if (idxj <= nl) {
                    idxj--;
                }
                cblas_drot(n, &U[idxjp * ldu], 1, &U[idxj * ldu], 1, c, s);
                cblas_drot(m, &VT[idxjp], ldvt, &VT[idxj], ldvt, c, s);

                if (COLTYP[j] != COLTYP[jprev]) {
                    COLTYP[j] = 3;
                }
                COLTYP[jprev] = 4;
                k2--;
                IDXP[k2] = jprev;
                jprev = j;
            } else {
                (*k)++;
                U2[*k - 1] = Z[jprev];
                DSIGMA[*k - 1] = D[jprev];
                IDXP[*k - 1] = jprev;
                jprev = j;
            }
        }
    }

    (*k)++;
    U2[*k - 1] = Z[jprev];
    DSIGMA[*k - 1] = D[jprev];
    IDXP[*k - 1] = jprev;

L120:
    for (j = 0; j < 4; j++) {
        ctot[j] = 0;
    }
    for (j = 1; j < n; j++) {
        ct = COLTYP[j];
        ctot[ct - 1]++;
    }

    psm[0] = 1;
    psm[1] = 1 + ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];

    for (j = 1; j < n; j++) {
        jp = IDXP[j];
        ct = COLTYP[jp];
        IDXC[psm[ct - 1]] = j;
        psm[ct - 1]++;
    }

    for (j = 1; j < n; j++) {
        jp = IDXP[j];
        DSIGMA[j] = D[jp];
        idxj = IDXQ[IDX[IDXP[IDXC[j]]] + 1];
        if (idxj <= nl) {
            idxj--;
        }
        cblas_dcopy(n, &U[idxj * ldu], 1, &U2[j * ldu2], 1);
        cblas_dcopy(m, &VT[idxj], ldvt, &VT2[j], ldvt2);
    }

    DSIGMA[0] = 0.0;
    hlftol = tol / 2.0;
    if (fabs(DSIGMA[1]) <= hlftol) {
        DSIGMA[1] = hlftol;
    }
    if (m > n) {
        Z[0] = dlapy2(z1, Z[m - 1]);
        if (Z[0] <= tol) {
            c = 1.0;
            s = 0.0;
            Z[0] = tol;
        } else {
            c = z1 / Z[0];
            s = Z[m - 1] / Z[0];
        }
    } else {
        if (fabs(z1) <= tol) {
            Z[0] = tol;
        } else {
            Z[0] = z1;
        }
    }

    cblas_dcopy(*k - 1, &U2[1], 1, &Z[1], 1);

    dlaset("A", n, 1, 0.0, 0.0, U2, ldu2);
    U2[nl] = 1.0;

    if (m > n) {
        for (i = 0; i <= nl; i++) {
            VT[m - 1 + i * ldvt] = -s * VT[nl + i * ldvt];
            VT2[i * ldvt2] = c * VT[nl + i * ldvt];
        }
        for (i = nl + 1; i < m; i++) {
            VT2[i * ldvt2] = s * VT[m - 1 + i * ldvt];
            VT[m - 1 + i * ldvt] = c * VT[m - 1 + i * ldvt];
        }
    } else {
        cblas_dcopy(m, &VT[nl], ldvt, VT2, ldvt2);
    }
    if (m > n) {
        cblas_dcopy(m, &VT[m - 1], ldvt, &VT2[m - 1], ldvt2);
    }

    if (n > *k) {
        cblas_dcopy(n - *k, &DSIGMA[*k], 1, &D[*k], 1);
        dlacpy("A", n, n - *k, &U2[*k * ldu2], ldu2, &U[*k * ldu], ldu);
        dlacpy("A", n - *k, m, &VT2[*k], ldvt2, &VT[*k], ldvt);
    }

    for (j = 0; j < 4; j++) {
        COLTYP[j] = ctot[j];
    }
}
