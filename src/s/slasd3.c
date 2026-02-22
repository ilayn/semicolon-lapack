/**
 * @file slasd3.c
 * @brief SLASD3 finds all square roots of the roots of the secular equation,
 *        then updates singular vectors by matrix multiplication.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include "semicolon_cblas.h"

/**
 * SLASD3 finds all the square roots of the roots of the secular
 * equation, as defined by the values in D and Z. It makes the
 * appropriate calls to SLASD4 and then updates the singular
 * vectors by matrix multiplication.
 *
 * SLASD3 is called from SLASD1.
 *
 * @param[in]     nl      Row dimension of upper block. nl >= 1.
 * @param[in]     nr      Row dimension of lower block. nr >= 1.
 * @param[in]     sqre    = 0: lower block is nr-by-nr square.
 *                         = 1: lower block is nr-by-(nr+1) rectangular.
 * @param[in]     k       Size of the secular equation. 1 <= k <= n.
 * @param[out]    D       Array of dimension k. Square roots of secular roots.
 * @param[out]    Q       Array (ldq, k). Workspace.
 * @param[in]     ldq     Leading dimension of Q. ldq >= k.
 * @param[in]     DSIGMA  Array of dimension k. Poles of secular equation.
 * @param[out]    U       Array (ldu, n). Left singular vectors.
 * @param[in]     ldu     Leading dimension of U. ldu >= n.
 * @param[in]     U2      Array (ldu2, n). Non-deflated left singular vectors.
 * @param[in]     ldu2    Leading dimension of U2. ldu2 >= n.
 * @param[out]    VT      Array (ldvt, m). Right singular vectors transposed.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= m.
 * @param[in,out] VT2     Array (ldvt2, n). Non-deflated right singular vectors.
 * @param[in]     ldvt2   Leading dimension of VT2. ldvt2 >= m.
 * @param[in]     IDXC    Integer array of dimension n. Column permutation.
 * @param[in]     CTOT    Integer array of dimension 4. Column type counts.
 * @param[in,out] Z       Array of dimension k. Deflation-adjusted updating row.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: not converged.
 */
void slasd3(const INT nl, const INT nr, const INT sqre, const INT k,
            f32* restrict D, f32* restrict Q, const INT ldq,
            const f32* restrict DSIGMA,
            f32* restrict U, const INT ldu,
            const f32* restrict U2, const INT ldu2,
            f32* restrict VT, const INT ldvt,
            f32* restrict VT2, const INT ldvt2,
            const INT* restrict IDXC, const INT* restrict CTOT,
            f32* restrict Z, INT* info)
{
    INT ctemp, i, j, jc, ktemp, m, n;
    f32 rho, temp;

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

    if (k < 1 || k > n) {
        *info = -4;
    } else if (ldq < k) {
        *info = -7;
    } else if (ldu < n) {
        *info = -10;
    } else if (ldu2 < n) {
        *info = -12;
    } else if (ldvt < m) {
        *info = -14;
    } else if (ldvt2 < m) {
        *info = -16;
    }
    if (*info != 0) {
        xerbla("SLASD3", -(*info));
        return;
    }

    if (k == 1) {
        D[0] = fabsf(Z[0]);
        cblas_scopy(m, VT2, ldvt2, VT, ldvt);
        if (Z[0] > 0.0f) {
            cblas_scopy(n, U2, 1, U, 1);
        } else {
            for (i = 0; i < n; i++) {
                U[i] = -U2[i];
            }
        }
        return;
    }

    cblas_scopy(k, Z, 1, Q, 1);

    rho = cblas_snrm2(k, Z, 1);
    slascl("G", 0, 0, rho, 1.0f, k, 1, Z, k, info);
    rho = rho * rho;

    for (j = 0; j < k; j++) {
        slasd4(k, j, DSIGMA, Z, &U[j * ldu], rho, &D[j], &VT[j * ldvt], info);
        if (*info != 0) {
            return;
        }
    }

    for (i = 0; i < k; i++) {
        Z[i] = U[i + (k - 1) * ldu] * VT[i + (k - 1) * ldvt];
        for (j = 0; j < i; j++) {
            Z[i] = Z[i] * (U[i + j * ldu] * VT[i + j * ldvt] /
                   (DSIGMA[i] - DSIGMA[j]) /
                   (DSIGMA[i] + DSIGMA[j]));
        }
        for (j = i; j < k - 1; j++) {
            Z[i] = Z[i] * (U[i + j * ldu] * VT[i + j * ldvt] /
                   (DSIGMA[i] - DSIGMA[j + 1]) /
                   (DSIGMA[i] + DSIGMA[j + 1]));
        }
        Z[i] = copysignf(sqrtf(fabsf(Z[i])), Q[i]);
    }

    for (i = 0; i < k; i++) {
        VT[0 + i * ldvt] = Z[0] / U[0 + i * ldu] / VT[0 + i * ldvt];
        U[0 + i * ldu] = -1.0f;
        for (j = 1; j < k; j++) {
            VT[j + i * ldvt] = Z[j] / U[j + i * ldu] / VT[j + i * ldvt];
            U[j + i * ldu] = DSIGMA[j] * VT[j + i * ldvt];
        }
        temp = cblas_snrm2(k, &U[i * ldu], 1);
        Q[0 + i * ldq] = U[0 + i * ldu] / temp;
        for (j = 1; j < k; j++) {
            jc = IDXC[j];
            Q[j + i * ldq] = U[jc + i * ldu] / temp;
        }
    }

    if (k == 2) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, k, k, 1.0f, U2, ldu2, Q, ldq, 0.0f, U, ldu);
        goto L100;
    }
    if (CTOT[0] > 0) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nl, k, CTOT[0], 1.0f, &U2[1 * ldu2], ldu2,
                    &Q[1], ldq, 0.0f, U, ldu);
        if (CTOT[2] > 0) {
            ktemp = 1 + CTOT[0] + CTOT[1];
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        nl, k, CTOT[2], 1.0f, &U2[ktemp * ldu2], ldu2,
                        &Q[ktemp], ldq, 1.0f, U, ldu);
        }
    } else if (CTOT[2] > 0) {
        ktemp = 1 + CTOT[0] + CTOT[1];
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nl, k, CTOT[2], 1.0f, &U2[ktemp * ldu2], ldu2,
                    &Q[ktemp], ldq, 0.0f, U, ldu);
    } else {
        slacpy("F", nl, k, U2, ldu2, U, ldu);
    }
    cblas_scopy(k, Q, ldq, &U[nl], ldu);
    ktemp = 1 + CTOT[0];
    ctemp = CTOT[1] + CTOT[2];
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                nr, k, ctemp, 1.0f, &U2[nl + 1 + ktemp * ldu2], ldu2,
                &Q[ktemp], ldq, 0.0f, &U[nl + 1], ldu);

L100:
    for (i = 0; i < k; i++) {
        temp = cblas_snrm2(k, &VT[i * ldvt], 1);
        Q[i + 0 * ldq] = VT[0 + i * ldvt] / temp;
        for (j = 1; j < k; j++) {
            jc = IDXC[j];
            Q[i + j * ldq] = VT[jc + i * ldvt] / temp;
        }
    }

    if (k == 2) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    k, m, k, 1.0f, Q, ldq, VT2, ldvt2, 0.0f, VT, ldvt);
        return;
    }
    ktemp = 1 + CTOT[0];
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                k, nl + 1, ktemp, 1.0f, Q, ldq, VT2, ldvt2, 0.0f, VT, ldvt);
    ktemp = 1 + CTOT[0] + CTOT[1];
    if (ktemp <= ldvt2) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    k, nl + 1, CTOT[2], 1.0f, &Q[ktemp * ldq], ldq,
                    &VT2[ktemp], ldvt2, 1.0f, VT, ldvt);
    }

    ktemp = CTOT[0];
    INT nrp1 = nr + sqre;
    if (ktemp > 0) {
        for (i = 0; i < k; i++) {
            Q[i + ktemp * ldq] = Q[i + 0 * ldq];
        }
        for (i = nl + 1; i < m; i++) {
            VT2[ktemp + i * ldvt2] = VT2[0 + i * ldvt2];
        }
    }
    ctemp = 1 + CTOT[1] + CTOT[2];
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                k, nrp1, ctemp, 1.0f, &Q[ktemp * ldq], ldq,
                &VT2[ktemp + (nl + 1) * ldvt2], ldvt2, 0.0f, &VT[(nl + 1) * ldvt], ldvt);
}
