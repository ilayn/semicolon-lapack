/**
 * @file dlasd7.c
 * @brief DLASD7 merges the two sets of singular values together into a single
 *        sorted set. Then it tries to deflate the size of the problem.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

void dlasd7(const INT icompq, const INT nl, const INT nr, const INT sqre,
            INT* k, f64* restrict D, f64* restrict Z,
            f64* restrict ZW, f64* restrict VF,
            f64* restrict VFW, f64* restrict VL,
            f64* restrict VLW, const f64 alpha, const f64 beta,
            f64* restrict DSIGMA, INT* restrict IDX,
            INT* restrict IDXP, INT* restrict IDXQ,
            INT* restrict PERM, INT* givptr,
            INT* restrict GIVCOL, const INT ldgcol,
            f64* restrict GIVNUM, const INT ldgnum,
            f64* c, f64* s, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 EIGHT = 8.0;

    INT i, idxi, idxj, idxjp, j, jp, jprev = 0, k2, m, n, nlp1, nlp2;
    f64 eps, hlftol, tau, tol, z1;

    *info = 0;
    n = nl + nr + 1;
    m = n + sqre;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (nl < 1) {
        *info = -2;
    } else if (nr < 1) {
        *info = -3;
    } else if (sqre < 0 || sqre > 1) {
        *info = -4;
    } else if (ldgcol < n) {
        *info = -22;
    } else if (ldgnum < n) {
        *info = -24;
    }
    if (*info != 0) {
        xerbla("DLASD7", -(*info));
        return;
    }

    nlp1 = nl + 1;
    nlp2 = nl + 2;
    if (icompq == 1) {
        *givptr = 0;
    }

    z1 = alpha * VL[nl];
    VL[nl] = ZERO;
    tau = VF[nl];
    for (i = nl; i >= 1; i--) {
        Z[i] = alpha * VL[i - 1];
        VL[i - 1] = ZERO;
        VF[i] = VF[i - 1];
        D[i] = D[i - 1];
        IDXQ[i] = IDXQ[i - 1] + 1;
    }
    VF[0] = tau;

    for (i = nlp2 - 1; i < m; i++) {
        Z[i] = beta * VF[i];
        VF[i] = ZERO;
    }

    for (i = nlp2 - 1; i < n; i++) {
        IDXQ[i] = IDXQ[i] + nlp1;
    }

    for (i = 1; i < n; i++) {
        DSIGMA[i] = D[IDXQ[i] - 1];
        ZW[i] = Z[IDXQ[i] - 1];
        VFW[i] = VF[IDXQ[i] - 1];
        VLW[i] = VL[IDXQ[i] - 1];
    }

    dlamrg(nl, nr, &DSIGMA[1], 1, 1, &IDX[1]);

    for (i = 1; i < n; i++) {
        idxi = 1 + IDX[i];
        D[i] = DSIGMA[idxi];
        Z[i] = ZW[idxi];
        VF[i] = VFW[idxi];
        VL[i] = VLW[idxi];
    }

    eps = dlamch("Epsilon");
    tol = fabs(alpha) > fabs(beta) ? fabs(alpha) : fabs(beta);
    tol = EIGHT * EIGHT * eps * (fabs(D[n - 1]) > tol ? fabs(D[n - 1]) : tol);

    *k = 1;
    k2 = n;

    for (j = 1; j < n; j++) {
        if (fabs(Z[j]) <= tol) {
            k2 = k2 - 1;
            IDXP[k2] = j;
            if (j == n - 1) {
                goto L100;
            }
        } else {
            jprev = j;
            goto L70;
        }
    }
L70:
    j = jprev;
L80:
    j = j + 1;
    if (j > n - 1) {
        goto L90;
    }
    if (fabs(Z[j]) <= tol) {
        k2 = k2 - 1;
        IDXP[k2] = j;
    } else {
        if (fabs(D[j] - D[jprev]) <= tol) {
            *s = Z[jprev];
            *c = Z[j];

            tau = dlapy2(*c, *s);
            Z[j] = tau;
            Z[jprev] = ZERO;
            *c = *c / tau;
            *s = -(*s) / tau;

            if (icompq == 1) {
                (*givptr)++;
                idxjp = IDXQ[IDX[jprev] + 1];
                idxj = IDXQ[IDX[j] + 1];
                if (idxjp <= nlp1) {
                    idxjp = idxjp - 1;
                }
                if (idxj <= nlp1) {
                    idxj = idxj - 1;
                }
                GIVCOL[*givptr - 1 + 1 * ldgcol] = idxjp - 1;
                GIVCOL[*givptr - 1 + 0 * ldgcol] = idxj - 1;
                GIVNUM[*givptr - 1 + 1 * ldgnum] = *c;
                GIVNUM[*givptr - 1 + 0 * ldgnum] = *s;
            }
            cblas_drot(1, &VF[jprev], 1, &VF[j], 1, *c, *s);
            cblas_drot(1, &VL[jprev], 1, &VL[j], 1, *c, *s);
            k2 = k2 - 1;
            IDXP[k2] = jprev;
            jprev = j;
        } else {
            (*k)++;
            ZW[*k - 1] = Z[jprev];
            DSIGMA[*k - 1] = D[jprev];
            IDXP[*k - 1] = jprev;
            jprev = j;
        }
    }
    goto L80;
L90:
    (*k)++;
    ZW[*k - 1] = Z[jprev];
    DSIGMA[*k - 1] = D[jprev];
    IDXP[*k - 1] = jprev;

L100:
    for (j = 1; j < n; j++) {
        jp = IDXP[j];
        DSIGMA[j] = D[jp];
        VFW[j] = VF[jp];
        VLW[j] = VL[jp];
    }
    if (icompq == 1) {
        for (j = 1; j < n; j++) {
            jp = IDXP[j];
            PERM[j] = IDXQ[IDX[jp] + 1];
            if (PERM[j] <= nlp1) {
                PERM[j] = PERM[j] - 1;
            }
            PERM[j] = PERM[j] - 1;
        }
    }

    cblas_dcopy(n - *k, &DSIGMA[*k], 1, &D[*k], 1);

    DSIGMA[0] = ZERO;
    hlftol = tol / TWO;
    if (fabs(DSIGMA[1]) <= hlftol) {
        DSIGMA[1] = hlftol;
    }
    if (m > n) {
        Z[0] = dlapy2(z1, Z[m - 1]);
        if (Z[0] <= tol) {
            *c = ONE;
            *s = ZERO;
            Z[0] = tol;
        } else {
            *c = z1 / Z[0];
            *s = -Z[m - 1] / Z[0];
        }
        cblas_drot(1, &VF[m - 1], 1, &VF[0], 1, *c, *s);
        cblas_drot(1, &VL[m - 1], 1, &VL[0], 1, *c, *s);
    } else {
        if (fabs(z1) <= tol) {
            Z[0] = tol;
        } else {
            Z[0] = z1;
        }
    }

    cblas_dcopy(*k - 1, &ZW[1], 1, &Z[1], 1);
    cblas_dcopy(n - 1, &VFW[1], 1, &VF[1], 1);
    cblas_dcopy(n - 1, &VLW[1], 1, &VL[1], 1);
}
