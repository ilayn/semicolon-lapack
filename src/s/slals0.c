/**
 * @file slals0.c
 * @brief SLALS0 applies back multiplying factors in solving the least squares
 *        problem using divide and conquer SVD approach. Used by sgelsd.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include "semicolon_cblas.h"

/** @cond */
static inline f32 dlamc3(f32 a, f32 b)
{
    volatile f32 result = a + b;
    return result;
}
/** @endcond */

void slals0(const INT icompq, const INT nl, const INT nr, const INT sqre,
            const INT nrhs, f32* restrict B, const INT ldb,
            f32* restrict BX, const INT ldbx,
            const INT* restrict perm, const INT givptr,
            const INT* restrict givcol, const INT ldgcol,
            const f32* restrict givnum, const INT ldgnum,
            const f32* restrict poles, const f32* restrict difl,
            const f32* restrict difr, const f32* restrict Z,
            const INT k, const f32 c, const f32 s,
            f32* restrict work, INT* info)
{
    INT i, j, m, n, nlp1;
    f32 diflj, difrj = 0.0f, dj, dsigj, dsigjp = 0.0f, temp;

    *info = 0;
    n = nl + nr + 1;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (nl < 1) {
        *info = -2;
    } else if (nr < 1) {
        *info = -3;
    } else if (sqre < 0 || sqre > 1) {
        *info = -4;
    } else if (nrhs < 1) {
        *info = -5;
    } else if (ldb < n) {
        *info = -7;
    } else if (ldbx < n) {
        *info = -9;
    } else if (givptr < 0) {
        *info = -11;
    } else if (ldgcol < n) {
        *info = -13;
    } else if (ldgnum < n) {
        *info = -15;
    } else if (k < 1) {
        *info = -20;
    }
    if (*info != 0) {
        xerbla("SLALS0", -(*info));
        return;
    }

    m = n + sqre;
    nlp1 = nl + 1;

    if (icompq == 0) {
        for (i = 1; i <= givptr; i++) {
            cblas_srot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                       &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                       givnum[i - 1 + 1 * ldgnum], givnum[i - 1 + 0 * ldgnum]);
        }

        cblas_scopy(nrhs, &B[nlp1 - 1], ldb, &BX[0], ldbx);
        for (i = 2; i <= n; i++) {
            cblas_scopy(nrhs, &B[perm[i - 1]], ldb, &BX[i - 1], ldbx);
        }

        if (k == 1) {
            cblas_scopy(nrhs, BX, ldbx, B, ldb);
            if (Z[0] < 0.0f) {
                cblas_sscal(nrhs, -1.0f, B, ldb);
            }
        } else {
            for (j = 1; j <= k; j++) {
                diflj = difl[j - 1];
                dj = poles[j - 1 + 0 * ldgnum];
                dsigj = -poles[j - 1 + 1 * ldgnum];
                if (j < k) {
                    difrj = -difr[j - 1 + 0 * ldgnum];
                    dsigjp = -poles[j + 1 * ldgnum];
                }
                if (Z[j - 1] == 0.0f || poles[j - 1 + 1 * ldgnum] == 0.0f) {
                    work[j - 1] = 0.0f;
                } else {
                    work[j - 1] = -poles[j - 1 + 1 * ldgnum] * Z[j - 1] / diflj /
                                  (poles[j - 1 + 1 * ldgnum] + dj);
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[i - 1] == 0.0f || poles[i - 1 + 1 * ldgnum] == 0.0f) {
                        work[i - 1] = 0.0f;
                    } else {
                        work[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                      (dlamc3(poles[i - 1 + 1 * ldgnum], dsigj) - diflj) /
                                      (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[i - 1] == 0.0f || poles[i - 1 + 1 * ldgnum] == 0.0f) {
                        work[i - 1] = 0.0f;
                    } else {
                        work[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                      (dlamc3(poles[i - 1 + 1 * ldgnum], dsigjp) + difrj) /
                                      (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                work[0] = -1.0f;
                temp = cblas_snrm2(k, work, 1);
                cblas_sgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0f, BX, ldbx,
                            work, 1, 0.0f, &B[j - 1], ldb);
                slascl("G", 0, 0, temp, 1.0f, 1, nrhs, &B[j - 1], ldb, info);
            }
        }

        if (k < (m > n ? m : n)) {
            slacpy("A", n - k, nrhs, &BX[k], ldbx, &B[k], ldb);
        }
    } else {
        if (k == 1) {
            cblas_scopy(nrhs, B, ldb, BX, ldbx);
        } else {
            for (j = 1; j <= k; j++) {
                dsigj = poles[j - 1 + 1 * ldgnum];
                if (Z[j - 1] == 0.0f) {
                    work[j - 1] = 0.0f;
                } else {
                    work[j - 1] = -Z[j - 1] / difl[j - 1] /
                                  (dsigj + poles[j - 1 + 0 * ldgnum]) / difr[j - 1 + 1 * ldgnum];
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[j - 1] == 0.0f) {
                        work[i - 1] = 0.0f;
                    } else {
                        work[i - 1] = Z[j - 1] /
                                      (dlamc3(dsigj, -poles[i + 1 * ldgnum]) - difr[i - 1 + 0 * ldgnum]) /
                                      (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[j - 1] == 0.0f) {
                        work[i - 1] = 0.0f;
                    } else {
                        work[i - 1] = Z[j - 1] /
                                      (dlamc3(dsigj, -poles[i - 1 + 1 * ldgnum]) - difl[i - 1]) /
                                      (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }
                cblas_sgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0f, B, ldb,
                            work, 1, 0.0f, &BX[j - 1], ldbx);
            }
        }

        if (sqre == 1) {
            cblas_scopy(nrhs, &B[m - 1], ldb, &BX[m - 1], ldbx);
            cblas_srot(nrhs, &BX[0], ldbx, &BX[m - 1], ldbx, c, s);
        }
        if (k < (m > n ? m : n)) {
            slacpy("A", n - k, nrhs, &B[k], ldb, &BX[k], ldbx);
        }

        cblas_scopy(nrhs, &BX[0], ldbx, &B[nlp1 - 1], ldb);
        if (sqre == 1) {
            cblas_scopy(nrhs, &BX[m - 1], ldbx, &B[m - 1], ldb);
        }
        for (i = 2; i <= n; i++) {
            cblas_scopy(nrhs, &BX[i - 1], ldbx, &B[perm[i - 1]], ldb);
        }

        for (i = givptr; i >= 1; i--) {
            cblas_srot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                       &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                       givnum[i - 1 + 1 * ldgnum], -givnum[i - 1 + 0 * ldgnum]);
        }
    }
}
