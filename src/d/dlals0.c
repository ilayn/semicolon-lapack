/**
 * @file dlals0.c
 * @brief DLALS0 applies back multiplying factors in solving the least squares
 *        problem using divide and conquer SVD approach. Used by dgelsd.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

static inline f64 dlamc3(f64 a, f64 b)
{
    volatile f64 result = a + b;
    return result;
}

void dlals0(const int icompq, const int nl, const int nr, const int sqre,
            const int nrhs, f64* restrict B, const int ldb,
            f64* restrict BX, const int ldbx,
            const int* restrict perm, const int givptr,
            const int* restrict givcol, const int ldgcol,
            const f64* restrict givnum, const int ldgnum,
            const f64* restrict poles, const f64* restrict difl,
            const f64* restrict difr, const f64* restrict Z,
            const int k, const f64 c, const f64 s,
            f64* restrict work, int* info)
{
    int i, j, m, n, nlp1;
    f64 diflj, difrj = 0.0, dj, dsigj, dsigjp = 0.0, temp;

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
        xerbla("DLALS0", -(*info));
        return;
    }

    m = n + sqre;
    nlp1 = nl + 1;

    if (icompq == 0) {
        for (i = 1; i <= givptr; i++) {
            cblas_drot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                       &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                       givnum[i - 1 + 1 * ldgnum], givnum[i - 1 + 0 * ldgnum]);
        }

        cblas_dcopy(nrhs, &B[nlp1 - 1], ldb, &BX[0], ldbx);
        for (i = 2; i <= n; i++) {
            cblas_dcopy(nrhs, &B[perm[i - 1]], ldb, &BX[i - 1], ldbx);
        }

        if (k == 1) {
            cblas_dcopy(nrhs, BX, ldbx, B, ldb);
            if (Z[0] < 0.0) {
                cblas_dscal(nrhs, -1.0, B, ldb);
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
                if (Z[j - 1] == 0.0 || poles[j - 1 + 1 * ldgnum] == 0.0) {
                    work[j - 1] = 0.0;
                } else {
                    work[j - 1] = -poles[j - 1 + 1 * ldgnum] * Z[j - 1] / diflj /
                                  (poles[j - 1 + 1 * ldgnum] + dj);
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[i - 1] == 0.0 || poles[i - 1 + 1 * ldgnum] == 0.0) {
                        work[i - 1] = 0.0;
                    } else {
                        work[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                      (dlamc3(poles[i - 1 + 1 * ldgnum], dsigj) - diflj) /
                                      (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[i - 1] == 0.0 || poles[i - 1 + 1 * ldgnum] == 0.0) {
                        work[i - 1] = 0.0;
                    } else {
                        work[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                      (dlamc3(poles[i - 1 + 1 * ldgnum], dsigjp) + difrj) /
                                      (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                work[0] = -1.0;
                temp = cblas_dnrm2(k, work, 1);
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0, BX, ldbx,
                            work, 1, 0.0, &B[j - 1], ldb);
                dlascl("G", 0, 0, temp, 1.0, 1, nrhs, &B[j - 1], ldb, info);
            }
        }

        if (k < (m > n ? m : n)) {
            dlacpy("A", n - k, nrhs, &BX[k], ldbx, &B[k], ldb);
        }
    } else {
        if (k == 1) {
            cblas_dcopy(nrhs, B, ldb, BX, ldbx);
        } else {
            for (j = 1; j <= k; j++) {
                dsigj = poles[j - 1 + 1 * ldgnum];
                if (Z[j - 1] == 0.0) {
                    work[j - 1] = 0.0;
                } else {
                    work[j - 1] = -Z[j - 1] / difl[j - 1] /
                                  (dsigj + poles[j - 1 + 0 * ldgnum]) / difr[j - 1 + 1 * ldgnum];
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[j - 1] == 0.0) {
                        work[i - 1] = 0.0;
                    } else {
                        work[i - 1] = Z[j - 1] /
                                      (dlamc3(dsigj, -poles[i + 1 * ldgnum]) - difr[i - 1 + 0 * ldgnum]) /
                                      (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[j - 1] == 0.0) {
                        work[i - 1] = 0.0;
                    } else {
                        work[i - 1] = Z[j - 1] /
                                      (dlamc3(dsigj, -poles[i - 1 + 1 * ldgnum]) - difl[i - 1]) /
                                      (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0, B, ldb,
                            work, 1, 0.0, &BX[j - 1], ldbx);
            }
        }

        if (sqre == 1) {
            cblas_dcopy(nrhs, &B[m - 1], ldb, &BX[m - 1], ldbx);
            cblas_drot(nrhs, &BX[0], ldbx, &BX[m - 1], ldbx, c, s);
        }
        if (k < (m > n ? m : n)) {
            dlacpy("A", n - k, nrhs, &B[k], ldb, &BX[k], ldbx);
        }

        cblas_dcopy(nrhs, &BX[0], ldbx, &B[nlp1 - 1], ldb);
        if (sqre == 1) {
            cblas_dcopy(nrhs, &BX[m - 1], ldbx, &B[m - 1], ldb);
        }
        for (i = 2; i <= n; i++) {
            cblas_dcopy(nrhs, &BX[i - 1], ldbx, &B[perm[i - 1]], ldb);
        }

        for (i = givptr; i >= 1; i--) {
            cblas_drot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                       &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                       givnum[i - 1 + 1 * ldgnum], -givnum[i - 1 + 0 * ldgnum]);
        }
    }
}
