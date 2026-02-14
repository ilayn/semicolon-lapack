/**
 * @file zlals0.c
 * @brief ZLALS0 applies back multiplying factors in solving the least squares
 *        problem using divide and conquer SVD approach. Used by zgelsd.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static inline f64 dlamc3(f64 a, f64 b)
{
    volatile f64 result = a + b;
    return result;
}

/**
 * ZLALS0 applies back the multiplying factors of either the left or the
 * right singular vector matrix of a diagonal matrix appended by a row
 * to the right hand side matrix B in solving the least squares problem
 * using the divide-and-conquer SVD approach.
 *
 * For the left singular vector matrix, three types of orthogonal
 * matrices are involved:
 *
 * (1L) Givens rotations: the number of such rotations is GIVPTR; the
 *      pairs of columns/rows they were applied to are stored in GIVCOL;
 *      and the C- and S-values of these rotations are stored in GIVNUM.
 *
 * (2L) Permutation. The (NL+1)-st row of B is to be moved to the first
 *      row, and for J=2:N, PERM(J)-th row of B is to be moved to the
 *      J-th row.
 *
 * (3L) The left singular vector matrix of the remaining matrix.
 *
 * For the right singular vector matrix, four types of orthogonal
 * matrices are involved:
 *
 * (1R) The right singular vector matrix of the remaining matrix.
 *
 * (2R) If SQRE = 1, one extra Givens rotation to generate the right
 *      null space.
 *
 * (3R) The inverse transformation of (2L).
 *
 * (4R) The inverse transformation of (1L).
 *
 * @param[in]     icompq  Specifies whether singular vectors are to be computed
 *                        in factored form:
 *                        = 0: Left singular vector matrix.
 *                        = 1: Right singular vector matrix.
 * @param[in]     nl      The row dimension of the upper block. nl >= 1.
 * @param[in]     nr      The row dimension of the lower block. nr >= 1.
 * @param[in]     sqre    = 0: the lower block is an NR-by-NR square matrix.
 *                        = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
 *                        The bidiagonal matrix has row dimension N = NL + NR + 1,
 *                        and column dimension M = N + SQRE.
 * @param[in]     nrhs    The number of columns of B and BX. nrhs must be at least 1.
 * @param[in,out] B       Complex array, dimension (ldb, nrhs).
 *                        On input, B contains the right hand sides of the least
 *                        squares problem in rows 1 through M. On output, B contains
 *                        the solution X in rows 1 through N.
 * @param[in]     ldb     The leading dimension of B. ldb must be at least
 *                        max(1, max(M, N)).
 * @param[out]    BX      Complex array, dimension (ldbx, nrhs).
 * @param[in]     ldbx    The leading dimension of BX.
 * @param[in]     perm    Integer array, dimension (N).
 *                        The permutations (from deflation and sorting) applied
 *                        to the two blocks.
 * @param[in]     givptr  The number of Givens rotations which took place in this
 *                        subproblem.
 * @param[in]     givcol  Integer array, dimension (ldgcol, 2).
 *                        Each pair of numbers indicates a pair of rows/columns
 *                        involved in a Givens rotation.
 * @param[in]     ldgcol  The leading dimension of GIVCOL, must be at least N.
 * @param[in]     givnum  Double array, dimension (ldgnum, 2).
 *                        Each number indicates the C or S value used in the
 *                        corresponding Givens rotation.
 * @param[in]     ldgnum  The leading dimension of arrays DIFR, POLES and
 *                        GIVNUM, must be at least K.
 * @param[in]     poles   Double array, dimension (ldgnum, 2).
 *                        On entry, POLES(1:K, 1) contains the new singular
 *                        values obtained from solving the secular equation, and
 *                        POLES(1:K, 2) is an array containing the poles in the
 *                        secular equation.
 * @param[in]     difl    Double array, dimension (K).
 *                        On entry, DIFL(I) is the distance between I-th updated
 *                        (undeflated) singular value and the I-th (undeflated) old
 *                        singular value.
 * @param[in]     difr    Double array, dimension (ldgnum, 2).
 *                        On entry, DIFR(I, 1) contains the distances between I-th
 *                        updated (undeflated) singular value and the I+1-th
 *                        (undeflated) old singular value. And DIFR(I, 2) is the
 *                        normalizing factor for the I-th right singular vector.
 * @param[in]     Z       Double array, dimension (K).
 *                        Contain the components of the deflation-adjusted updating
 *                        row vector.
 * @param[in]     k       Contains the dimension of the non-deflated matrix,
 *                        This is the order of the related secular equation.
 *                        1 <= K <= N.
 * @param[in]     c       C contains garbage if SQRE = 0 and the C-value of a Givens
 *                        rotation related to the right null space if SQRE = 1.
 * @param[in]     s       S contains garbage if SQRE = 0 and the S-value of a Givens
 *                        rotation related to the right null space if SQRE = 1.
 * @param[out]    rwork   Double array, dimension (K*(1+NRHS) + 2*NRHS).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void zlals0(const int icompq, const int nl, const int nr, const int sqre,
            const int nrhs, c128* const restrict B, const int ldb,
            c128* const restrict BX, const int ldbx,
            const int* const restrict perm, const int givptr,
            const int* const restrict givcol, const int ldgcol,
            const f64* const restrict givnum, const int ldgnum,
            const f64* const restrict poles, const f64* const restrict difl,
            const f64* const restrict difr, const f64* const restrict Z,
            const int k, const f64 c, const f64 s,
            f64* const restrict rwork, int* info)
{
    int i, j, jcol, jrow, m, n, nlp1;
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
        xerbla("ZLALS0", -(*info));
        return;
    }

    m = n + sqre;
    nlp1 = nl + 1;

    if (icompq == 0) {

        /* Step (1L): apply back the Givens rotations performed. */

        for (i = 1; i <= givptr; i++) {
            cblas_zdrot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                        &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                        givnum[i - 1 + 1 * ldgnum], givnum[i - 1 + 0 * ldgnum]);
        }

        /* Step (2L): permute rows of B. */

        cblas_zcopy(nrhs, &B[nlp1 - 1], ldb, &BX[0], ldbx);
        for (i = 2; i <= n; i++) {
            cblas_zcopy(nrhs, &B[perm[i - 1]], ldb, &BX[i - 1], ldbx);
        }

        /* Step (3L): apply the inverse of the left singular vector
         * matrix to BX. */

        if (k == 1) {
            cblas_zcopy(nrhs, BX, ldbx, B, ldb);
            if (Z[0] < 0.0) {
                cblas_zdscal(nrhs, -1.0, B, ldb);
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
                    rwork[j - 1] = 0.0;
                } else {
                    rwork[j - 1] = -poles[j - 1 + 1 * ldgnum] * Z[j - 1] / diflj /
                                   (poles[j - 1 + 1 * ldgnum] + dj);
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[i - 1] == 0.0 || poles[i - 1 + 1 * ldgnum] == 0.0) {
                        rwork[i - 1] = 0.0;
                    } else {
                        rwork[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                       (dlamc3(poles[i - 1 + 1 * ldgnum], dsigj) - diflj) /
                                       (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[i - 1] == 0.0 || poles[i - 1 + 1 * ldgnum] == 0.0) {
                        rwork[i - 1] = 0.0;
                    } else {
                        rwork[i - 1] = poles[i - 1 + 1 * ldgnum] * Z[i - 1] /
                                       (dlamc3(poles[i - 1 + 1 * ldgnum], dsigjp) + difrj) /
                                       (poles[i - 1 + 1 * ldgnum] + dj);
                    }
                }
                rwork[0] = -1.0;
                temp = cblas_dnrm2(k, rwork, 1);

                /* Since B and BX are complex, the following call to DGEMV
                 * is performed in two steps (real and imaginary parts). */

                i = k + nrhs * 2;
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    for (jrow = 1; jrow <= k; jrow++) {
                        rwork[i] = creal(BX[(jrow - 1) + (jcol - 1) * ldbx]);
                        i++;
                    }
                }
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0,
                            &rwork[k + nrhs * 2], k, rwork, 1, 0.0, &rwork[k], 1);
                i = k + nrhs * 2;
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    for (jrow = 1; jrow <= k; jrow++) {
                        rwork[i] = cimag(BX[(jrow - 1) + (jcol - 1) * ldbx]);
                        i++;
                    }
                }
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0,
                            &rwork[k + nrhs * 2], k, rwork, 1, 0.0, &rwork[k + nrhs], 1);
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    B[(j - 1) + (jcol - 1) * ldb] = CMPLX(rwork[k + jcol - 1],
                                                           rwork[k + nrhs + jcol - 1]);
                }
                zlascl("G", 0, 0, temp, 1.0, 1, nrhs, &B[j - 1], ldb, info);
            }
        }

        /* Move the deflated rows of BX to B also. */

        if (k < (m > n ? m : n)) {
            zlacpy("A", n - k, nrhs, &BX[k], ldbx, &B[k], ldb);
        }
    } else {

        /* Step (1R): apply back the new right singular vector matrix to B. */

        if (k == 1) {
            cblas_zcopy(nrhs, B, ldb, BX, ldbx);
        } else {
            for (j = 1; j <= k; j++) {
                dsigj = poles[j - 1 + 1 * ldgnum];
                if (Z[j - 1] == 0.0) {
                    rwork[j - 1] = 0.0;
                } else {
                    rwork[j - 1] = -Z[j - 1] / difl[j - 1] /
                                   (dsigj + poles[j - 1 + 0 * ldgnum]) / difr[j - 1 + 1 * ldgnum];
                }
                for (i = 1; i <= j - 1; i++) {
                    if (Z[j - 1] == 0.0) {
                        rwork[i - 1] = 0.0;
                    } else {
                        rwork[i - 1] = Z[j - 1] /
                                       (dlamc3(dsigj, -poles[i + 1 * ldgnum]) - difr[i - 1 + 0 * ldgnum]) /
                                       (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }
                for (i = j + 1; i <= k; i++) {
                    if (Z[j - 1] == 0.0) {
                        rwork[i - 1] = 0.0;
                    } else {
                        rwork[i - 1] = Z[j - 1] /
                                       (dlamc3(dsigj, -poles[i - 1 + 1 * ldgnum]) - difl[i - 1]) /
                                       (dsigj + poles[i - 1 + 0 * ldgnum]) / difr[i - 1 + 1 * ldgnum];
                    }
                }

                /* Since B and BX are complex, the following call to DGEMV
                 * is performed in two steps (real and imaginary parts). */

                i = k + nrhs * 2;
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    for (jrow = 1; jrow <= k; jrow++) {
                        rwork[i] = creal(B[(jrow - 1) + (jcol - 1) * ldb]);
                        i++;
                    }
                }
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0,
                            &rwork[k + nrhs * 2], k, rwork, 1, 0.0, &rwork[k], 1);
                i = k + nrhs * 2;
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    for (jrow = 1; jrow <= k; jrow++) {
                        rwork[i] = cimag(B[(jrow - 1) + (jcol - 1) * ldb]);
                        i++;
                    }
                }
                cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, 1.0,
                            &rwork[k + nrhs * 2], k, rwork, 1, 0.0, &rwork[k + nrhs], 1);
                for (jcol = 1; jcol <= nrhs; jcol++) {
                    BX[(j - 1) + (jcol - 1) * ldbx] = CMPLX(rwork[k + jcol - 1],
                                                              rwork[k + nrhs + jcol - 1]);
                }
            }
        }

        /* Step (2R): if SQRE = 1, apply back the rotation that is
         * related to the right null space of the subproblem. */

        if (sqre == 1) {
            cblas_zcopy(nrhs, &B[m - 1], ldb, &BX[m - 1], ldbx);
            cblas_zdrot(nrhs, &BX[0], ldbx, &BX[m - 1], ldbx, c, s);
        }
        if (k < (m > n ? m : n)) {
            zlacpy("A", n - k, nrhs, &B[k], ldb, &BX[k], ldbx);
        }

        /* Step (3R): permute rows of B. */

        cblas_zcopy(nrhs, &BX[0], ldbx, &B[nlp1 - 1], ldb);
        if (sqre == 1) {
            cblas_zcopy(nrhs, &BX[m - 1], ldbx, &B[m - 1], ldb);
        }
        for (i = 2; i <= n; i++) {
            cblas_zcopy(nrhs, &BX[i - 1], ldbx, &B[perm[i - 1]], ldb);
        }

        /* Step (4R): apply back the Givens rotations performed. */

        for (i = givptr; i >= 1; i--) {
            cblas_zdrot(nrhs, &B[givcol[i - 1 + 1 * ldgcol]], ldb,
                        &B[givcol[i - 1 + 0 * ldgcol]], ldb,
                        givnum[i - 1 + 1 * ldgnum], -givnum[i - 1 + 0 * ldgnum]);
        }
    }
}
