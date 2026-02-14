/**
 * @file dlasd6.c
 * @brief DLASD6 computes the SVD of an updated upper bidiagonal matrix
 *        obtained by merging two smaller ones by appending a row.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/**
 * DLASD6 computes the SVD of an updated upper bidiagonal matrix B
 * obtained by merging two smaller ones by appending a row. This
 * routine is used only for the problem which requires all singular
 * values and optionally singular vector matrices in factored form.
 *
 * DLASD6 is called from DLASDA.
 *
 * @param[in]     icompq  = 0: Compute singular values only.
 *                         = 1: Compute singular vectors in factored form.
 * @param[in]     nl      Row dimension of upper block. nl >= 1.
 * @param[in]     nr      Row dimension of lower block. nr >= 1.
 * @param[in]     sqre    = 0: lower block is nr-by-nr square.
 *                         = 1: lower block is nr-by-(nr+1) rectangular.
 * @param[in,out] D       Array of dimension n. Singular values.
 * @param[in,out] VF      Array of dimension m. First components of right SVs.
 * @param[in,out] VL      Array of dimension m. Last components of right SVs.
 * @param[in,out] alpha   Diagonal element of added row.
 * @param[in,out] beta    Off-diagonal element of added row.
 * @param[in,out] IDXQ    Integer array of dimension n. Sorting permutation.
 * @param[out]    PERM    Integer array of dimension n. Permutation output.
 * @param[out]    givptr  Number of Givens rotations.
 * @param[out]    GIVCOL  Integer array (ldgcol, 2). Givens columns.
 * @param[in]     ldgcol  Leading dimension of GIVCOL. ldgcol >= n.
 * @param[out]    GIVNUM  Double array (ldgnum, 2). Givens values.
 * @param[in]     ldgnum  Leading dimension of GIVNUM and POLES. ldgnum >= n.
 * @param[out]    POLES   Double array (ldgnum, 2). New and old singular values.
 * @param[out]    DIFL    Double array of dimension n. Distances.
 * @param[out]    DIFR    Double array. Distances and normalizing factors.
 * @param[out]    Z       Double array of dimension m. Deflation-adjusted row.
 * @param[out]    k       Dimension of non-deflated matrix. 1 <= k <= n.
 * @param[out]    c       C-value of Givens rotation.
 * @param[out]    s       S-value of Givens rotation.
 * @param[out]    work    Double array of dimension 4*m.
 * @param[out]    IWORK   Integer array of dimension 3*n.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dlasd6(const int icompq, const int nl, const int nr, const int sqre,
            f64* const restrict D, f64* const restrict VF,
            f64* const restrict VL, f64* alpha, f64* beta,
            int* const restrict IDXQ, int* const restrict PERM,
            int* givptr, int* const restrict GIVCOL, const int ldgcol,
            f64* const restrict GIVNUM, const int ldgnum,
            f64* const restrict POLES, f64* const restrict DIFL,
            f64* const restrict DIFR, f64* const restrict Z,
            int* k, f64* c, f64* s,
            f64* const restrict work, int* const restrict IWORK, int* info)
{
    int i, idx, idxc, idxp, isigma, ivfw, ivlw, iw, m, n, n1, n2;
    f64 orgnrm;

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
        *info = -14;
    } else if (ldgnum < n) {
        *info = -16;
    }
    if (*info != 0) {
        xerbla("DLASD6", -(*info));
        return;
    }

    isigma = 0;
    iw = isigma + n;
    ivfw = iw + m;
    ivlw = ivfw + m;

    idx = 0;
    idxc = idx + n;
    idxp = idxc + n;

    orgnrm = fabs(*alpha) > fabs(*beta) ? fabs(*alpha) : fabs(*beta);
    D[nl] = 0.0;
    for (i = 0; i < n; i++) {
        if (fabs(D[i]) > orgnrm) {
            orgnrm = fabs(D[i]);
        }
    }
    dlascl("G", 0, 0, orgnrm, 1.0, n, 1, D, n, info);
    *alpha = *alpha / orgnrm;
    *beta = *beta / orgnrm;

    dlasd7(icompq, nl, nr, sqre, k, D, Z, &work[iw], VF,
           &work[ivfw], VL, &work[ivlw], *alpha, *beta,
           &work[isigma], &IWORK[idx], &IWORK[idxp], IDXQ,
           PERM, givptr, GIVCOL, ldgcol, GIVNUM, ldgnum, c, s, info);

    dlasd8(icompq, *k, D, Z, VF, VL, DIFL, DIFR, ldgnum,
           &work[isigma], &work[iw], info);

    if (*info != 0) {
        return;
    }

    if (icompq == 1) {
        cblas_dcopy(*k, D, 1, &POLES[0 + 0 * ldgnum], 1);
        cblas_dcopy(*k, &work[isigma], 1, &POLES[0 + 1 * ldgnum], 1);
    }

    dlascl("G", 0, 0, 1.0, orgnrm, n, 1, D, n, info);

    n1 = *k;
    n2 = n - *k;
    dlamrg(n1, n2, D, 1, -1, IDXQ);
}
