/**
 * @file slasd7.c
 * @brief SLASD7 merges the two sets of singular values together into a single
 *        sorted set. Then it tries to deflate the size of the problem.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include "semicolon_cblas.h"

/**
 * SLASD7 merges the two sets of singular values together into a single
 * sorted set. Then it tries to deflate the size of the problem. There
 * are two ways in which deflation can occur: when two or more singular
 * values are close together or if there is a tiny entry in the `Z`
 * vector. For each such occurrence the order of the related
 * secular equation problem is reduced by one.
 *
 * SLASD7 is called from SLASD6.
 *
 * @param[in]     icompq  Specifies whether singular vectors are to be computed
 *                        in compact form, as follows:
 *                        `icompq=0`: Compute singular values only.
 *                        `icompq=1`: Compute singular vectors of upper bidiagonal
 *                        matrix in compact form.
 * @param[in]     nl      The row dimension of the upper block. `nl>=1`.
 * @param[in]     nr      The row dimension of the lower block. `nr>=1`.
 * @param[in]     sqre    `sqre=0`: the lower block is an `nr`-by-`nr` square matrix.
 *                        `sqre=1`: the lower block is an `nr`-by-`(nr+1)` rectangular
 *                        matrix.
 *                        The bidiagonal matrix has `n = nl+nr+1` rows and
 *                        `m = n+sqre >= n` columns.
 * @param[out]    k       Contains the dimension of the non-deflated matrix, this is
 *                        the order of the related secular equation. `1<=k<=n`.
 * @param[in,out] D       Array of dimension (`n`). On entry `D` contains the singular
 *                        values of the two submatrices to be combined. On exit `D`
 *                        contains the trailing (`n-k`) updated singular values (those
 *                        which were deflated) sorted into decreasing order.
 * @param[out]    Z       Array of dimension (`m`). On exit `Z` contains the updating
 *                        row vector in the secular equation.
 * @param[out]    ZW      Array of dimension (`m`). Workspace for `Z`.
 * @param[in,out] VF      Array of dimension (`m`). On entry, `VF[0:nl]` contains the
 *                        first components of all right singular vectors of the upper
 *                        block; and `VF[nl+1:m-1]` contains the first components of all
 *                        right singular vectors of the lower block. On exit, `VF`
 *                        contains the first components of all right singular vectors of
 *                        the bidiagonal matrix.
 * @param[out]    VFW     Array of dimension (`m`). Workspace for `VF`.
 * @param[in,out] VL      Array of dimension (`m`). On entry, `VL[0:nl]` contains the
 *                        last components of all right singular vectors of the upper
 *                        block; and `VL[nl+1:m-1]` contains the last components of all
 *                        right singular vectors of the lower block. On exit, `VL`
 *                        contains the last components of all right singular vectors of
 *                        the bidiagonal matrix.
 * @param[out]    VLW     Array of dimension (`m`). Workspace for `VL`.
 * @param[in]     alpha   Contains the diagonal element associated with the added row.
 * @param[in]     beta    Contains the off-diagonal element associated with the added
 *                        row.
 * @param[out]    DSIGMA  Array of dimension (`n`). Contains a copy of the diagonal
 *                        elements (`k-1` singular values and one zero) in the secular
 *                        equation.
 * @param[out]    IDX     Integer array of dimension (`n`). This will contain the
 *                        permutation used to sort the contents of `D` into ascending
 *                        order.
 * @param[out]    IDXP    Integer array of dimension (`n`). This will contain the
 *                        permutation used to place deflated values of `D` at the end of
 *                        the array. On output `IDXP[1:k-1]` points to the nondeflated
 *                        D-values and `IDXP[k:n-1]` points to the deflated singular
 *                        values.
 * @param[in]     IDXQ    Integer array of dimension (`n`). This contains the permutation
 *                        which separately sorts the two sub-problems in `D` into
 *                        ascending order. Note that entries in the first half of this
 *                        permutation must first be moved one position backward; and
 *                        entries in the second half must first have `nl+1` added to
 *                        their values.
 * @param[out]    PERM    Integer array of dimension (`n`). The permutations (from
 *                        deflation and sorting) to be applied to each singular block.
 *                        Not referenced if `icompq=0`.
 * @param[out]    givptr  The number of Givens rotations which took place in this
 *                        subproblem. Not referenced if `icompq=0`.
 * @param[out]    GIVCOL  Integer array of dimension (`ldgcol`, 2). Each pair of numbers
 *                        indicates a pair of columns to take place in a Givens rotation.
 *                        Not referenced if `icompq=0`.
 * @param[in]     ldgcol  The leading dimension of `GIVCOL`, must be at least `n`.
 * @param[out]    GIVNUM  Array of dimension (`ldgnum`, 2). Each number indicates the C
 *                        or S value to be used in the corresponding Givens rotation.
 *                        Not referenced if `icompq=0`.
 * @param[in]     ldgnum  The leading dimension of `GIVNUM`, must be at least `n`.
 * @param[out]    c       `c` contains garbage if `sqre=0` and the C-value of a Givens
 *                        rotation related to the right null space if `sqre=1`.
 * @param[out]    s       `s` contains garbage if `sqre=0` and the S-value of a Givens
 *                        rotation related to the right null space if `sqre=1`.
 * @param[out]    info    `info=0`: successful exit.
 *                        `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *
 * @par Contributors:
 * @rst
 * Ming Gu and Huan Ren, Computer Science Division, University of
 * California at Berkeley, USA
 * @endrst
 */
void slasd7(const INT icompq, const INT nl, const INT nr, const INT sqre,
            INT* k, f32* restrict D, f32* restrict Z,
            f32* restrict ZW, f32* restrict VF,
            f32* restrict VFW, f32* restrict VL,
            f32* restrict VLW, const f32 alpha, const f32 beta,
            f32* restrict DSIGMA, INT* restrict IDX,
            INT* restrict IDXP, INT* restrict IDXQ,
            INT* restrict PERM, INT* givptr,
            INT* restrict GIVCOL, const INT ldgcol,
            f32* restrict GIVNUM, const INT ldgnum,
            f32* c, f32* s, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 EIGHT = 8.0f;

    INT i, idxi, idxj, idxjp, j, jp, jprev = 0, k2, m, n, nlp1, nlp2;
    f32 eps, hlftol, tau, tol, z1;

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
        xerbla("SLASD7", -(*info));
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
        DSIGMA[i] = D[IDXQ[i]];
        ZW[i] = Z[IDXQ[i]];
        VFW[i] = VF[IDXQ[i]];
        VLW[i] = VL[IDXQ[i]];
    }

    slamrg(nl, nr, &DSIGMA[1], 1, 1, &IDX[1]);

    for (i = 1; i < n; i++) {
        idxi = 1 + IDX[i];
        D[i] = DSIGMA[idxi];
        Z[i] = ZW[idxi];
        VF[i] = VFW[idxi];
        VL[i] = VLW[idxi];
    }

    eps = slamch("Epsilon");
    tol = fabsf(alpha) > fabsf(beta) ? fabsf(alpha) : fabsf(beta);
    tol = EIGHT * EIGHT * eps * (fabsf(D[n - 1]) > tol ? fabsf(D[n - 1]) : tol);

    *k = 1;
    k2 = n;

    for (j = 1; j < n; j++) {
        if (fabsf(Z[j]) <= tol) {
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
    if (fabsf(Z[j]) <= tol) {
        k2 = k2 - 1;
        IDXP[k2] = j;
    } else {
        if (fabsf(D[j] - D[jprev]) <= tol) {
            *s = Z[jprev];
            *c = Z[j];

            tau = slapy2(*c, *s);
            Z[j] = tau;
            Z[jprev] = ZERO;
            *c = *c / tau;
            *s = -(*s) / tau;

            if (icompq == 1) {
                (*givptr)++;
                idxjp = IDXQ[IDX[jprev] + 1];
                idxj = IDXQ[IDX[j] + 1];
                if (idxjp <= nl) {
                    idxjp--;
                }
                if (idxj <= nl) {
                    idxj--;
                }
                GIVCOL[*givptr - 1 + 1 * ldgcol] = idxjp;
                GIVCOL[*givptr - 1 + 0 * ldgcol] = idxj;
                GIVNUM[*givptr - 1 + 1 * ldgnum] = *c;
                GIVNUM[*givptr - 1 + 0 * ldgnum] = *s;
            }
            cblas_srot(1, &VF[jprev], 1, &VF[j], 1, *c, *s);
            cblas_srot(1, &VL[jprev], 1, &VL[j], 1, *c, *s);
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
            if (PERM[j] <= nl) {
                PERM[j]--;
            }
        }
    }

    cblas_scopy(n - *k, &DSIGMA[*k], 1, &D[*k], 1);

    DSIGMA[0] = ZERO;
    hlftol = tol / TWO;
    if (fabsf(DSIGMA[1]) <= hlftol) {
        DSIGMA[1] = hlftol;
    }
    if (m > n) {
        Z[0] = slapy2(z1, Z[m - 1]);
        if (Z[0] <= tol) {
            *c = ONE;
            *s = ZERO;
            Z[0] = tol;
        } else {
            *c = z1 / Z[0];
            *s = -Z[m - 1] / Z[0];
        }
        cblas_srot(1, &VF[m - 1], 1, &VF[0], 1, *c, *s);
        cblas_srot(1, &VL[m - 1], 1, &VL[0], 1, *c, *s);
    } else {
        if (fabsf(z1) <= tol) {
            Z[0] = tol;
        } else {
            Z[0] = z1;
        }
    }

    cblas_scopy(*k - 1, &ZW[1], 1, &Z[1], 1);
    cblas_scopy(n - 1, &VFW[1], 1, &VF[1], 1);
    cblas_scopy(n - 1, &VLW[1], 1, &VL[1], 1);
}
