/**
 * @file dlalsa.c
 * @brief DLALSA computes the SVD of the coefficient matrix in compact form.
 *        Used by dgelsd.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DLALSA is an intermediate step in solving the least squares problem
 * by computing the SVD of the coefficient matrix in compact form (The
 * singular vectors are computed as products of simple orthogonal
 * matrices.).
 *
 * If icompq = 0, DLALSA applies the inverse of the left singular vector
 * matrix of an upper bidiagonal matrix to the right hand side; and if
 * icompq = 1, DLALSA applies the right singular vector matrix to the
 * right hand side. The singular vector matrices were generated in
 * compact form by DLALSA.
 *
 * @param[in]     icompq   Specifies whether left or right singular vector matrix:
 *                         = 0: Left singular vector matrix
 *                         = 1: Right singular vector matrix
 * @param[in]     smlsiz   The maximum size of subproblems at the bottom of the tree.
 * @param[in]     n        The row and column dimensions of the upper bidiagonal matrix.
 * @param[in]     nrhs     The number of columns of B and BX. nrhs >= 1.
 * @param[in,out] B        Array of dimension (ldb, nrhs).
 *                         On input, contains the right hand sides.
 *                         On output, contains the solution X.
 * @param[in]     ldb      The leading dimension of B. ldb >= max(1, n).
 * @param[out]    BX       Array of dimension (ldbx, nrhs). Workspace.
 * @param[in]     ldbx     The leading dimension of BX.
 * @param[in]     U        Array of dimension (ldu, smlsiz).
 *                         Left singular vector matrices of subproblems at bottom level.
 * @param[in]     ldu      The leading dimension of U, VT, DIFL, DIFR, POLES, GIVNUM, Z.
 *                         ldu >= n.
 * @param[in]     VT       Array of dimension (ldu, smlsiz+1).
 *                         VT^T contains right singular vector matrices at bottom level.
 * @param[in]     K        Integer array of dimension n.
 * @param[in]     difl     Array of dimension (ldu, nlvl).
 * @param[in]     difr     Array of dimension (ldu, 2*nlvl).
 * @param[in]     Z        Array of dimension (ldu, nlvl).
 * @param[in]     poles    Array of dimension (ldu, 2*nlvl).
 * @param[in]     givptr   Integer array of dimension n.
 * @param[in]     givcol   Integer array of dimension (ldgcol, 2*nlvl).
 * @param[in]     ldgcol   Leading dimension of givcol and perm. ldgcol >= n.
 * @param[in]     perm     Integer array of dimension (ldgcol, nlvl).
 * @param[in]     givnum   Array of dimension (ldu, 2*nlvl).
 * @param[in]     C        Array of dimension n.
 * @param[in]     S        Array of dimension n.
 * @param[out]    work     Array of dimension n.
 * @param[out]    iwork    Integer array of dimension 3*n.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had illegal value.
 */
void dlalsa(const int icompq, const int smlsiz, const int n, const int nrhs,
            f64* const restrict B, const int ldb,
            f64* const restrict BX, const int ldbx,
            const f64* const restrict U, const int ldu,
            const f64* const restrict VT, const int* const restrict K,
            const f64* const restrict difl, const f64* const restrict difr,
            const f64* const restrict Z, const f64* const restrict poles,
            const int* const restrict givptr, const int* const restrict givcol,
            const int ldgcol, const int* const restrict perm,
            const f64* const restrict givnum,
            const f64* const restrict C, const f64* const restrict S,
            f64* const restrict work, int* const restrict iwork, int* info)
{
    int i, ic, im1, inode, j, lf, ll, lvl, lvl2;
    int nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl;
    int nr, nrf, nrp1, sqre;

    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (smlsiz < 3) {
        *info = -2;
    } else if (n < smlsiz) {
        *info = -3;
    } else if (nrhs < 1) {
        *info = -4;
    } else if (ldb < n) {
        *info = -6;
    } else if (ldbx < n) {
        *info = -8;
    } else if (ldu < n) {
        *info = -10;
    } else if (ldgcol < n) {
        *info = -19;
    }
    if (*info != 0) {
        xerbla("DLALSA", -(*info));
        return;
    }

    inode = 0;
    ndiml = n;
    ndimr = ndiml + n;

    dlasdt(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], smlsiz);

    if (icompq == 1) {
        goto L50;
    }

    ndb1 = (nd + 1) / 2;
    for (i = ndb1; i <= nd; i++) {
        ic = iwork[inode + i - 1];
        nl = iwork[ndiml + i - 1];
        nr = iwork[ndimr + i - 1];
        nlf = ic - nl;
        nrf = ic + 1;

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nl, nrhs, nl, 1.0, &U[nlf], ldu, &B[nlf], ldb,
                    0.0, &BX[nlf], ldbx);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nr, nrhs, nr, 1.0, &U[nrf], ldu, &B[nrf], ldb,
                    0.0, &BX[nrf], ldbx);
    }

    for (i = 1; i <= nd; i++) {
        ic = iwork[inode + i - 1];
        cblas_dcopy(nrhs, &B[ic], ldb, &BX[ic], ldbx);
    }

    j = 1 << nlvl;
    sqre = 0;

    for (lvl = nlvl; lvl >= 1; lvl--) {
        lvl2 = 2 * lvl - 1;

        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }
        for (i = lf; i <= ll; i++) {
            im1 = i - 1;
            ic = iwork[inode + im1];
            nl = iwork[ndiml + im1];
            nr = iwork[ndimr + im1];
            nlf = ic - nl;
            nrf = ic + 1;
            j = j - 1;
            dlals0(icompq, nl, nr, sqre, nrhs,
                   &BX[nlf], ldbx, &B[nlf], ldb,
                   &perm[nlf + (lvl - 1) * ldgcol],
                   givptr[j - 1],
                   &givcol[nlf + (lvl2 - 1) * ldgcol], ldgcol,
                   &givnum[nlf + (lvl2 - 1) * ldu], ldu,
                   &poles[nlf + (lvl2 - 1) * ldu],
                   &difl[nlf + (lvl - 1) * ldu],
                   &difr[nlf + (lvl2 - 1) * ldu],
                   &Z[nlf + (lvl - 1) * ldu],
                   K[j - 1], C[j - 1], S[j - 1], work, info);
        }
    }
    goto L90;

L50:
    j = 0;
    for (lvl = 1; lvl <= nlvl; lvl++) {
        lvl2 = 2 * lvl - 1;

        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }
        for (i = ll; i >= lf; i--) {
            im1 = i - 1;
            ic = iwork[inode + im1];
            nl = iwork[ndiml + im1];
            nr = iwork[ndimr + im1];
            nlf = ic - nl;
            nrf = ic + 1;
            if (i == ll) {
                sqre = 0;
            } else {
                sqre = 1;
            }
            j = j + 1;
            dlals0(icompq, nl, nr, sqre, nrhs,
                   &B[nlf], ldb, &BX[nlf], ldbx,
                   &perm[nlf + (lvl - 1) * ldgcol],
                   givptr[j - 1],
                   &givcol[nlf + (lvl2 - 1) * ldgcol], ldgcol,
                   &givnum[nlf + (lvl2 - 1) * ldu], ldu,
                   &poles[nlf + (lvl2 - 1) * ldu],
                   &difl[nlf + (lvl - 1) * ldu],
                   &difr[nlf + (lvl2 - 1) * ldu],
                   &Z[nlf + (lvl - 1) * ldu],
                   K[j - 1], C[j - 1], S[j - 1], work, info);
        }
    }

    ndb1 = (nd + 1) / 2;
    for (i = ndb1; i <= nd; i++) {
        ic = iwork[inode + i - 1];
        nl = iwork[ndiml + i - 1];
        nr = iwork[ndimr + i - 1];
        nlp1 = nl + 1;
        if (i == nd) {
            nrp1 = nr;
        } else {
            nrp1 = nr + 1;
        }
        nlf = ic - nl;
        nrf = ic + 1;

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nlp1, nrhs, nlp1, 1.0, &VT[nlf], ldu, &B[nlf], ldb,
                    0.0, &BX[nlf], ldbx);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nrp1, nrhs, nrp1, 1.0, &VT[nrf], ldu, &B[nrf], ldb,
                    0.0, &BX[nrf], ldbx);
    }

L90:
    return;
}
