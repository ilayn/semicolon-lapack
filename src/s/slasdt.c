/**
 * @file slasdt.c
 * @brief SLASDT creates a tree of subproblems for bidiagonal divide and conquer.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include <math.h>

/**
 * SLASDT creates a tree of subproblems for bidiagonal divide and conquer.
 *
 * @param[in]     n      The number of diagonal elements of the bidiagonal matrix.
 * @param[out]    lvl    The number of levels on the computation tree.
 * @param[out]    nd     The number of nodes on the tree.
 * @param[out]    inode  Integer array, dimension (n). On exit, centers of subproblems.
 * @param[out]    ndiml  Integer array, dimension (n). On exit, row dimensions of left children.
 * @param[out]    ndimr  Integer array, dimension (n). On exit, row dimensions of right children.
 * @param[in]     msub   The maximum row dimension each subproblem at the bottom of the tree can be of.
 */
void slasdt(const INT n, INT* lvl, INT* nd,
            INT* restrict inode, INT* restrict ndiml, INT* restrict ndimr,
            const INT msub)
{
    INT maxn = (n > 1) ? n : 1;
    f32 temp = logf((f32)maxn / (f32)(msub + 1)) / logf(2.0f);
    *lvl = (INT)temp + 1;

    INT i = n / 2;
    inode[0] = i;
    ndiml[0] = i;
    ndimr[0] = n - i - 1;

    INT il = -1;
    INT ir = 0;
    INT llst = 1;

    for (INT nlvl = 1; nlvl <= *lvl - 1; nlvl++) {
        for (INT j = 0; j < llst; j++) {
            il += 2;
            ir += 2;
            INT ncrnt = llst - 1 + j;
            ndiml[il] = ndiml[ncrnt] / 2;
            ndimr[il] = ndiml[ncrnt] - ndiml[il] - 1;
            inode[il] = inode[ncrnt] - ndimr[il] - 1;
            ndiml[ir] = ndimr[ncrnt] / 2;
            ndimr[ir] = ndimr[ncrnt] - ndiml[ir] - 1;
            inode[ir] = inode[ncrnt] + ndiml[ir] + 1;
        }
        llst *= 2;
    }
    *nd = llst * 2 - 1;
}
