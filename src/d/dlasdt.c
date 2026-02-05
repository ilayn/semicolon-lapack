/**
 * @file dlasdt.c
 * @brief DLASDT creates a tree of subproblems for bidiagonal divide and conquer.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLASDT creates a tree of subproblems for bidiagonal divide and conquer.
 *
 * @param[in]     n      The number of diagonal elements of the bidiagonal matrix.
 * @param[out]    lvl    The number of levels on the computation tree.
 * @param[out]    nd     The number of nodes on the tree.
 * @param[out]    inode  Integer array, dimension (n). On exit, centers of subproblems.
 * @param[out]    ndiml  Integer array, dimension (n). On exit, row dimensions of left children.
 * @param[out]    ndimr  Integer array, dimension (n). On exit, row dimensions of right children.
 * @param[in]     msub   The maximum row dimension each subproblem at the bottom of the tree can be of.
 */
void dlasdt(const int n, int* lvl, int* nd,
            int* const restrict inode, int* const restrict ndiml, int* const restrict ndimr,
            const int msub)
{
    int maxn = (n > 1) ? n : 1;
    double temp = log((double)maxn / (double)(msub + 1)) / log(2.0);
    *lvl = (int)temp + 1;

    int i = n / 2;
    inode[0] = i;
    ndiml[0] = i;
    ndimr[0] = n - i - 1;

    int il = -1;
    int ir = 0;
    int llst = 1;

    for (int nlvl = 1; nlvl <= *lvl - 1; nlvl++) {
        for (int j = 0; j < llst; j++) {
            il += 2;
            ir += 2;
            int ncrnt = llst - 1 + j;
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
