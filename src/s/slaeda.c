/**
 * @file slaeda.c
 * @brief SLAEDA computes the Z vector determining the rank-one modification
 *        of the diagonal matrix. Used when the original matrix is dense.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAEDA computes the Z vector corresponding to the merge step in the
 * CURLVLth step of the merge process with TLVLS steps for the CURPBMth
 * problem.
 *
 * @param[in]     n       The dimension of the symmetric tridiagonal matrix. n >= 0.
 * @param[in]     tlvls   The total number of merging levels in the overall divide and
 *                        conquer tree.
 * @param[in]     curlvl  The current level in the overall merge routine,
 *                        0 <= curlvl <= tlvls.
 * @param[in]     curpbm  The current problem in the current level in the overall
 *                        merge routine (counting from upper left to lower right).
 * @param[in]     prmptr  Integer array, dimension (N lg N).
 *                        Contains a list of pointers which indicate where in PERM a
 *                        level's permutation is stored. prmptr[i+1] - prmptr[i]
 *                        indicates the size of the permutation and incidentally the
 *                        size of the full, non-deflated problem.
 * @param[in]     perm    Integer array, dimension (N lg N).
 *                        Contains the permutations (from deflation and sorting) to be
 *                        applied to each eigenblock.
 * @param[in]     givptr  Integer array, dimension (N lg N).
 *                        Contains a list of pointers which indicate where in GIVCOL a
 *                        level's Givens rotations are stored. givptr[i+1] - givptr[i]
 *                        indicates the number of Givens rotations.
 * @param[in]     givcol  Integer array, dimension (2, N lg N).
 *                        Each pair of numbers indicates a pair of columns to take place
 *                        in a Givens rotation.
 * @param[in]     givnum  Double precision array, dimension (2, N lg N).
 *                        Each number indicates the S value to be used in the
 *                        corresponding Givens rotation.
 * @param[in]     Q       Double precision array, dimension (N**2).
 *                        Contains the square eigenblocks from previous levels, the
 *                        starting positions for blocks are given by QPTR.
 * @param[in]     qptr    Integer array, dimension (N+2).
 *                        Contains a list of pointers which indicate where in Q an
 *                        eigenblock is stored. sqrt(qptr[i+1] - qptr[i]) indicates
 *                        the size of the block.
 * @param[out]    Z       Double precision array, dimension (N).
 *                        On output this vector contains the updating vector (the last
 *                        row of the first sub-eigenvector matrix and the first row of
 *                        the second sub-eigenvector matrix).
 * @param[out]    ztemp   Double precision array, dimension (N).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void slaeda(const int n, const int tlvls, const int curlvl, const int curpbm,
            int* prmptr, int* perm, int* givptr, int* givcol,
            f32* givnum, f32* Q, int* qptr, f32* Z,
            f32* ztemp, int* info)
{
    int bsiz1, bsiz2, curr, i, k, mid, psiz1, psiz2, ptr, zptr1;

    /* Test the input parameters. */
    *info = 0;

    if (n < 0) {
        *info = -1;
    }
    if (*info != 0) {
        xerbla("SLAEDA", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Determine location of first number in second half. */
    mid = n / 2;

    /* Gather last/first rows of appropriate eigenblocks into center of Z */
    ptr = 0;

    /* Determine location of lowest level subproblem in the full storage
     * scheme */
    curr = ptr + curpbm * (1 << curlvl) + (1 << (curlvl - 1)) - 1;

    /* Determine size of these matrices. We add 0.5 to the value of
     * the sqrt in case the machine underestimates one of these square
     * roots. */
    bsiz1 = (int)(0.5f + sqrtf((f32)(qptr[curr + 1] - qptr[curr])));
    bsiz2 = (int)(0.5f + sqrtf((f32)(qptr[curr + 2] - qptr[curr + 1])));

    for (k = 0; k < mid - bsiz1; k++) {
        Z[k] = 0.0f;
    }
    cblas_scopy(bsiz1, &Q[qptr[curr] + bsiz1 - 1], bsiz1,
                &Z[mid - bsiz1], 1);
    cblas_scopy(bsiz2, &Q[qptr[curr + 1]], bsiz2, &Z[mid], 1);
    for (k = mid + bsiz2; k < n; k++) {
        Z[k] = 0.0f;
    }

    /* Loop through remaining levels 1 -> CURLVL applying the Givens
     * rotations and permutation and then multiplying the center matrices
     * against the current Z. */
    ptr = 1 << tlvls;
    for (k = 1; k <= curlvl - 1; k++) {
        curr = ptr + curpbm * (1 << (curlvl - k)) + (1 << (curlvl - k - 1)) - 1;
        psiz1 = prmptr[curr + 1] - prmptr[curr];
        zptr1 = mid - psiz1;

        /* Apply Givens at CURR and CURR+1 */
        for (i = givptr[curr]; i < givptr[curr + 1]; i++) {
            cblas_srot(1, &Z[zptr1 + givcol[2 * i]], 1,
                       &Z[zptr1 + givcol[2 * i + 1]], 1,
                       givnum[2 * i], givnum[2 * i + 1]);
        }
        for (i = givptr[curr + 1]; i < givptr[curr + 2]; i++) {
            cblas_srot(1, &Z[mid + givcol[2 * i]], 1,
                       &Z[mid + givcol[2 * i + 1]], 1,
                       givnum[2 * i], givnum[2 * i + 1]);
        }
        psiz1 = prmptr[curr + 1] - prmptr[curr];
        psiz2 = prmptr[curr + 2] - prmptr[curr + 1];
        for (i = 0; i < psiz1; i++) {
            ztemp[i] = Z[zptr1 + perm[prmptr[curr] + i]];
        }
        for (i = 0; i < psiz2; i++) {
            ztemp[psiz1 + i] = Z[mid + perm[prmptr[curr + 1] + i]];
        }

        /* Multiply Blocks at CURR and CURR+1 */

        /* Determine size of these matrices. We add 0.5 to the value of
         * the sqrt in case the machine underestimates one of these
         * square roots. */
        bsiz1 = (int)(0.5f + sqrtf((f32)(qptr[curr + 1] - qptr[curr])));
        bsiz2 = (int)(0.5f + sqrtf((f32)(qptr[curr + 2] - qptr[curr + 1])));
        if (bsiz1 > 0) {
            cblas_sgemv(CblasColMajor, CblasTrans, bsiz1, bsiz1, 1.0f,
                        &Q[qptr[curr]], bsiz1, &ztemp[0], 1, 0.0f,
                        &Z[zptr1], 1);
        }
        cblas_scopy(psiz1 - bsiz1, &ztemp[bsiz1], 1,
                    &Z[zptr1 + bsiz1], 1);
        if (bsiz2 > 0) {
            cblas_sgemv(CblasColMajor, CblasTrans, bsiz2, bsiz2, 1.0f,
                        &Q[qptr[curr + 1]], bsiz2, &ztemp[psiz1], 1, 0.0f,
                        &Z[mid], 1);
        }
        cblas_scopy(psiz2 - bsiz2, &ztemp[psiz1 + bsiz2], 1,
                    &Z[mid + bsiz2], 1);

        ptr = ptr + (1 << (tlvls - k));
    }
}
