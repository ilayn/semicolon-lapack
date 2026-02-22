/**
 * @file slapmr.c
 * @brief SLAPMR rearranges rows of a matrix as specified by a permutation vector.
 */

#include "semicolon_lapack_single.h"

/**
 * SLAPMR rearranges the rows of the M by N matrix X as specified
 * by the permutation K(1),K(2),...,K(M) of the integers 1,...,M.
 * If FORWRD = 1,  forward permutation:
 *
 *      X(K(I),*) is moved X(I,*) for I = 1,2,...,M.
 *
 * If FORWRD = 0, backward permutation:
 *
 *      X(I,*) is moved to X(K(I),*) for I = 1,2,...,M.
 *
 * @param[in] forwrd
 *          = 1: forward permutation
 *          = 0: backward permutation
 *
 * @param[in] m
 *          The number of rows of the matrix X. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix X. n >= 0.
 *
 * @param[in,out] X
 *          Double precision array, dimension (ldx, n).
 *          On entry, the M by N matrix X.
 *          On exit, X contains the permuted matrix X.
 *
 * @param[in] ldx
 *          The leading dimension of the array X, ldx >= max(1, m).
 *
 * @param[in,out] K
 *          Integer array, dimension (m).
 *          On entry, K contains the permutation vector (0-based indices).
 *          K is used as internal workspace, but reset to its original
 *          value on output.
 */
void slapmr(
    const INT forwrd,
    const INT m,
    const INT n,
    f32* restrict X,
    const INT ldx,
    INT* restrict K)
{
    INT i, in, j, jj;
    f32 temp;

    if (m <= 1)
        return;

    for (i = 0; i < m; i++) {
        K[i] = -(K[i] + 1);
    }

    if (forwrd) {

        for (i = 0; i < m; i++) {

            if (K[i] >= 0)
                continue;

            j = i;
            K[j] = -(K[j] + 1);
            in = K[j];

            while (K[in] < 0) {

                for (jj = 0; jj < n; jj++) {
                    temp = X[j + jj * ldx];
                    X[j + jj * ldx] = X[in + jj * ldx];
                    X[in + jj * ldx] = temp;
                }

                K[in] = -(K[in] + 1);
                j = in;
                in = K[in];
            }

        }

    } else {

        for (i = 0; i < m; i++) {

            if (K[i] >= 0)
                continue;

            K[i] = -(K[i] + 1);
            j = K[i];

            while (j != i) {

                for (jj = 0; jj < n; jj++) {
                    temp = X[i + jj * ldx];
                    X[i + jj * ldx] = X[j + jj * ldx];
                    X[j + jj * ldx] = temp;
                }

                K[j] = -(K[j] + 1);
                j = K[j];
            }

        }

    }
}
