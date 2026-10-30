/**
 * @file dlapmt.c
 * @brief DLAPMT performs a forward or backward permutation of the columns of a matrix.
 */

#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"

/**
 * DLAPMT rearranges the columns of the M by N matrix X as specified
 * by the permutation K(1),K(2),...,K(N) of the integers 1,...,N.
 *
 * If FORWRD = 1 (true), forward permutation:
 *      X(*,K(J)) is moved to X(*,J) for J = 1,2,...,N.
 *
 * If FORWRD = 0 (false), backward permutation:
 *      X(*,J) is moved to X(*,K(J)) for J = 1,2,...,N.
 *
 * @param[in]     forwrd  = 1: forward permutation. = 0: backward permutation.
 * @param[in]     m       Number of rows of X. m >= 0.
 * @param[in]     n       Number of columns of X. n >= 0.
 * @param[in,out] X       Array (ldx, n). On exit, the permuted matrix.
 * @param[in]     ldx     Leading dimension of X. ldx >= max(1,m).
 * @param[in,out] K       Array (n). Permutation indices (0-based in C).
 *                        On exit, modified internally and then restored.
 */
void dlapmt(const INT forwrd, const INT m, const INT n,
            f64* restrict X, const INT ldx,
            INT* restrict K)
{
    INT i, j, in;

    if (n <= 1) return;

    for (i = 0; i < n; i++) {
        K[i] = -K[i] - 1;
    }

    if (forwrd) {
        /* Forward permutation */
        for (i = 0; i < n; i++) {
            if (K[i] >= 0) {
                continue;
            }

            j = i;
            K[j] = -K[j] - 1;
            in = K[j];

            while (K[in] < 0) {
                cblas_dswap(m, &X[j * ldx], 1, &X[in * ldx], 1);

                K[in] = -K[in] - 1;
                j = in;
                in = K[in];
            }
        }
    } else {
        /* Backward permutation */
        for (i = 0; i < n; i++) {
            if (K[i] >= 0) {
                continue;
            }

            K[i] = -K[i] - 1;
            j = K[i];

            while (j != i) {
                cblas_dswap(m, &X[i * ldx], 1, &X[j * ldx], 1);

                K[j] = -K[j] - 1;
                j = K[j];
            }
        }
    }
}
