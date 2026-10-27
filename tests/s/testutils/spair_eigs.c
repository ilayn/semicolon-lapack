/**
 * @file spair_eigs.c
 * @brief SPAIR_EIGS pairs two lists of the same eigenvalues by closest match.
 */

#include <math.h>
#include <stdlib.h>
#include "verify.h"

/**
 * SPAIR_EIGS pairs the eigenvalues (wr1(k), wi1(k)) one-to-one with
 * (wr(j), wi(j)), taking the closest remaining pair first, with distance
 * |wr(j) - wr1(k)| + |wi(j) - wi1(k)|. Two runs that compute the same
 * eigenvalues can return them in a different order; comparing them through
 * this pairing instead of by index tells a reordering apart from an error.
 *
 * @param[in]  n     The number of eigenvalues in each list. n >= 0.
 * @param[in]  wr    Real parts of the first list, dimension (n).
 * @param[in]  wi    Imaginary parts of the first list, dimension (n).
 * @param[in]  wr1   Real parts of the second list, dimension (n).
 * @param[in]  wi1   Imaginary parts of the second list, dimension (n).
 * @param[out] perm  Integer array, dimension (n). Eigenvalue j of the first
 *                   list is paired with eigenvalue perm[j] of the second.
 *                   Identical lists give perm[j] = j.
 */
void spair_eigs(const INT n, const f32* wr, const f32* wi,
                const f32* wr1, const f32* wi1, INT* perm)
{
    INT* used = malloc(n * sizeof(INT));
    for (INT j = 0; j < n; j++) {
        perm[j] = -1;
        used[j] = 0;
    }
    for (INT step = 0; step < n; step++) {
        INT bj = -1, bk = -1;
        f32 best = 0.0f;
        for (INT j = 0; j < n; j++) {
            if (perm[j] >= 0) continue;
            for (INT k = 0; k < n; k++) {
                if (used[k]) continue;
                f32 d = fabsf(wr[j] - wr1[k]) + fabsf(wi[j] - wi1[k]);
                if (bj < 0 || d < best) {
                    best = d;
                    bj = j;
                    bk = k;
                }
            }
        }
        perm[bj] = bk;
        used[bk] = 1;
    }
    free(used);
}
