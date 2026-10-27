/**
 * @file cpair_eigs.c
 * @brief CPAIR_EIGS pairs two lists of the same eigenvalues by closest match.
 */

#include <math.h>
#include <stdlib.h>
#include "verify.h"

/**
 * CPAIR_EIGS pairs the eigenvalues w1(k) one-to-one with w(j), taking the
 * closest remaining pair first, with distance |w(j) - w1(k)|. Two runs that
 * compute the same eigenvalues can return them in a different order;
 * comparing them through this pairing instead of by index tells a
 * reordering apart from an error.
 *
 * @param[in]  n     The number of eigenvalues in each list. n >= 0.
 * @param[in]  w     The first list, dimension (n).
 * @param[in]  w1    The second list, dimension (n).
 * @param[out] perm  Integer array, dimension (n). Eigenvalue j of the first
 *                   list is paired with eigenvalue perm[j] of the second.
 *                   Identical lists give perm[j] = j.
 */
void cpair_eigs(const INT n, const c64* w, const c64* w1, INT* perm)
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
                f32 d = cabsf(w[j] - w1[k]);
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
