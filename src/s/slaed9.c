/**
 * @file slaed9.c
 * @brief SLAED9 finds the roots of the secular equation and updates the
 *        eigenvectors. Used when the original matrix is dense.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAED9 finds the roots of the secular equation, as defined by the
 * values in D, Z, and RHO, between KSTART and KSTOP. It makes the
 * appropriate calls to SLAED4 and then stores the new matrix of
 * eigenvectors for use in calculating the next level of Z vectors.
 *
 * @param[in]     k       The number of terms in the rational function to be
 *                        solved by SLAED4. k >= 0.
 * @param[in]     kstart  The starting index (0-based) of eigenvalues to compute.
 * @param[in]     kstop   The stopping index (0-based, exclusive). The updated
 *                        eigenvalues D(i), kstart <= i < kstop, are computed.
 *                        0 <= kstart < kstop <= k.
 * @param[in]     n       The number of rows and columns in the Q matrix.
 *                        n >= k (deflation may result in n > k).
 * @param[out]    D       Double precision array, dimension (n).
 *                        D(i) contains the updated eigenvalues
 *                        for kstart <= i < kstop.
 * @param[out]    Q       Double precision array, dimension (ldq, n).
 * @param[in]     ldq     The leading dimension of the array Q. ldq >= max(1, k).
 * @param[in]     rho     The value of the parameter in the rank one update
 *                        equation. rho >= 0 required.
 * @param[in]     dlambda Double precision array, dimension (k).
 *                        The first k elements contain the old roots of the
 *                        deflated updating problem. These are the poles of the
 *                        secular equation.
 * @param[in]     W       Double precision array, dimension (k).
 *                        The first k elements contain the components of the
 *                        deflation-adjusted updating vector.
 * @param[out]    S       Double precision array, dimension (lds, k).
 *                        Will contain the eigenvectors of the repaired matrix
 *                        which will be stored for subsequent Z vector calculation
 *                        and multiplied by the previously accumulated eigenvectors
 *                        to update the system.
 * @param[in]     lds     The leading dimension of S. lds >= max(1, k).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 *                        > 0: if info = 1, an eigenvalue did not converge.
 */
void slaed9(const int k, const int kstart, const int kstop, const int n,
            float* D, float* Q, const int ldq, const float rho,
            const float* dlambda, float* W, float* S, const int lds,
            int* info)
{
    int i, j;
    float temp;

    /* Test the input parameters. */
    *info = 0;

    if (k < 0) {
        *info = -1;
    } else if (kstart < 0 || kstart > (k > 1 ? k - 1 : 0)) {
        *info = -2;
    } else if (kstop < kstart + 1 || kstop > k) {
        *info = -3;
    } else if (n < k) {
        *info = -4;
    } else if (ldq < (1 > k ? 1 : k)) {
        *info = -7;
    } else if (lds < (1 > k ? 1 : k)) {
        *info = -12;
    }
    if (*info != 0) {
        xerbla("SLAED9", -(*info));
        return;
    }

    /* Quick return if possible */
    if (k == 0)
        return;

    /* For each eigenvalue in [kstart, kstop), call slaed4 to find the root. */
    for (j = kstart; j < kstop; j++) {
        slaed4(k, j, dlambda, W, &Q[j * ldq], rho, &D[j], info);

        /* If the zero finder fails, the computation is terminated. */
        if (*info != 0)
            return;
    }

    if (k == 1 || k == 2) {
        for (i = 0; i < k; i++) {
            for (j = 0; j < k; j++) {
                S[i * lds + j] = Q[i * ldq + j];
            }
        }
        return;
    }

    /* Compute updated W. */
    /* Save W into first column of S */
    cblas_scopy(k, W, 1, S, 1);

    /* Initialize W(i) = Q(i,i) */
    cblas_scopy(k, Q, ldq + 1, W, 1);

    for (j = 0; j < k; j++) {
        for (i = 0; i < j; i++) {
            W[i] = W[i] * (Q[j * ldq + i] / (dlambda[i] - dlambda[j]));
        }
        for (i = j + 1; i < k; i++) {
            W[i] = W[i] * (Q[j * ldq + i] / (dlambda[i] - dlambda[j]));
        }
    }
    for (i = 0; i < k; i++) {
        W[i] = copysignf(sqrtf(-W[i]), S[i]);
    }

    /* Compute eigenvectors of the modified rank-1 modification. */
    for (j = 0; j < k; j++) {
        for (i = 0; i < k; i++) {
            Q[j * ldq + i] = W[i] / Q[j * ldq + i];
        }
        temp = cblas_snrm2(k, &Q[j * ldq], 1);
        for (i = 0; i < k; i++) {
            S[j * lds + i] = Q[j * ldq + i] / temp;
        }
    }
}
