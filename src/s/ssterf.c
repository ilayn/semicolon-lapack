/**
 * @file ssterf.c
 * @brief SSTERF computes all eigenvalues of a symmetric tridiagonal matrix
 *        using the Pal-Walker-Kahan variant of the QL or QR algorithm.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SSTERF computes all eigenvalues of a symmetric tridiagonal matrix
 * using the Pal-Walker-Kahan variant of the QL or QR algorithm.
 *
 * @param[in]     n     The order of the matrix. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal
 *                      matrix.
 *                      On exit, if info = 0, the eigenvalues in ascending
 *                      order.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the
 *                      tridiagonal matrix.
 *                      On exit, E has been destroyed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value
 *                         - > 0: the algorithm failed to find all of the
 *                           eigenvalues in a total of 30*N iterations;
 *                           if info = i, then i elements of E have not
 *                           converged to zero.
 */
void ssterf(const int n, float* const restrict D,
            float* const restrict E, int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float THREE = 3.0f;
    const int MAXIT = 30;

    /* Test the input parameters. */
    *info = 0;

    /* Quick return if possible */
    if (n < 0) {
        *info = -1;
        xerbla("SSTERF", -(*info));
        return;
    }
    if (n <= 1)
        return;

    /* Determine the unit roundoff for this environment. */
    float eps = slamch("E");
    float eps2 = eps * eps;
    float safmin = slamch("S");
    float safmax = ONE / safmin;
    float ssfmax = sqrtf(safmax) / THREE;
    float ssfmin = sqrtf(safmin) / eps2;

    /* Compute the eigenvalues of the tridiagonal matrix. */
    int nmaxit = n * MAXIT;
    float sigma = ZERO;
    int jtot = 0;

    /* Determine where the matrix splits and choose QL or QR iteration
     * for each block, according to whether top or bottom diagonal
     * element is smaller. */

    /* Fortran: L1=1..N (1-based). C: l1=0..n-1 (0-based). */
    int l1 = 0;

    while (l1 < n) {
        /* Look for a small subdiagonal element: E(m) negligible. */
        if (l1 > 0)
            E[l1 - 1] = ZERO;

        int m;
        for (m = l1; m < n - 1; m++) {
            if (fabsf(E[m]) <= (sqrtf(fabsf(D[m])) * sqrtf(fabsf(D[m + 1]))) * eps) {
                E[m] = ZERO;
                break;
            }
        }
        if (m == n - 1 && (fabsf(E[m - 1]) > (sqrtf(fabsf(D[m - 1])) * sqrtf(fabsf(D[m]))) * eps)) {
            /* No split found - block extends to end */
            m = n - 1;
            /* Check the last element again; for loop ended without break */
        }
        /* After this: the block is D[l..m], subdiagonals E[l..m-1]. */

        int l = l1;
        int lsv = l;
        int lend = m;
        int lendsv = lend;
        l1 = m + 1;
        if (lend == l)
            continue;

        /* Scale submatrix in rows and columns l to lend */
        int subsize = lend - l + 1;
        float anorm = slanst("M", subsize, &D[l], &E[l]);
        int iscale = 0;
        if (anorm == ZERO)
            continue;

        if (anorm > ssfmax) {
            iscale = 1;
            int linfo;
            slascl("G", 0, 0, anorm, ssfmax, subsize, 1, &D[l], n, &linfo);
            slascl("G", 0, 0, anorm, ssfmax, subsize - 1, 1, &E[l], n, &linfo);
        } else if (anorm < ssfmin) {
            iscale = 2;
            int linfo;
            slascl("G", 0, 0, anorm, ssfmin, subsize, 1, &D[l], n, &linfo);
            slascl("G", 0, 0, anorm, ssfmin, subsize - 1, 1, &E[l], n, &linfo);
        }

        /* Square the subdiagonal elements */
        for (int i = l; i < lend; i++) {
            E[i] = E[i] * E[i];
        }

        /* Choose between QL and QR iteration */
        if (fabsf(D[lend]) < fabsf(D[l])) {
            lend = lsv;
            l = lendsv;
        }

        if (lend >= l) {
            /* QL Iteration */
            /* Look for small subdiagonal element. */
            for (;;) {
                int mm;
                if (l != lend) {
                    for (mm = l; mm < lend; mm++) {
                        if (fabsf(E[mm]) <= eps2 * fabsf(D[mm] * D[mm + 1]))
                            break;
                    }
                    if (mm == lend) {
                        /* No small element found */
                    }
                } else {
                    mm = lend;
                }

                if (mm < lend)
                    E[mm] = ZERO;
                float p = D[l];
                if (mm == l) {
                    /* Eigenvalue found. */
                    D[l] = p;
                    l = l + 1;
                    if (l <= lend)
                        continue;
                    break;
                }

                /* If remaining matrix is 2 by 2, use SLAE2 to compute its
                 * eigenvalues. */
                if (mm == l + 1) {
                    float rte = sqrtf(E[l]);
                    float rt1, rt2;
                    slae2(D[l], rte, D[l + 1], &rt1, &rt2);
                    D[l] = rt1;
                    D[l + 1] = rt2;
                    E[l] = ZERO;
                    l = l + 2;
                    if (l <= lend)
                        continue;
                    break;
                }

                if (jtot == nmaxit)
                    break;
                jtot = jtot + 1;

                /* Form shift. */
                float rte = sqrtf(E[l]);
                sigma = (D[l + 1] - p) / (TWO * rte);
                float r = slapy2(sigma, ONE);
                sigma = p - (rte / (sigma + copysignf(r, sigma)));

                float c = ONE;
                float s = ZERO;
                float gamma = D[mm] - sigma;
                p = gamma * gamma;

                /* Inner loop */
                for (int i = mm - 1; i >= l; i--) {
                    float bb = E[i];
                    r = p + bb;
                    if (i != mm - 1)
                        E[i + 1] = s * r;
                    float oldc = c;
                    c = p / r;
                    s = bb / r;
                    float oldgam = gamma;
                    float alpha = D[i];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    D[i + 1] = oldgam + (alpha - gamma);
                    if (c != ZERO) {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                }

                E[l] = s * p;
                D[l] = sigma + gamma;
                /* Continue QL iteration */
            }
        } else {
            /* QR Iteration */
            /* Look for small superdiagonal element. */
            for (;;) {
                int mm;
                if (l != lend) {
                    for (mm = l; mm > lend; mm--) {
                        if (fabsf(E[mm - 1]) <= eps2 * fabsf(D[mm] * D[mm - 1]))
                            break;
                    }
                    if (mm == lend) {
                        /* No small element found */
                    }
                } else {
                    mm = lend;
                }

                if (mm > lend)
                    E[mm - 1] = ZERO;
                float p = D[l];
                if (mm == l) {
                    /* Eigenvalue found. */
                    D[l] = p;
                    l = l - 1;
                    if (l >= lend)
                        continue;
                    break;
                }

                /* If remaining matrix is 2 by 2, use SLAE2 to compute its
                 * eigenvalues. */
                if (mm == l - 1) {
                    float rte = sqrtf(E[l - 1]);
                    float rt1, rt2;
                    slae2(D[l], rte, D[l - 1], &rt1, &rt2);
                    D[l] = rt1;
                    D[l - 1] = rt2;
                    E[l - 1] = ZERO;
                    l = l - 2;
                    if (l >= lend)
                        continue;
                    break;
                }

                if (jtot == nmaxit)
                    break;
                jtot = jtot + 1;

                /* Form shift. */
                float rte = sqrtf(E[l - 1]);
                sigma = (D[l - 1] - p) / (TWO * rte);
                float r = slapy2(sigma, ONE);
                sigma = p - (rte / (sigma + copysignf(r, sigma)));

                float c = ONE;
                float s = ZERO;
                float gamma = D[mm] - sigma;
                p = gamma * gamma;

                /* Inner loop */
                for (int i = mm; i < l; i++) {
                    float bb = E[i];
                    r = p + bb;
                    if (i != mm)
                        E[i - 1] = s * r;
                    float oldc = c;
                    c = p / r;
                    s = bb / r;
                    float oldgam = gamma;
                    float alpha = D[i + 1];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    D[i] = oldgam + (alpha - gamma);
                    if (c != ZERO) {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                }

                E[l - 1] = s * p;
                D[l] = sigma + gamma;
                /* Continue QR iteration */
            }
        }

        /* Undo scaling if necessary */
        if (iscale == 1) {
            int linfo;
            slascl("G", 0, 0, ssfmax, anorm, lendsv - lsv + 1, 1,
                   &D[lsv], n, &linfo);
        }
        if (iscale == 2) {
            int linfo;
            slascl("G", 0, 0, ssfmin, anorm, lendsv - lsv + 1, 1,
                   &D[lsv], n, &linfo);
        }

        /* Check for no convergence to an eigenvalue after a total
         * of N*MAXIT iterations. */
        if (jtot >= nmaxit) {
            for (int i = 0; i < n - 1; i++) {
                if (E[i] != ZERO)
                    (*info)++;
            }
            return;
        }
    }

    /* Sort eigenvalues in increasing order. */
    int linfo;
    slasrt("I", n, D, &linfo);
}
