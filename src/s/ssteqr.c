/**
 * @file ssteqr.c
 * @brief SSTEQR computes all eigenvalues and, optionally, eigenvectors of a
 *        symmetric tridiagonal matrix using the implicit QL or QR method.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSTEQR computes all eigenvalues and, optionally, eigenvectors of a
 * symmetric tridiagonal matrix using the implicit QL or QR method.
 * The eigenvectors of a full or band symmetric matrix can also be found
 * if SSYTRD or SSPTRD or SSBTRD has been used to reduce this matrix to
 * tridiagonal form.
 *
 * @param[in]     compz = 'N': Compute eigenvalues only.
 *                        = 'V': Compute eigenvalues and eigenvectors of the
 *                               original symmetric matrix. On entry, Z must
 *                               contain the orthogonal matrix used to reduce
 *                               the original matrix to tridiagonal form.
 *                        = 'I': Compute eigenvalues and eigenvectors of the
 *                               tridiagonal matrix. Z is initialized to the
 *                               identity matrix.
 * @param[in]     n     The order of the matrix. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the diagonal elements of the tridiagonal
 *                      matrix. On exit, if info = 0, the eigenvalues in
 *                      ascending order.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the subdiagonal elements of the tridiagonal
 *                      matrix. On exit, E has been destroyed.
 * @param[in,out] Z     Double precision array, dimension (ldz, n).
 *                      On entry, if compz = 'V', then Z contains the
 *                      orthogonal matrix used in the reduction to tridiagonal
 *                      form. On exit, if info = 0, then if compz = 'V', Z
 *                      contains the orthonormal eigenvectors of the original
 *                      symmetric matrix, and if compz = 'I', Z contains the
 *                      orthonormal eigenvectors of the symmetric tridiagonal
 *                      matrix. If compz = 'N', then Z is not referenced.
 * @param[in]     ldz   The leading dimension of the array Z. ldz >= 1, and if
 *                      eigenvectors are desired, then ldz >= max(1,n).
 * @param[out]    work  Double precision array, dimension (max(1,2*n-2)).
 *                      If compz = 'N', then work is not referenced.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value
 *                         - > 0: the algorithm has failed to find all the
 *                           eigenvalues in a total of 30*N iterations;
 *                           if info = i, then i elements of E have not
 *                           converged to zero.
 */
void ssteqr(const char* compz, const INT n,
            f32* restrict D, f32* restrict E,
            f32* restrict Z, const INT ldz,
            f32* restrict work, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;
    const INT MAXIT = 30;

    /* Test the input parameters. */
    *info = 0;

    INT icompz;
    if (compz[0] == 'N' || compz[0] == 'n') {
        icompz = 0;
    } else if (compz[0] == 'V' || compz[0] == 'v') {
        icompz = 1;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        icompz = 2;
    } else {
        icompz = -1;
    }

    if (icompz < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldz < 1 || (icompz > 0 && ldz < (1 > n ? 1 : n))) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SSTEQR", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    if (n == 1) {
        if (icompz == 2)
            Z[0] = ONE;
        return;
    }

    /* Determine the unit roundoff and over/underflow thresholds. */
    f32 eps = slamch("E");
    f32 eps2 = eps * eps;
    f32 safmin = slamch("S");
    f32 safmax = ONE / safmin;
    f32 ssfmax = sqrtf(safmax) / THREE;
    f32 ssfmin = sqrtf(safmin) / eps2;

    /* Compute the eigenvalues and eigenvectors of the tridiagonal matrix. */
    if (icompz == 2)
        slaset("F", n, n, ZERO, ONE, Z, ldz);

    INT nmaxit = n * MAXIT;
    INT jtot = 0;

    /* Determine where the matrix splits and choose QL or QR iteration
     * for each block, according to whether top or bottom diagonal
     * element is smaller. */
    INT l1 = 0;
    INT nm1 = n - 1;

    while (l1 < n) {
        if (l1 > 0)
            E[l1 - 1] = ZERO;

        INT m;
        if (l1 <= nm1 - 1) {
            for (m = l1; m <= nm1 - 1; m++) {
                f32 tst = fabsf(E[m]);
                if (tst == ZERO)
                    break;
                if (tst <= (sqrtf(fabsf(D[m])) * sqrtf(fabsf(D[m + 1]))) * eps) {
                    E[m] = ZERO;
                    break;
                }
            }
            if (m > nm1 - 1)
                m = n - 1;
        } else {
            m = n - 1;
        }

        INT l = l1;
        INT lsv = l;
        INT lend = m;
        INT lendsv = lend;
        l1 = m + 1;
        if (lend == l)
            continue;

        /* Scale submatrix in rows and columns l to lend */
        INT subsize = lend - l + 1;
        f32 anorm = slanst("M", subsize, &D[l], &E[l]);
        INT iscale = 0;
        if (anorm == ZERO)
            continue;

        if (anorm > ssfmax) {
            iscale = 1;
            INT linfo;
            slascl("G", 0, 0, anorm, ssfmax, subsize, 1, &D[l], n, &linfo);
            slascl("G", 0, 0, anorm, ssfmax, subsize - 1, 1, &E[l], n, &linfo);
        } else if (anorm < ssfmin) {
            iscale = 2;
            INT linfo;
            slascl("G", 0, 0, anorm, ssfmin, subsize, 1, &D[l], n, &linfo);
            slascl("G", 0, 0, anorm, ssfmin, subsize - 1, 1, &E[l], n, &linfo);
        }

        /* Choose between QL and QR iteration */
        if (fabsf(D[lend]) < fabsf(D[l])) {
            lend = lsv;
            l = lendsv;
        }

        if (lend > l) {
            /* QL Iteration
             * Look for small subdiagonal element. */
            for (;;) {
                INT mm;
                if (l != lend) {
                    INT lendm1 = lend - 1;
                    for (mm = l; mm <= lendm1; mm++) {
                        f32 tst = fabsf(E[mm]) * fabsf(E[mm]);
                        if (tst <= (eps2 * fabsf(D[mm])) * fabsf(D[mm + 1]) + safmin)
                            break;
                    }
                    if (mm > lendm1)
                        mm = lend;
                } else {
                    mm = lend;
                }

                if (mm < lend)
                    E[mm] = ZERO;
                f32 p = D[l];
                if (mm == l) {
                    /* Eigenvalue found. */
                    D[l] = p;
                    l = l + 1;
                    if (l <= lend)
                        continue;
                    break;
                }

                /* If remaining matrix is 2-by-2, use SLAE2 or SLAEV2
                 * to compute its eigensystem. */
                if (mm == l + 1) {
                    f32 rt1, rt2;
                    if (icompz > 0) {
                        f32 c, s;
                        slaev2(D[l], E[l], D[l + 1], &rt1, &rt2, &c, &s);
                        work[l] = c;
                        work[n - 1 + l] = s;
                        slasr("R", "V", "B", n, 2, &work[l],
                              &work[n - 1 + l], &Z[0 + l * ldz], ldz);
                    } else {
                        slae2(D[l], E[l], D[l + 1], &rt1, &rt2);
                    }
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
                f32 g = (D[l + 1] - p) / (TWO * E[l]);
                f32 r = slapy2(g, ONE);
                g = D[mm] - p + (E[l] / (g + copysignf(r, g)));

                f32 s = ONE;
                f32 c = ONE;
                p = ZERO;

                /* Inner loop */
                INT mm1 = mm - 1;
                for (INT i = mm1; i >= l; i--) {
                    f32 f = s * E[i];
                    f32 b = c * E[i];
                    slartg(g, f, &c, &s, &r);
                    if (i != mm - 1)
                        E[i + 1] = r;
                    g = D[i + 1] - p;
                    r = (D[i] - g) * s + TWO * c * b;
                    p = s * r;
                    D[i + 1] = g + p;
                    g = c * r - b;

                    /* If eigenvectors are desired, then save rotations. */
                    if (icompz > 0) {
                        work[i] = c;
                        work[n - 1 + i] = -s;
                    }
                }

                /* If eigenvectors are desired, then apply saved rotations. */
                if (icompz > 0) {
                    INT mmm = mm - l + 1;
                    slasr("R", "V", "B", n, mmm, &work[l],
                          &work[n - 1 + l], &Z[0 + l * ldz], ldz);
                }

                D[l] = D[l] - p;
                E[l] = g;
                /* Continue QL iteration */
            }
        } else {
            /* QR Iteration
             * Look for small superdiagonal element. */
            for (;;) {
                INT mm;
                if (l != lend) {
                    INT lendp1 = lend + 1;
                    for (mm = l; mm >= lendp1; mm--) {
                        f32 tst = fabsf(E[mm - 1]) * fabsf(E[mm - 1]);
                        if (tst <= (eps2 * fabsf(D[mm])) * fabsf(D[mm - 1]) + safmin)
                            break;
                    }
                    if (mm < lendp1)
                        mm = lend;
                } else {
                    mm = lend;
                }

                if (mm > lend)
                    E[mm - 1] = ZERO;
                f32 p = D[l];
                if (mm == l) {
                    /* Eigenvalue found. */
                    D[l] = p;
                    l = l - 1;
                    if (l >= lend)
                        continue;
                    break;
                }

                /* If remaining matrix is 2-by-2, use SLAE2 or SLAEV2
                 * to compute its eigensystem. */
                if (mm == l - 1) {
                    f32 rt1, rt2;
                    if (icompz > 0) {
                        f32 c, s;
                        slaev2(D[l - 1], E[l - 1], D[l], &rt1, &rt2, &c, &s);
                        work[mm] = c;
                        work[n - 1 + mm] = s;
                        slasr("R", "V", "F", n, 2, &work[mm],
                              &work[n - 1 + mm], &Z[0 + (l - 1) * ldz], ldz);
                    } else {
                        slae2(D[l - 1], E[l - 1], D[l], &rt1, &rt2);
                    }
                    D[l - 1] = rt1;
                    D[l] = rt2;
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
                f32 g = (D[l - 1] - p) / (TWO * E[l - 1]);
                f32 r = slapy2(g, ONE);
                g = D[mm] - p + (E[l - 1] / (g + copysignf(r, g)));

                f32 s = ONE;
                f32 c = ONE;
                p = ZERO;

                /* Inner loop */
                INT lm1 = l - 1;
                for (INT i = mm; i <= lm1; i++) {
                    f32 f = s * E[i];
                    f32 b = c * E[i];
                    slartg(g, f, &c, &s, &r);
                    if (i != mm)
                        E[i - 1] = r;
                    g = D[i] - p;
                    r = (D[i + 1] - g) * s + TWO * c * b;
                    p = s * r;
                    D[i] = g + p;
                    g = c * r - b;

                    /* If eigenvectors are desired, then save rotations. */
                    if (icompz > 0) {
                        work[i] = c;
                        work[n - 1 + i] = s;
                    }
                }

                /* If eigenvectors are desired, then apply saved rotations. */
                if (icompz > 0) {
                    INT mmm = l - mm + 1;
                    slasr("R", "V", "F", n, mmm, &work[mm],
                          &work[n - 1 + mm], &Z[0 + mm * ldz], ldz);
                }

                D[l] = D[l] - p;
                E[lm1] = g;
                /* Continue QR iteration */
            }
        }

        /* Undo scaling if necessary */
        if (iscale == 1) {
            INT linfo;
            slascl("G", 0, 0, ssfmax, anorm, lendsv - lsv + 1, 1,
                   &D[lsv], n, &linfo);
            slascl("G", 0, 0, ssfmax, anorm, lendsv - lsv, 1,
                   &E[lsv], n, &linfo);
        } else if (iscale == 2) {
            INT linfo;
            slascl("G", 0, 0, ssfmin, anorm, lendsv - lsv + 1, 1,
                   &D[lsv], n, &linfo);
            slascl("G", 0, 0, ssfmin, anorm, lendsv - lsv, 1,
                   &E[lsv], n, &linfo);
        }

        /* Check for no convergence to an eigenvalue after a total
         * of N*MAXIT iterations. */
        if (jtot >= nmaxit) {
            for (INT i = 0; i < n - 1; i++) {
                if (E[i] != ZERO)
                    (*info)++;
            }
            return;
        }
    }

    /* Order eigenvalues and eigenvectors. */
    if (icompz == 0) {
        /* Use Quick Sort */
        INT linfo;
        slasrt("I", n, D, &linfo);
    } else {
        /* Use Selection Sort to minimize swaps of eigenvectors */
        for (INT ii = 1; ii < n; ii++) {
            INT i = ii - 1;
            INT k = i;
            f32 p = D[i];
            for (INT j = ii; j < n; j++) {
                if (D[j] < p) {
                    k = j;
                    p = D[j];
                }
            }
            if (k != i) {
                D[k] = D[i];
                D[i] = p;
                cblas_sswap(n, &Z[0 + i * ldz], 1, &Z[0 + k * ldz], 1);
            }
        }
    }
}
