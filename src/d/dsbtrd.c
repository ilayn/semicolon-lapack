/**
 * @file dsbtrd.c
 * @brief DSBTRD reduces a symmetric band matrix to symmetric tridiagonal form.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSBTRD reduces a real symmetric band matrix A to symmetric
 * tridiagonal form T by an orthogonal similarity transformation:
 * Q**T * A * Q = T.
 *
 * @param[in]     vect   = 'N': do not form Q
 *                        = 'V': form Q
 *                        = 'U': update a matrix X, by forming X*Q
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, the diagonal and off-diagonal elements are
 *                       overwritten by the tridiagonal matrix T.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    D      The diagonal elements of T. Array of dimension (n).
 * @param[out]    E      The off-diagonal elements of T. Array of dimension (n-1).
 * @param[in,out] Q      On entry, if vect='U', an n-by-n matrix X.
 *                       On exit, if vect='V', the orthogonal Q.
 *                       If vect='U', the product X*Q.
 *                       Array of dimension (ldq, n).
 * @param[in]     ldq    The leading dimension of Q. ldq >= 1, and ldq >= n if vect='V' or 'U'.
 * @param[out]    work   Workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dsbtrd(
    const char* vect,
    const char* uplo,
    const int n,
    const int kd,
    f64* restrict AB,
    const int ldab,
    f64* restrict D,
    f64* restrict E,
    f64* restrict Q,
    const int ldq,
    f64* restrict work,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int initq, upper, wantq;
    int i, i2, ibl, inca, incx, iqaend, iqb, iqend, j, j1, j1end, j1inc, j2;
    int jend, jin, jinc, k, kd1, kdm1, kdn, l, last, lend, nq, nr, nrt;
    f64 temp;

    initq = (vect[0] == 'V' || vect[0] == 'v');
    wantq = initq || (vect[0] == 'U' || vect[0] == 'u');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    kd1 = kd + 1;
    kdm1 = kd - 1;
    incx = ldab - 1;
    iqend = 1;

    *info = 0;
    if (!wantq && !(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (kd < 0) {
        *info = -4;
    } else if (ldab < kd1) {
        *info = -6;
    } else if (ldq < (1 > n ? 1 : n) && wantq) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DSBTRD", -(*info));
        return;
    }

    if (n == 0)
        return;

    if (initq)
        dlaset("F", n, n, ZERO, ONE, Q, ldq);

    inca = kd1 * ldab;
    kdn = (n - 1 < kd) ? (n - 1) : kd;

    if (upper) {
        if (kd > 1) {
            // Reduce to tridiagonal form, working with upper triangle
            nr = 0;
            j1 = kdn + 2;
            j2 = 1;

            for (i = 1; i <= n - 2; i++) {
                // Reduce i-th row of matrix to tridiagonal form
                for (k = kdn + 1; k >= 2; k--) {
                    j1 = j1 + kdn;
                    j2 = j2 + kdn;

                    if (nr > 0) {
                        // Generate plane rotations to annihilate nonzero elements
                        // which have been created outside the band
                        dlargv(nr, &AB[0 + (j1 - 2) * ldab], inca, &work[j1 - 1], kd1, &D[j1 - 1], kd1);

                        // Apply rotations from the right
                        if (nr >= 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                dlartv(nr, &AB[l + (j1 - 2) * ldab], inca,
                                       &AB[l - 1 + (j1 - 1) * ldab], inca, &D[j1 - 1],
                                       &work[j1 - 1], kd1);
                            }
                        } else {
                            jend = j1 + (nr - 1) * kd1;
                            for (jinc = j1; jinc <= jend; jinc += kd1) {
                                cblas_drot(kdm1, &AB[1 + (jinc - 2) * ldab], 1,
                                           &AB[0 + (jinc - 1) * ldab], 1, D[jinc - 1],
                                           work[jinc - 1]);
                            }
                        }
                    }

                    if (k > 2) {
                        if (k <= n - i + 1) {
                            // Generate plane rotation to annihilate a(i,i+k-1)
                            // within the band
                            dlartg(AB[kd - k + 2 + (i + k - 3) * ldab],
                                   AB[kd - k + 1 + (i + k - 2) * ldab], &D[i + k - 2],
                                   &work[i + k - 2], &temp);
                            AB[kd - k + 2 + (i + k - 3) * ldab] = temp;

                            // Apply rotation from the right
                            cblas_drot(k - 3, &AB[kd - k + 3 + (i + k - 3) * ldab], 1,
                                       &AB[kd - k + 2 + (i + k - 2) * ldab], 1, D[i + k - 2],
                                       work[i + k - 2]);
                        }
                        nr = nr + 1;
                        j1 = j1 - kdn - 1;
                    }

                    // Apply plane rotations from both sides to diagonal blocks
                    if (nr > 0)
                        dlar2v(nr, &AB[kd + (j1 - 2) * ldab], &AB[kd + (j1 - 1) * ldab],
                               &AB[kd - 1 + (j1 - 1) * ldab], inca, &D[j1 - 1],
                               &work[j1 - 1], kd1);

                    // Apply plane rotations from the left
                    if (nr > 0) {
                        if (2 * kd - 1 < nr) {
                            for (l = 1; l <= kd - 1; l++) {
                                if (j2 + l > n) {
                                    nrt = nr - 1;
                                } else {
                                    nrt = nr;
                                }
                                if (nrt > 0)
                                    dlartv(nrt, &AB[kd - l - 1 + (j1 + l - 1) * ldab], inca,
                                           &AB[kd - l + (j1 + l - 1) * ldab], inca,
                                           &D[j1 - 1], &work[j1 - 1], kd1);
                            }
                        } else {
                            j1end = j1 + kd1 * (nr - 2);
                            if (j1end >= j1) {
                                for (jin = j1; jin <= j1end; jin += kd1) {
                                    cblas_drot(kd - 1, &AB[kd - 2 + jin * ldab], incx,
                                               &AB[kd - 1 + jin * ldab], incx,
                                               D[jin - 1], work[jin - 1]);
                                }
                            }
                            lend = (kdm1 < n - j2) ? kdm1 : (n - j2);
                            last = j1end + kd1;
                            if (lend > 0)
                                cblas_drot(lend, &AB[kd - 2 + last * ldab], incx,
                                           &AB[kd - 1 + last * ldab], incx,
                                           D[last - 1], work[last - 1]);
                        }
                    }

                    if (wantq) {
                        // Accumulate product of plane rotations in Q
                        if (initq) {
                            // Take advantage of the fact that Q was initially identity
                            iqend = (iqend > j2) ? iqend : j2;
                            i2 = (0 > k - 3) ? 0 : (k - 3);
                            iqaend = 1 + i * kd;
                            if (k == 2)
                                iqaend = iqaend + kd;
                            iqaend = (iqaend < iqend) ? iqaend : iqend;
                            for (j = j1; j <= j2; j += kd1) {
                                ibl = i - i2 / kdm1;
                                i2 = i2 + 1;
                                iqb = (1 > j - ibl) ? 1 : (j - ibl);
                                nq = 1 + iqaend - iqb;
                                iqaend = (iqaend + kd < iqend) ? (iqaend + kd) : iqend;
                                cblas_drot(nq, &Q[(iqb - 1) + (j - 2) * ldq], 1,
                                           &Q[(iqb - 1) + (j - 1) * ldq], 1,
                                           D[j - 1], work[j - 1]);
                            }
                        } else {
                            for (j = j1; j <= j2; j += kd1) {
                                cblas_drot(n, &Q[0 + (j - 2) * ldq], 1,
                                           &Q[0 + (j - 1) * ldq], 1,
                                           D[j - 1], work[j - 1]);
                            }
                        }
                    }

                    if (j2 + kdn > n) {
                        // Adjust J2 to keep within the bounds of the matrix
                        nr = nr - 1;
                        j2 = j2 - kdn - 1;
                    }

                    for (j = j1; j <= j2; j += kd1) {
                        // Create nonzero element a(j-1,j+kd) outside the band
                        // and store it in WORK
                        work[j + kd - 1] = work[j - 1] * AB[0 + (j + kd - 1) * ldab];
                        AB[0 + (j + kd - 1) * ldab] = D[j - 1] * AB[0 + (j + kd - 1) * ldab];
                    }
                }
            }
        }

        if (kd > 0) {
            // Copy off-diagonal elements to E
            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = AB[kd - 1 + i * ldab];
            }
        } else {
            // Set E to zero if original matrix was diagonal
            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = ZERO;
            }
        }

        // Copy diagonal elements to D
        for (i = 1; i <= n; i++) {
            D[i - 1] = AB[kd + (i - 1) * ldab];
        }

    } else {
        if (kd > 1) {
            // Reduce to tridiagonal form, working with lower triangle
            nr = 0;
            j1 = kdn + 2;
            j2 = 1;

            for (i = 1; i <= n - 2; i++) {
                // Reduce i-th column of matrix to tridiagonal form
                for (k = kdn + 1; k >= 2; k--) {
                    j1 = j1 + kdn;
                    j2 = j2 + kdn;

                    if (nr > 0) {
                        // Generate plane rotations to annihilate nonzero elements
                        // which have been created outside the band
                        dlargv(nr, &AB[kd + (j1 - kd1 - 1) * ldab], inca, &work[j1 - 1], kd1, &D[j1 - 1], kd1);

                        // Apply plane rotations from one side
                        if (nr > 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                dlartv(nr, &AB[kd - l + (j1 - kd1 + l - 1) * ldab], inca,
                                       &AB[kd - l + 1 + (j1 - kd1 + l - 1) * ldab], inca,
                                       &D[j1 - 1], &work[j1 - 1], kd1);
                            }
                        } else {
                            jend = j1 + kd1 * (nr - 1);
                            for (jinc = j1; jinc <= jend; jinc += kd1) {
                                cblas_drot(kdm1, &AB[kd - 1 + (jinc - kd - 1) * ldab], incx,
                                           &AB[kd + (jinc - kd - 1) * ldab], incx,
                                           D[jinc - 1], work[jinc - 1]);
                            }
                        }
                    }

                    if (k > 2) {
                        if (k <= n - i + 1) {
                            // Generate plane rotation to annihilate a(i+k-1,i)
                            // within the band
                            dlartg(AB[k - 2 + (i - 1) * ldab], AB[k - 1 + (i - 1) * ldab],
                                   &D[i + k - 2], &work[i + k - 2], &temp);
                            AB[k - 2 + (i - 1) * ldab] = temp;

                            // Apply rotation from the left
                            cblas_drot(k - 3, &AB[k - 3 + i * ldab], ldab - 1,
                                       &AB[k - 2 + i * ldab], ldab - 1, D[i + k - 2],
                                       work[i + k - 2]);
                        }
                        nr = nr + 1;
                        j1 = j1 - kdn - 1;
                    }

                    // Apply plane rotations from both sides to diagonal blocks
                    if (nr > 0)
                        dlar2v(nr, &AB[0 + (j1 - 2) * ldab], &AB[0 + (j1 - 1) * ldab],
                               &AB[1 + (j1 - 2) * ldab], inca, &D[j1 - 1],
                               &work[j1 - 1], kd1);

                    // Apply plane rotations from the right
                    if (nr > 0) {
                        if (nr > 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                if (j2 + l > n) {
                                    nrt = nr - 1;
                                } else {
                                    nrt = nr;
                                }
                                if (nrt > 0)
                                    dlartv(nrt, &AB[l + 1 + (j1 - 2) * ldab], inca,
                                           &AB[l + (j1 - 1) * ldab], inca, &D[j1 - 1],
                                           &work[j1 - 1], kd1);
                            }
                        } else {
                            j1end = j1 + kd1 * (nr - 2);
                            if (j1end >= j1) {
                                for (j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                    cblas_drot(kdm1, &AB[2 + (j1inc - 2) * ldab], 1,
                                               &AB[1 + (j1inc - 1) * ldab], 1, D[j1inc - 1],
                                               work[j1inc - 1]);
                                }
                            }
                            lend = (kdm1 < n - j2) ? kdm1 : (n - j2);
                            last = j1end + kd1;
                            if (lend > 0)
                                cblas_drot(lend, &AB[2 + (last - 2) * ldab], 1,
                                           &AB[1 + (last - 1) * ldab], 1, D[last - 1],
                                           work[last - 1]);
                        }
                    }

                    if (wantq) {
                        // Accumulate product of plane rotations in Q
                        if (initq) {
                            // Take advantage of the fact that Q was initially identity
                            iqend = (iqend > j2) ? iqend : j2;
                            i2 = (0 > k - 3) ? 0 : (k - 3);
                            iqaend = 1 + i * kd;
                            if (k == 2)
                                iqaend = iqaend + kd;
                            iqaend = (iqaend < iqend) ? iqaend : iqend;
                            for (j = j1; j <= j2; j += kd1) {
                                ibl = i - i2 / kdm1;
                                i2 = i2 + 1;
                                iqb = (1 > j - ibl) ? 1 : (j - ibl);
                                nq = 1 + iqaend - iqb;
                                iqaend = (iqaend + kd < iqend) ? (iqaend + kd) : iqend;
                                cblas_drot(nq, &Q[(iqb - 1) + (j - 2) * ldq], 1,
                                           &Q[(iqb - 1) + (j - 1) * ldq], 1,
                                           D[j - 1], work[j - 1]);
                            }
                        } else {
                            for (j = j1; j <= j2; j += kd1) {
                                cblas_drot(n, &Q[0 + (j - 2) * ldq], 1,
                                           &Q[0 + (j - 1) * ldq], 1,
                                           D[j - 1], work[j - 1]);
                            }
                        }
                    }

                    if (j2 + kdn > n) {
                        // Adjust J2 to keep within the bounds of the matrix
                        nr = nr - 1;
                        j2 = j2 - kdn - 1;
                    }

                    for (j = j1; j <= j2; j += kd1) {
                        // Create nonzero element a(j+kd,j-1) outside the band
                        // and store it in WORK
                        work[j + kd - 1] = work[j - 1] * AB[kd + (j - 1) * ldab];
                        AB[kd + (j - 1) * ldab] = D[j - 1] * AB[kd + (j - 1) * ldab];
                    }
                }
            }
        }

        if (kd > 0) {
            // Copy off-diagonal elements to E
            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = AB[1 + (i - 1) * ldab];
            }
        } else {
            // Set E to zero if original matrix was diagonal
            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = ZERO;
            }
        }

        // Copy diagonal elements to D
        for (i = 1; i <= n; i++) {
            D[i - 1] = AB[0 + (i - 1) * ldab];
        }
    }
}
