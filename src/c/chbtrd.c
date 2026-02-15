/**
 * @file chbtrd.c
 * @brief CHBTRD reduces a complex Hermitian band matrix to real symmetric
 *        tridiagonal form.
 */

#include <cblas.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHBTRD reduces a complex Hermitian band matrix A to real symmetric
 * tridiagonal form T by a unitary similarity transformation:
 * Q**H * A * Q = T.
 *
 * @param[in]     vect   = 'N': do not form Q
 *                        = 'V': form Q
 *                        = 'U': update a matrix X, by forming X*Q
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     Complex*16 array, dimension (ldab, n).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix A, stored in the first kd+1 rows.
 *                       On exit, the diagonal elements are overwritten by the
 *                       diagonal elements of T; if kd > 0, the elements on the
 *                       first superdiagonal (if uplo='U') or subdiagonal
 *                       (if uplo='L') are overwritten by the off-diagonal
 *                       elements of T; the rest is overwritten by values
 *                       generated during the reduction.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    D      Single precision array, dimension (n).
 *                       The diagonal elements of T.
 * @param[out]    E      Single precision array, dimension (n-1).
 *                       The off-diagonal elements of T.
 * @param[in,out] Q      Complex*16 array, dimension (ldq, n).
 *                       On entry, if vect='U', an n-by-n matrix X.
 *                       On exit, if vect='V', the unitary Q.
 *                       If vect='U', the product X*Q.
 *                       If vect='N', not referenced.
 * @param[in]     ldq    The leading dimension of Q. ldq >= 1, and
 *                       ldq >= n if vect='V' or 'U'.
 * @param[out]    work   Complex*16 workspace array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an
 *                           illegal value
 */
void chbtrd(
    const char* vect,
    const char* uplo,
    const int n,
    const int kd,
    c64* restrict AB,
    const int ldab,
    f32* restrict D,
    f32* restrict E,
    c64* restrict Q,
    const int ldq,
    c64* restrict work,
    int* info)
{
    const f32 ZERO = 0.0f;
    const c64 CONE = 1.0f;

    int initq, upper, wantq;
    int i, i2, ibl, inca, incx, iqaend, iqb, iqend, j, j1, j1end, j1inc, j2;
    int jend, jin, jinc, k, kd1, kdm1, kdn, l, last, lend, nq, nr, nrt;
    f32 abst;
    c64 t, temp;

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
        xerbla("CHBTRD", -(*info));
        return;
    }

    if (n == 0)
        return;

    if (initq)
        claset("F", n, n, 0.0f, CONE, Q, ldq);

    inca = kd1 * ldab;
    kdn = (n - 1 < kd) ? (n - 1) : kd;

    if (upper) {

        if (kd > 1) {

            nr = 0;
            j1 = kdn + 2;
            j2 = 1;

            AB[kd] = crealf(AB[kd]);

            for (i = 1; i <= n - 2; i++) {

                for (k = kdn + 1; k >= 2; k--) {
                    j1 = j1 + kdn;
                    j2 = j2 + kdn;

                    if (nr > 0) {

                        clargv(nr, &AB[0 + (j1 - 2) * ldab], inca,
                               &work[j1 - 1], kd1, &D[j1 - 1], kd1);

                        if (nr >= 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                clartv(nr, &AB[l + (j1 - 2) * ldab], inca,
                                       &AB[l - 1 + (j1 - 1) * ldab], inca,
                                       &D[j1 - 1], &work[j1 - 1], kd1);
                            }

                        } else {
                            jend = j1 + (nr - 1) * kd1;
                            for (jinc = j1; jinc <= jend; jinc += kd1) {
                                crot(kdm1, &AB[1 + (jinc - 2) * ldab], 1,
                                     &AB[0 + (jinc - 1) * ldab], 1,
                                     D[jinc - 1], work[jinc - 1]);
                            }
                        }
                    }

                    if (k > 2) {
                        if (k <= n - i + 1) {

                            clartg(AB[kd - k + 2 + (i + k - 3) * ldab],
                                   AB[kd - k + 1 + (i + k - 2) * ldab],
                                   &D[i + k - 2], &work[i + k - 2], &temp);
                            AB[kd - k + 2 + (i + k - 3) * ldab] = temp;

                            crot(k - 3, &AB[kd - k + 3 + (i + k - 3) * ldab], 1,
                                 &AB[kd - k + 2 + (i + k - 2) * ldab], 1,
                                 D[i + k - 2], work[i + k - 2]);
                        }
                        nr = nr + 1;
                        j1 = j1 - kdn - 1;
                    }

                    if (nr > 0)
                        clar2v(nr, &AB[kd + (j1 - 2) * ldab],
                               &AB[kd + (j1 - 1) * ldab],
                               &AB[kd - 1 + (j1 - 1) * ldab], inca,
                               &D[j1 - 1], &work[j1 - 1], kd1);

                    if (nr > 0) {
                        clacgv(nr, &work[j1 - 1], kd1);
                        if (2 * kd - 1 < nr) {

                            for (l = 1; l <= kd - 1; l++) {
                                if (j2 + l > n) {
                                    nrt = nr - 1;
                                } else {
                                    nrt = nr;
                                }
                                if (nrt > 0)
                                    clartv(nrt, &AB[kd - l - 1 + (j1 + l - 1) * ldab],
                                           inca,
                                           &AB[kd - l + (j1 + l - 1) * ldab], inca,
                                           &D[j1 - 1], &work[j1 - 1], kd1);
                            }
                        } else {
                            j1end = j1 + kd1 * (nr - 2);
                            if (j1end >= j1) {
                                for (jin = j1; jin <= j1end; jin += kd1) {
                                    crot(kd - 1, &AB[kd - 2 + jin * ldab], incx,
                                         &AB[kd - 1 + jin * ldab], incx,
                                         D[jin - 1], work[jin - 1]);
                                }
                            }
                            lend = (kdm1 < n - j2) ? kdm1 : (n - j2);
                            last = j1end + kd1;
                            if (lend > 0)
                                crot(lend, &AB[kd - 2 + last * ldab], incx,
                                     &AB[kd - 1 + last * ldab], incx,
                                     D[last - 1], work[last - 1]);
                        }
                    }

                    if (wantq) {

                        if (initq) {

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
                                crot(nq, &Q[(iqb - 1) + (j - 2) * ldq], 1,
                                     &Q[(iqb - 1) + (j - 1) * ldq], 1,
                                     D[j - 1], conjf(work[j - 1]));
                            }
                        } else {

                            for (j = j1; j <= j2; j += kd1) {
                                crot(n, &Q[0 + (j - 2) * ldq], 1,
                                     &Q[0 + (j - 1) * ldq], 1,
                                     D[j - 1], conjf(work[j - 1]));
                            }
                        }

                    }

                    if (j2 + kdn > n) {

                        nr = nr - 1;
                        j2 = j2 - kdn - 1;
                    }

                    for (j = j1; j <= j2; j += kd1) {

                        work[j + kd - 1] = work[j - 1] * AB[0 + (j + kd - 1) * ldab];
                        AB[0 + (j + kd - 1) * ldab] = D[j - 1] * AB[0 + (j + kd - 1) * ldab];
                    }
                }
            }
        }

        if (kd > 0) {

            for (i = 1; i <= n - 1; i++) {
                t = AB[kd - 1 + i * ldab];
                abst = cabsf(t);
                AB[kd - 1 + i * ldab] = abst;
                E[i - 1] = abst;
                if (abst != ZERO) {
                    t = t / abst;
                } else {
                    t = CONE;
                }
                if (i < n - 1)
                    AB[kd - 1 + (i + 1) * ldab] = AB[kd - 1 + (i + 1) * ldab] * t;
                if (wantq) {
                    c64 ct = conjf(t);
                    cblas_cscal(n, &ct, &Q[0 + i * ldq], 1);
                }
            }
        } else {

            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = ZERO;
            }
        }

        for (i = 1; i <= n; i++) {
            D[i - 1] = crealf(AB[kd + (i - 1) * ldab]);
        }

    } else {

        if (kd > 1) {

            nr = 0;
            j1 = kdn + 2;
            j2 = 1;

            AB[0] = crealf(AB[0]);

            for (i = 1; i <= n - 2; i++) {

                for (k = kdn + 1; k >= 2; k--) {
                    j1 = j1 + kdn;
                    j2 = j2 + kdn;

                    if (nr > 0) {

                        clargv(nr, &AB[kd + (j1 - kd1 - 1) * ldab], inca,
                               &work[j1 - 1], kd1, &D[j1 - 1], kd1);

                        if (nr > 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                clartv(nr, &AB[kd - l + (j1 - kd1 + l - 1) * ldab], inca,
                                       &AB[kd - l + 1 + (j1 - kd1 + l - 1) * ldab], inca,
                                       &D[j1 - 1], &work[j1 - 1], kd1);
                            }
                        } else {
                            jend = j1 + kd1 * (nr - 1);
                            for (jinc = j1; jinc <= jend; jinc += kd1) {
                                crot(kdm1, &AB[kd - 1 + (jinc - kd - 1) * ldab], incx,
                                     &AB[kd + (jinc - kd - 1) * ldab], incx,
                                     D[jinc - 1], work[jinc - 1]);
                            }
                        }

                    }

                    if (k > 2) {
                        if (k <= n - i + 1) {

                            clartg(AB[k - 2 + (i - 1) * ldab],
                                   AB[k - 1 + (i - 1) * ldab],
                                   &D[i + k - 2], &work[i + k - 2], &temp);
                            AB[k - 2 + (i - 1) * ldab] = temp;

                            crot(k - 3, &AB[k - 3 + i * ldab], ldab - 1,
                                 &AB[k - 2 + i * ldab], ldab - 1,
                                 D[i + k - 2], work[i + k - 2]);
                        }
                        nr = nr + 1;
                        j1 = j1 - kdn - 1;
                    }

                    if (nr > 0)
                        clar2v(nr, &AB[0 + (j1 - 2) * ldab],
                               &AB[0 + (j1 - 1) * ldab],
                               &AB[1 + (j1 - 2) * ldab], inca,
                               &D[j1 - 1], &work[j1 - 1], kd1);

                    if (nr > 0) {
                        clacgv(nr, &work[j1 - 1], kd1);
                        if (nr > 2 * kd - 1) {
                            for (l = 1; l <= kd - 1; l++) {
                                if (j2 + l > n) {
                                    nrt = nr - 1;
                                } else {
                                    nrt = nr;
                                }
                                if (nrt > 0)
                                    clartv(nrt, &AB[l + 1 + (j1 - 2) * ldab], inca,
                                           &AB[l + (j1 - 1) * ldab], inca,
                                           &D[j1 - 1], &work[j1 - 1], kd1);
                            }
                        } else {
                            j1end = j1 + kd1 * (nr - 2);
                            if (j1end >= j1) {
                                for (j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                    crot(kdm1, &AB[2 + (j1inc - 2) * ldab], 1,
                                         &AB[1 + (j1inc - 1) * ldab], 1,
                                         D[j1inc - 1], work[j1inc - 1]);
                                }
                            }
                            lend = (kdm1 < n - j2) ? kdm1 : (n - j2);
                            last = j1end + kd1;
                            if (lend > 0)
                                crot(lend, &AB[2 + (last - 2) * ldab], 1,
                                     &AB[1 + (last - 1) * ldab], 1,
                                     D[last - 1], work[last - 1]);
                        }
                    }

                    if (wantq) {

                        if (initq) {

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
                                crot(nq, &Q[(iqb - 1) + (j - 2) * ldq], 1,
                                     &Q[(iqb - 1) + (j - 1) * ldq], 1,
                                     D[j - 1], work[j - 1]);
                            }
                        } else {

                            for (j = j1; j <= j2; j += kd1) {
                                crot(n, &Q[0 + (j - 2) * ldq], 1,
                                     &Q[0 + (j - 1) * ldq], 1,
                                     D[j - 1], work[j - 1]);
                            }
                        }
                    }

                    if (j2 + kdn > n) {

                        nr = nr - 1;
                        j2 = j2 - kdn - 1;
                    }

                    for (j = j1; j <= j2; j += kd1) {

                        work[j + kd - 1] = work[j - 1] * AB[kd + (j - 1) * ldab];
                        AB[kd + (j - 1) * ldab] = D[j - 1] * AB[kd + (j - 1) * ldab];
                    }
                }
            }
        }

        if (kd > 0) {

            for (i = 1; i <= n - 1; i++) {
                t = AB[1 + (i - 1) * ldab];
                abst = cabsf(t);
                AB[1 + (i - 1) * ldab] = abst;
                E[i - 1] = abst;
                if (abst != ZERO) {
                    t = t / abst;
                } else {
                    t = CONE;
                }
                if (i < n - 1)
                    AB[1 + i * ldab] = AB[1 + i * ldab] * t;
                if (wantq) {
                    cblas_cscal(n, &t, &Q[0 + i * ldq], 1);
                }
            }
        } else {

            for (i = 1; i <= n - 1; i++) {
                E[i - 1] = ZERO;
            }
        }

        for (i = 1; i <= n; i++) {
            D[i - 1] = crealf(AB[0 + (i - 1) * ldab]);
        }
    }
}
