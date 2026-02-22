/**
 * @file cgbbrd.c
 * @brief CGBBRD reduces a complex general band matrix to real upper bidiagonal form.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * CGBBRD reduces a complex general m-by-n band matrix A to real upper
 * bidiagonal form B by a unitary transformation: Q**H * A * P = B.
 *
 * The routine computes B, and optionally forms Q or P**H, or computes
 * Q**H*C for a given matrix C.
 *
 * @param[in]     vect  Specifies whether or not the matrices Q and P**H are to be
 *                      formed.
 *                      = 'N': do not form Q or P**H;
 *                      = 'Q': form Q only;
 *                      = 'P': form P**H only;
 *                      = 'B': form both.
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     ncc   The number of columns of the matrix C. ncc >= 0.
 * @param[in]     kl    The number of subdiagonals of the matrix A. kl >= 0.
 * @param[in]     ku    The number of superdiagonals of the matrix A. ku >= 0.
 * @param[in,out] AB    Complex array, dimension (ldab, n).
 *                      On entry, the m-by-n band matrix A, stored in rows 0 to
 *                      kl+ku. The j-th column of A is stored in the j-th column of
 *                      the array AB as follows:
 *                      AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku)<=i<=min(m-1,j+kl).
 *                      On exit, A is overwritten by values generated during the
 *                      reduction.
 * @param[in]     ldab  The leading dimension of the array A. ldab >= kl+ku+1.
 * @param[out]    D     Single precision array, dimension (min(m,n)).
 *                      The diagonal elements of the bidiagonal matrix B.
 * @param[out]    E     Single precision array, dimension (min(m,n)-1).
 *                      The superdiagonal elements of the bidiagonal matrix B.
 * @param[out]    Q     Complex array, dimension (ldq, m).
 *                      If vect = 'Q' or 'B', the m-by-m unitary matrix Q.
 *                      If vect = 'N' or 'P', the array Q is not referenced.
 * @param[in]     ldq   The leading dimension of the array Q.
 *                      ldq >= max(1,m) if vect = 'Q' or 'B'; ldq >= 1 otherwise.
 * @param[out]    PT    Complex array, dimension (ldpt, n).
 *                      If vect = 'P' or 'B', the n-by-n unitary matrix P'.
 *                      If vect = 'N' or 'Q', the array PT is not referenced.
 * @param[in]     ldpt  The leading dimension of the array PT.
 *                      ldpt >= max(1,n) if vect = 'P' or 'B'; ldpt >= 1 otherwise.
 * @param[in,out] C     Complex array, dimension (ldc, ncc).
 *                      On entry, an m-by-ncc matrix C.
 *                      On exit, C is overwritten by Q**H*C.
 *                      C is not referenced if ncc = 0.
 * @param[in]     ldc   The leading dimension of the array C.
 *                      ldc >= max(1,m) if ncc > 0; ldc >= 1 if ncc = 0.
 * @param[out]    work  Complex array, dimension (max(m,n)).
 * @param[out]    rwork Single precision array, dimension (max(m,n)).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgbbrd(const char* vect, const INT m, const INT n, const INT ncc,
            const INT kl, const INT ku,
            c64* restrict AB, const INT ldab,
            f32* restrict D, f32* restrict E,
            c64* restrict Q, const INT ldq,
            c64* restrict PT, const INT ldpt,
            c64* restrict C, const INT ldc,
            c64* restrict work,
            f32* restrict rwork, INT* info)
{
    const f32 ZERO = 0.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT wantb = (vect[0] == 'B' || vect[0] == 'b');
    INT wantq = (vect[0] == 'Q' || vect[0] == 'q') || wantb;
    INT wantpt = (vect[0] == 'P' || vect[0] == 'p') || wantb;
    INT wantc = (ncc > 0);
    INT klu1 = kl + ku + 1;

    *info = 0;
    if (!wantq && !wantpt && !(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ncc < 0) {
        *info = -4;
    } else if (kl < 0) {
        *info = -5;
    } else if (ku < 0) {
        *info = -6;
    } else if (ldab < klu1) {
        *info = -8;
    } else if (ldq < 1 || (wantq && ldq < MAX(1, m))) {
        *info = -12;
    } else if (ldpt < 1 || (wantpt && ldpt < MAX(1, n))) {
        *info = -14;
    } else if (ldc < 1 || (wantc && ldc < MAX(1, m))) {
        *info = -16;
    }

    if (*info != 0) {
        xerbla("CGBBRD", -(*info));
        return;
    }

    if (wantq) {
        claset("Full", m, m, CZERO, CONE, Q, ldq);
    }
    if (wantpt) {
        claset("Full", n, n, CZERO, CONE, PT, ldpt);
    }

    if (m == 0 || n == 0) {
        return;
    }

    INT minmn = MIN(m, n);

    if (kl + ku > 1) {
        INT ml0, mu0;
        if (ku > 0) {
            ml0 = 1;
            mu0 = 2;
        } else {
            ml0 = 2;
            mu0 = 1;
        }

        INT klm = MIN(m - 1, kl);
        INT kun = MIN(n - 1, ku);
        INT kb = klm + kun;
        INT kb1 = kb + 1;
        INT inca = kb1 * ldab;
        INT nr = 0;
        INT j1 = klm + 1;
        INT j2 = -kun;

        for (INT i = 0; i < minmn; i++) {

            INT ml = klm + 1;
            INT mu = kun + 1;

            for (INT kk = 0; kk < kb; kk++) {
                j1 += kb;
                j2 += kb;

                if (nr > 0) {
                    clargv(nr, &AB[(klu1 - 1) + (j1 - klm - 1) * ldab], inca,
                           &work[j1], kb1, &rwork[j1], kb1);
                }

                for (INT l = 0; l < kb; l++) {
                    INT nrt;
                    if (j2 - klm + l > n - 1) {
                        nrt = nr - 1;
                    } else {
                        nrt = nr;
                    }
                    if (nrt > 0) {
                        clartv(nrt, &AB[(klu1 - l - 2) + (j1 - klm + l) * ldab], inca,
                               &AB[(klu1 - l - 1) + (j1 - klm + l) * ldab], inca,
                               &rwork[j1], &work[j1], kb1);
                    }
                }

                if (ml > ml0) {
                    if (ml <= m - i) {
                        c64 ra;
                        clartg(AB[(ku + ml - 2) + i * ldab], AB[(ku + ml - 1) + i * ldab],
                               &rwork[i + ml - 1], &work[i + ml - 1], &ra);
                        AB[(ku + ml - 2) + i * ldab] = ra;
                        if (i < n - 1) {
                            crot(MIN(ku + ml - 2, n - i - 1),
                                 &AB[(ku + ml - 3) + (i + 1) * ldab], ldab - 1,
                                 &AB[(ku + ml - 2) + (i + 1) * ldab], ldab - 1,
                                 rwork[i + ml - 1], work[i + ml - 1]);
                        }
                    }
                    nr++;
                    j1 -= kb1;
                }

                if (wantq) {
                    for (INT j = j1; j <= j2; j += kb1) {
                        crot(m, &Q[(j - 1) * ldq], 1, &Q[j * ldq], 1,
                             rwork[j], conjf(work[j]));
                    }
                }

                if (wantc) {
                    for (INT j = j1; j <= j2; j += kb1) {
                        crot(ncc, &C[(j - 1)], ldc, &C[j], ldc,
                             rwork[j], work[j]);
                    }
                }

                if (j2 + kun > n - 1) {
                    nr--;
                    j2 -= kb1;
                }

                for (INT j = j1; j <= j2; j += kb1) {
                    work[j + kun] = work[j] * AB[(j + kun) * ldab];
                    AB[(j + kun) * ldab] = rwork[j] * AB[(j + kun) * ldab];
                }

                if (nr > 0) {
                    clargv(nr, &AB[(j1 + kun - 1) * ldab], inca,
                           &work[j1 + kun], kb1, &rwork[j1 + kun], kb1);
                }

                for (INT l = 0; l < kb; l++) {
                    INT nrt;
                    if (j2 + l > m - 1) {
                        nrt = nr - 1;
                    } else {
                        nrt = nr;
                    }
                    if (nrt > 0) {
                        clartv(nrt, &AB[(l + 1) + (j1 + kun - 1) * ldab], inca,
                               &AB[l + (j1 + kun) * ldab], inca,
                               &rwork[j1 + kun], &work[j1 + kun], kb1);
                    }
                }

                if (ml == ml0 && mu > mu0) {
                    if (mu <= n - i) {
                        c64 ra;
                        clartg(AB[(ku - mu + 2) + (i + mu - 2) * ldab],
                               AB[(ku - mu + 1) + (i + mu - 1) * ldab],
                               &rwork[i + mu - 1], &work[i + mu - 1], &ra);
                        AB[(ku - mu + 2) + (i + mu - 2) * ldab] = ra;
                        crot(MIN(kl + mu - 2, m - i - 1),
                             &AB[(ku - mu + 3) + (i + mu - 2) * ldab], 1,
                             &AB[(ku - mu + 2) + (i + mu - 1) * ldab], 1,
                             rwork[i + mu - 1], work[i + mu - 1]);
                    }
                    nr++;
                    j1 -= kb1;
                }

                if (wantpt) {
                    for (INT j = j1; j <= j2; j += kb1) {
                        crot(n, &PT[j + kun - 1], ldpt,
                             &PT[j + kun], ldpt,
                             rwork[j + kun], conjf(work[j + kun]));
                    }
                }

                if (j2 + kb > m - 1) {
                    nr--;
                    j2 -= kb1;
                }

                for (INT j = j1; j <= j2; j += kb1) {
                    work[j + kb] = work[j + kun] * AB[(klu1 - 1) + (j + kun) * ldab];
                    AB[(klu1 - 1) + (j + kun) * ldab] = rwork[j + kun] * AB[(klu1 - 1) + (j + kun) * ldab];
                }

                if (ml > ml0) {
                    ml--;
                } else {
                    mu--;
                }
            }
        }
    }

    if (ku == 0 && kl > 0) {
        for (INT i = 0; i < MIN(m - 1, n); i++) {
            f32 rc;
            c64 rs, ra;
            clartg(AB[i * ldab], AB[1 + i * ldab], &rc, &rs, &ra);
            AB[i * ldab] = ra;
            if (i < n - 1) {
                AB[1 + i * ldab] = rs * AB[(i + 1) * ldab];
                AB[(i + 1) * ldab] = rc * AB[(i + 1) * ldab];
            }
            if (wantq) {
                crot(m, &Q[i * ldq], 1, &Q[(i + 1) * ldq], 1, rc, conjf(rs));
            }
            if (wantc) {
                crot(ncc, &C[i], ldc, &C[i + 1], ldc, rc, rs);
            }
        }
    } else if (ku > 0 && m < n) {
        c64 rb = AB[(ku - 1) + m * ldab];
        for (INT i = m - 1; i >= 0; i--) {
            f32 rc;
            c64 rs, ra;
            clartg(AB[ku + i * ldab], rb, &rc, &rs, &ra);
            AB[ku + i * ldab] = ra;
            if (i > 0) {
                rb = -conjf(rs) * AB[(ku - 1) + i * ldab];
                AB[(ku - 1) + i * ldab] = rc * AB[(ku - 1) + i * ldab];
            }
            if (wantpt) {
                crot(n, &PT[i], ldpt, &PT[m], ldpt, rc, conjf(rs));
            }
        }
    }

    c64 t = AB[ku];
    for (INT i = 0; i < minmn; i++) {
        f32 abst = cabsf(t);
        D[i] = abst;
        if (abst != ZERO) {
            t = t / abst;
        } else {
            t = CONE;
        }
        if (wantq) {
            cblas_cscal(m, &t, &Q[i * ldq], 1);
        }
        if (wantc) {
            c64 tc = conjf(t);
            cblas_cscal(ncc, &tc, &C[i], ldc);
        }
        if (i < minmn - 1) {
            if (ku == 0 && kl == 0) {
                E[i] = ZERO;
                t = AB[(i + 1) * ldab];
            } else {
                if (ku == 0) {
                    t = AB[1 + i * ldab] * conjf(t);
                } else {
                    t = AB[(ku - 1) + (i + 1) * ldab] * conjf(t);
                }
                abst = cabsf(t);
                E[i] = abst;
                if (abst != ZERO) {
                    t = t / abst;
                } else {
                    t = CONE;
                }
                if (wantpt) {
                    cblas_cscal(n, &t, &PT[(i + 1) * ldpt], 1);
                }
                t = AB[ku + (i + 1) * ldab] * conjf(t);
            }
        }
    }
}
