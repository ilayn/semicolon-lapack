/**
 * @file sgbbrd.c
 * @brief SGBBRD reduces a real general band matrix to bidiagonal form.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/* Macro for min/max */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * SGBBRD reduces a real general m-by-n band matrix A to upper
 * bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
 *
 * The routine computes B, and optionally forms Q or P**T, or computes
 * Q**T*C for a given matrix C.
 *
 * @param[in]     vect  Specifies whether or not the matrices Q and P**T are to be
 *                      formed.
 *                      = 'N': do not form Q or P**T;
 *                      = 'Q': form Q only;
 *                      = 'P': form P**T only;
 *                      = 'B': form both.
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     ncc   The number of columns of the matrix C. ncc >= 0.
 * @param[in]     kl    The number of subdiagonals of the matrix A. kl >= 0.
 * @param[in]     ku    The number of superdiagonals of the matrix A. ku >= 0.
 * @param[in,out] AB    Double precision array, dimension (ldab, n).
 *                      On entry, the m-by-n band matrix A, stored in rows 0 to
 *                      kl+ku. The j-th column of A is stored in the j-th column of
 *                      the array AB as follows:
 *                      AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku)<=i<=min(m-1,j+kl).
 *                      On exit, A is overwritten by values generated during the
 *                      reduction.
 * @param[in]     ldab  The leading dimension of the array A. ldab >= kl+ku+1.
 * @param[out]    D     Double precision array, dimension (min(m,n)).
 *                      The diagonal elements of the bidiagonal matrix B.
 * @param[out]    E     Double precision array, dimension (min(m,n)-1).
 *                      The superdiagonal elements of the bidiagonal matrix B.
 * @param[out]    Q     Double precision array, dimension (ldq, m).
 *                      If vect = 'Q' or 'B', the m-by-m orthogonal matrix Q.
 *                      If vect = 'N' or 'P', the array Q is not referenced.
 * @param[in]     ldq   The leading dimension of the array Q.
 *                      ldq >= max(1,m) if vect = 'Q' or 'B'; ldq >= 1 otherwise.
 * @param[out]    PT    Double precision array, dimension (ldpt, n).
 *                      If vect = 'P' or 'B', the n-by-n orthogonal matrix P'.
 *                      If vect = 'N' or 'Q', the array PT is not referenced.
 * @param[in]     ldpt  The leading dimension of the array PT.
 *                      ldpt >= max(1,n) if vect = 'P' or 'B'; ldpt >= 1 otherwise.
 * @param[in,out] C     Double precision array, dimension (ldc, ncc).
 *                      On entry, an m-by-ncc matrix C.
 *                      On exit, C is overwritten by Q**T*C.
 *                      C is not referenced if ncc = 0.
 * @param[in]     ldc   The leading dimension of the array C.
 *                      ldc >= max(1,m) if ncc > 0; ldc >= 1 if ncc = 0.
 * @param[out]    work  Double precision array, dimension (2*max(m,n)).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgbbrd(const char* vect, const int m, const int n, const int ncc,
            const int kl, const int ku,
            f32* restrict AB, const int ldab,
            f32* restrict D, f32* restrict E,
            f32* restrict Q, const int ldq,
            f32* restrict PT, const int ldpt,
            f32* restrict C, const int ldc,
            f32* restrict work, int* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    /* Test the input parameters */
    int wantb = (vect[0] == 'B' || vect[0] == 'b');
    int wantq = (vect[0] == 'Q' || vect[0] == 'q') || wantb;
    int wantpt = (vect[0] == 'P' || vect[0] == 'p') || wantb;
    int wantc = (ncc > 0);
    int klu1 = kl + ku + 1;

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
        xerbla("SGBBRD", -(*info));
        return;
    }

    /* Initialize Q and P**T to the unit matrix, if needed */
    if (wantq) {
        slaset("Full", m, m, zero, one, Q, ldq);
    }
    if (wantpt) {
        slaset("Full", n, n, zero, one, PT, ldpt);
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    int minmn = MIN(m, n);

    if (kl + ku > 1) {
        /*
         * Reduce to upper bidiagonal form if ku > 0; if ku = 0, reduce
         * first to lower bidiagonal form and then transform to upper
         * bidiagonal
         */
        int ml0, mu0;
        if (ku > 0) {
            ml0 = 1;
            mu0 = 2;
        } else {
            ml0 = 2;
            mu0 = 1;
        }

        /*
         * Wherever possible, plane rotations are generated and applied in
         * vector operations of length nr over the index set j1:j2:kb1.
         *
         * The sines of the plane rotations are stored in work[0:mn-1]
         * and the cosines in work[mn:2*mn-1].
         */
        int mn = MAX(m, n);
        int klm = MIN(m - 1, kl);
        int kun = MIN(n - 1, ku);
        int kb = klm + kun;
        int kb1 = kb + 1;
        int inca = kb1 * ldab;
        int nr = 0;
        int j1 = klm + 1;   /* 0-indexed: Fortran klm+2 becomes klm+1 */
        int j2 = -kun;      /* 0-indexed: Fortran 1-kun becomes -kun */

        for (int i = 0; i < minmn; i++) {
            /* Reduce i-th column and i-th row of matrix to bidiagonal form */

            int ml = klm + 1;
            int mu = kun + 1;

            for (int kk = 0; kk < kb; kk++) {
                j1 += kb;
                j2 += kb;

                /*
                 * Generate plane rotations to annihilate nonzero elements
                 * which have been created below the band
                 */
                if (nr > 0) {
                    /* AB[klu1-1 + (j1-klm-1)*ldab] in 0-indexed */
                    slargv(nr, &AB[(klu1 - 1) + (j1 - klm - 1) * ldab], inca,
                           &work[j1], kb1, &work[mn + j1], kb1);
                }

                /* Apply plane rotations from the left */
                for (int l = 0; l < kb; l++) {
                    int nrt;
                    if (j2 - klm + l > n - 1) {
                        nrt = nr - 1;
                    } else {
                        nrt = nr;
                    }
                    if (nrt > 0) {
                        /* AB[klu1-l-2 + (j1-klm+l)*ldab] and AB[klu1-l-1 + (j1-klm+l)*ldab] */
                        slartv(nrt, &AB[(klu1 - l - 2) + (j1 - klm + l) * ldab], inca,
                               &AB[(klu1 - l - 1) + (j1 - klm + l) * ldab], inca,
                               &work[mn + j1], &work[j1], kb1);
                    }
                }

                if (ml > ml0) {
                    if (ml <= m - i) {
                        /*
                         * Generate plane rotation to annihilate a(i+ml-1,i)
                         * within the band, and apply rotation from the left
                         */
                        f32 ra;
                        /* AB[ku+ml-2 + i*ldab] and AB[ku+ml-1 + i*ldab] in 0-indexed */
                        slartg(AB[(ku + ml - 2) + i * ldab], AB[(ku + ml - 1) + i * ldab],
                               &work[mn + i + ml - 1], &work[i + ml - 1], &ra);
                        AB[(ku + ml - 2) + i * ldab] = ra;
                        if (i < n - 1) {
                            /* AB[ku+ml-3 + (i+1)*ldab] and AB[ku+ml-2 + (i+1)*ldab] */
                            cblas_srot(MIN(ku + ml - 2, n - i - 1),
                                       &AB[(ku + ml - 3) + (i + 1) * ldab], ldab - 1,
                                       &AB[(ku + ml - 2) + (i + 1) * ldab], ldab - 1,
                                       work[mn + i + ml - 1], work[i + ml - 1]);
                        }
                    }
                    nr++;
                    j1 -= kb1;
                }

                if (wantq) {
                    /* Accumulate product of plane rotations in Q */
                    for (int j = j1; j <= j2; j += kb1) {
                        /* Q columns j-1 and j (0-indexed: j-1 and j become j-1 and j) */
                        cblas_srot(m, &Q[(j - 1) * ldq], 1, &Q[j * ldq], 1,
                                   work[mn + j], work[j]);
                    }
                }

                if (wantc) {
                    /* Apply plane rotations to C */
                    for (int j = j1; j <= j2; j += kb1) {
                        /* C rows j-1 and j */
                        cblas_srot(ncc, &C[(j - 1)], ldc, &C[j], ldc,
                                   work[mn + j], work[j]);
                    }
                }

                if (j2 + kun > n - 1) {
                    /* Adjust j2 to keep within the bounds of the matrix */
                    nr--;
                    j2 -= kb1;
                }

                for (int j = j1; j <= j2; j += kb1) {
                    /*
                     * Create nonzero element a(j-1,j+ku) above the band
                     * and store it in work[0:n-1]
                     */
                    /* AB[0 + (j+kun)*ldab] in 0-indexed */
                    work[j + kun] = work[j] * AB[(j + kun) * ldab];
                    AB[(j + kun) * ldab] = work[mn + j] * AB[(j + kun) * ldab];
                }

                /*
                 * Generate plane rotations to annihilate nonzero elements
                 * which have been generated above the band
                 */
                if (nr > 0) {
                    /* AB[0 + (j1+kun-1)*ldab] in 0-indexed */
                    slargv(nr, &AB[(j1 + kun - 1) * ldab], inca,
                           &work[j1 + kun], kb1, &work[mn + j1 + kun], kb1);
                }

                /* Apply plane rotations from the right */
                for (int l = 0; l < kb; l++) {
                    int nrt;
                    if (j2 + l > m - 1) {
                        nrt = nr - 1;
                    } else {
                        nrt = nr;
                    }
                    if (nrt > 0) {
                        /* AB[l+1 + (j1+kun-1)*ldab] and AB[l + (j1+kun)*ldab] */
                        slartv(nrt, &AB[(l + 1) + (j1 + kun - 1) * ldab], inca,
                               &AB[l + (j1 + kun) * ldab], inca,
                               &work[mn + j1 + kun], &work[j1 + kun], kb1);
                    }
                }

                if (ml == ml0 && mu > mu0) {
                    if (mu <= n - i) {
                        /*
                         * Generate plane rotation to annihilate a(i,i+mu-1)
                         * within the band, and apply rotation from the right
                         */
                        f32 ra;
                        /* AB[ku-mu+2 + (i+mu-2)*ldab] and AB[ku-mu+1 + (i+mu-1)*ldab] */
                        slartg(AB[(ku - mu + 2) + (i + mu - 2) * ldab],
                               AB[(ku - mu + 1) + (i + mu - 1) * ldab],
                               &work[mn + i + mu - 1], &work[i + mu - 1], &ra);
                        AB[(ku - mu + 2) + (i + mu - 2) * ldab] = ra;
                        /* AB[ku-mu+3 + (i+mu-2)*ldab] and AB[ku-mu+2 + (i+mu-1)*ldab] */
                        cblas_srot(MIN(kl + mu - 2, m - i - 1),
                                   &AB[(ku - mu + 3) + (i + mu - 2) * ldab], 1,
                                   &AB[(ku - mu + 2) + (i + mu - 1) * ldab], 1,
                                   work[mn + i + mu - 1], work[i + mu - 1]);
                    }
                    nr++;
                    j1 -= kb1;
                }

                if (wantpt) {
                    /* Accumulate product of plane rotations in P**T */
                    for (int j = j1; j <= j2; j += kb1) {
                        /* PT rows j+kun-1 and j+kun (0-indexed) */
                        cblas_srot(n, &PT[(j + kun - 1) * ldpt], 1,
                                   &PT[(j + kun) * ldpt], 1,
                                   work[mn + j + kun], work[j + kun]);
                    }
                }

                if (j2 + kb > m - 1) {
                    /* Adjust j2 to keep within the bounds of the matrix */
                    nr--;
                    j2 -= kb1;
                }

                for (int j = j1; j <= j2; j += kb1) {
                    /*
                     * Create nonzero element a(j+kl+ku,j+ku-1) below the
                     * band and store it in work[0:n-1]
                     */
                    /* AB[klu1-1 + (j+kun)*ldab] in 0-indexed */
                    work[j + kb] = work[j + kun] * AB[(klu1 - 1) + (j + kun) * ldab];
                    AB[(klu1 - 1) + (j + kun) * ldab] = work[mn + j + kun] * AB[(klu1 - 1) + (j + kun) * ldab];
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
        /*
         * A has been reduced to lower bidiagonal form
         *
         * Transform lower bidiagonal form to upper bidiagonal by applying
         * plane rotations from the left, storing diagonal elements in D
         * and off-diagonal elements in E
         */
        for (int i = 0; i < MIN(m - 1, n); i++) {
            f32 rc, rs, ra;
            /* AB[0 + i*ldab] and AB[1 + i*ldab] */
            slartg(AB[i * ldab], AB[1 + i * ldab], &rc, &rs, &ra);
            D[i] = ra;
            if (i < n - 1) {
                E[i] = rs * AB[(i + 1) * ldab];
                AB[(i + 1) * ldab] = rc * AB[(i + 1) * ldab];
            }
            if (wantq) {
                cblas_srot(m, &Q[i * ldq], 1, &Q[(i + 1) * ldq], 1, rc, rs);
            }
            if (wantc) {
                cblas_srot(ncc, &C[i], ldc, &C[i + 1], ldc, rc, rs);
            }
        }
        if (m <= n) {
            D[m - 1] = AB[(m - 1) * ldab];
        }
    } else if (ku > 0) {
        /*
         * A has been reduced to upper bidiagonal form
         */
        if (m < n) {
            /*
             * Annihilate a(m,m+1) by applying plane rotations from the
             * right, storing diagonal elements in D and off-diagonal
             * elements in E
             */
            f32 rb = AB[(ku - 1) + m * ldab];  /* AB[ku-1 + m*ldab] = A(m-1, m) in 0-indexed */
            for (int i = m - 1; i >= 0; i--) {
                f32 rc, rs, ra;
                /* AB[ku + i*ldab] = diagonal element */
                slartg(AB[ku + i * ldab], rb, &rc, &rs, &ra);
                D[i] = ra;
                if (i > 0) {
                    rb = -rs * AB[(ku - 1) + i * ldab];
                    E[i - 1] = rc * AB[(ku - 1) + i * ldab];
                }
                if (wantpt) {
                    cblas_srot(n, &PT[i * ldpt], 1, &PT[m * ldpt], 1, rc, rs);
                }
            }
        } else {
            /*
             * Copy off-diagonal elements to E and diagonal elements to D
             */
            for (int i = 0; i < minmn - 1; i++) {
                E[i] = AB[(ku - 1) + (i + 1) * ldab];
            }
            for (int i = 0; i < minmn; i++) {
                D[i] = AB[ku + i * ldab];
            }
        }
    } else {
        /*
         * A is diagonal. Set elements of E to zero and copy diagonal
         * elements to D.
         */
        for (int i = 0; i < minmn - 1; i++) {
            E[i] = zero;
        }
        for (int i = 0; i < minmn; i++) {
            D[i] = AB[i * ldab];
        }
    }
}
