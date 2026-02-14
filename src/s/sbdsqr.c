/**
 * @file sbdsqr.c
 * @brief SBDSQR computes the SVD of a real bidiagonal matrix using the
 *        implicit zero-shift QR algorithm.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

/**
 * SBDSQR computes the singular values and, optionally, the right and/or
 * left singular vectors from the singular value decomposition (SVD) of
 * a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
 * zero-shift QR algorithm.  The SVD of B has the form
 *
 *    B = Q * S * P**T
 *
 * where S is the diagonal matrix of singular values, Q is an orthogonal
 * matrix of left singular vectors, and P is an orthogonal matrix of
 * right singular vectors.  If left singular vectors are requested, this
 * subroutine actually returns U*Q instead of Q, and, if right singular
 * vectors are requested, this subroutine returns P**T*VT instead of
 * P**T, for given real input matrices U and VT.  When U and VT are the
 * orthogonal matrices that reduce a general matrix A to bidiagonal
 * form:  A = U*B*VT, as computed by SGEBRD, then
 *
 *    A = (U*Q) * S * (P**T*VT)
 *
 * is the SVD of A.  Optionally, the subroutine may also compute Q**T*C
 * for a given real input matrix C.
 *
 * @param[in]     uplo  = 'U': B is upper bidiagonal;
 *                      = 'L': B is lower bidiagonal.
 * @param[in]     n     The order of the matrix B. n >= 0.
 * @param[in]     ncvt  The number of columns of the matrix VT. ncvt >= 0.
 * @param[in]     nru   The number of rows of the matrix U. nru >= 0.
 * @param[in]     ncc   The number of columns of the matrix C. ncc >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the n diagonal elements of the bidiagonal matrix B.
 *                      On exit, if info=0, the singular values of B in decreasing
 *                      order.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the N-1 offdiagonal elements of the bidiagonal
 *                      matrix B.
 *                      On exit, if info = 0, E is destroyed; if info > 0, D and E
 *                      will contain the diagonal and superdiagonal elements of a
 *                      bidiagonal matrix orthogonally equivalent to the one given
 *                      as input.
 * @param[in,out] VT    Double precision array, dimension (ldvt, ncvt).
 *                      On entry, an N-by-NCVT matrix VT.
 *                      On exit, VT is overwritten by P**T * VT.
 *                      Not referenced if ncvt = 0.
 * @param[in]     ldvt  The leading dimension of the array VT.
 *                      ldvt >= max(1,n) if ncvt > 0; ldvt >= 1 if ncvt = 0.
 * @param[in,out] U     Double precision array, dimension (ldu, n).
 *                      On entry, an NRU-by-N matrix U.
 *                      On exit, U is overwritten by U * Q.
 *                      Not referenced if nru = 0.
 * @param[in]     ldu   The leading dimension of the array U. ldu >= max(1,nru).
 * @param[in,out] C     Double precision array, dimension (ldc, ncc).
 *                      On entry, an N-by-NCC matrix C.
 *                      On exit, C is overwritten by Q**T * C.
 *                      Not referenced if ncc = 0.
 * @param[in]     ldc   The leading dimension of the array C.
 *                      ldc >= max(1,n) if ncc > 0; ldc >=1 if ncc = 0.
 * @param[out]    work  Double precision array, dimension (lwork).
 *                      lwork = 4*n if ncvt = nru = ncc = 0,
 *                      lwork = 4*(n-1) otherwise.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm did not converge; D and E contain the
 *                           elements of a bidiagonal matrix which is orthogonally
 *                           similar to the input matrix B; if info = i, i
 *                           elements of E have not converged to zero.
 */
void sbdsqr(const char* uplo, const int n, const int ncvt, const int nru,
            const int ncc, f32* restrict D, f32* restrict E,
            f32* restrict VT, const int ldvt,
            f32* restrict U, const int ldu,
            f32* restrict C, const int ldc,
            f32* restrict work, int* info)
{
    /* Constants from LAPACK */
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 NEGONE = -1.0f;
    const f32 HNDRTH = 0.01f;
    const f32 TEN = 10.0f;
    const f32 HNDRD = 100.0f;
    const f32 MEIGTH = -0.125f;
    const int MAXITR = 6;

    /* Local variables */
    int lower, rotate;
    int i, idir, isub, iter, iterdivn, j, ll, lll, m;
    int maxitdivn, nm1, nm12, nm13, oldll, oldm;
    f32 abse, abss, cosl, cosr, cs, eps, f, g, h, mu;
    f32 oldcs, oldsn, r, shift, sigmn, sigmx, sinl, sinr;
    f32 sll, smax, smin, sminoa, sn, thresh, tol, tolmul, unfl;

    /* Test the input parameters */
    *info = 0;
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !lower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ncvt < 0) {
        *info = -3;
    } else if (nru < 0) {
        *info = -4;
    } else if (ncc < 0) {
        *info = -5;
    } else if ((ncvt == 0 && ldvt < 1) ||
               (ncvt > 0 && ldvt < ((1 > n) ? 1 : n))) {
        *info = -9;
    } else if (ldu < ((1 > nru) ? 1 : nru)) {
        *info = -11;
    } else if ((ncc == 0 && ldc < 1) ||
               (ncc > 0 && ldc < ((1 > n) ? 1 : n))) {
        *info = -13;
    }

    if (*info != 0) {
        xerbla("SBDSQR", -(*info));
        return;
    }
    if (n == 0) return;
    if (n == 1) goto L160;

    /* ROTATE is true if any singular vectors desired, false otherwise */
    rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);

    /* If no singular vectors desired, use qd algorithm */
    if (!rotate) {
        slasq1(n, D, E, work, info);

        /* If INFO equals 2, dqds didn't finish, try to finish */
        if (*info != 2) return;
        *info = 0;
    }

    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;

    /* Get machine constants */
    eps = slamch("Epsilon");
    unfl = slamch("Safe minimum");

    /* If matrix lower bidiagonal, rotate to be upper bidiagonal
     * by applying Givens rotations on the left */
    if (lower) {
        for (i = 0; i < n - 1; i++) {
            slartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            work[i] = cs;
            work[nm1 + i] = sn;
        }

        /* Update singular vectors if desired */
        if (nru > 0) {
            slasr("R", "V", "F", nru, n, work, &work[nm1], U, ldu);
        }
        if (ncc > 0) {
            slasr("L", "V", "F", n, ncc, work, &work[nm1], C, ldc);
        }
    }

    /* Compute singular values to relative accuracy TOL
     * (By setting TOL to be negative, algorithm will compute
     * singular values to absolute accuracy ABS(TOL)*norm(input matrix)) */
    tolmul = (TEN > (HNDRD < powf(eps, MEIGTH) ? HNDRD : powf(eps, MEIGTH)))
             ? TEN
             : (HNDRD < powf(eps, MEIGTH) ? HNDRD : powf(eps, MEIGTH));
    tol = tolmul * eps;

    /* Compute approximate maximum, minimum singular values */
    smax = ZERO;
    for (i = 0; i < n; i++) {
        smax = (smax > fabsf(D[i])) ? smax : fabsf(D[i]);
    }
    for (i = 0; i < n - 1; i++) {
        smax = (smax > fabsf(E[i])) ? smax : fabsf(E[i]);
    }

    smin = ZERO;
    if (tol >= ZERO) {
        /* Relative accuracy desired */
        sminoa = fabsf(D[0]);
        if (sminoa == ZERO) goto L50;
        mu = sminoa;
        for (i = 1; i < n; i++) {
            mu = fabsf(D[i]) * (mu / (mu + fabsf(E[i - 1])));
            sminoa = (sminoa < mu) ? sminoa : mu;
            if (sminoa == ZERO) goto L50;
        }
    L50:
        sminoa = sminoa / sqrtf((f32)n);
        thresh = (tol * sminoa > MAXITR * (n * (n * unfl)))
                 ? tol * sminoa
                 : MAXITR * (n * (n * unfl));
    } else {
        /* Absolute accuracy desired */
        thresh = (fabsf(tol) * smax > MAXITR * (n * (n * unfl)))
                 ? fabsf(tol) * smax
                 : MAXITR * (n * (n * unfl));
    }

    /* Prepare for main iteration loop for the singular values
     * (MAXITDIVN is the maximum number of passes through the inner
     * loop permitted before nonconvergence signalled.) */
    maxitdivn = MAXITR * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;

    /* M points to last element of unconverged part of matrix */
    m = n;

    /* Begin main iteration loop */
L60:
    /* Check for convergence or exceeding iteration count */
    if (m <= 1) goto L160;

    if (iter >= n) {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if (iterdivn >= maxitdivn) goto L200;
    }

    /* Find diagonal block of matrix to work on */
    if (tol < ZERO && fabsf(D[m - 1]) <= thresh) {
        D[m - 1] = ZERO;
    }
    smax = fabsf(D[m - 1]);

    for (lll = 1; lll <= m - 1; lll++) {
        ll = m - lll - 1;  /* 0-based: ll = m - lll - 1 maps to Fortran LL = M - LLL */
        abss = fabsf(D[ll]);
        abse = fabsf(E[ll]);
        if (tol < ZERO && abss <= thresh) {
            D[ll] = ZERO;
        }
        if (abse <= thresh) goto L80;
        smax = (smax > abss) ? smax : abss;
        smax = (smax > abse) ? smax : abse;
    }
    ll = -1;  /* Fortran LL = 0, so 0-based ll = -1 means we use ll+1 = 0 */
    goto L90;

L80:
    E[ll] = ZERO;

    /* Matrix splits since E(LL) = 0 */
    if (ll == m - 2) {
        /* Convergence of bottom singular value, return to top of loop */
        m = m - 1;
        goto L60;
    }

L90:
    ll = ll + 1;  /* Now ll is 0-based index of first element in block */

    /* E(LL) through E(M-1) are nonzero, E(LL-1) is zero (or LL=0) */
    if (ll == m - 2) {
        /* 2 by 2 block, handle separately */
        slasv2(D[m - 2], E[m - 2], D[m - 1], &sigmn, &sigmx, &sinr,
               &cosr, &sinl, &cosl);
        D[m - 2] = sigmx;
        E[m - 2] = ZERO;
        D[m - 1] = sigmn;

        /* Compute singular vectors, if desired */
        if (ncvt > 0) {
            cblas_srot(ncvt, &VT[m - 2], ldvt, &VT[m - 1], ldvt,
                       cosr, sinr);
        }
        if (nru > 0) {
            cblas_srot(nru, &U[(m - 2) * ldu], 1, &U[(m - 1) * ldu], 1, cosl, sinl);
        }
        if (ncc > 0) {
            cblas_srot(ncc, &C[m - 2], ldc, &C[m - 1], ldc, cosl, sinl);
        }
        m = m - 2;
        goto L60;
    }

    /* If working on new submatrix, choose shift direction
     * (from larger end diagonal element towards smaller) */
    if (ll > oldm || m < oldll) {
        if (fabsf(D[ll]) >= fabsf(D[m - 1])) {
            /* Chase bulge from top (big end) to bottom (small end) */
            idir = 1;
        } else {
            /* Chase bulge from bottom (big end) to top (small end) */
            idir = 2;
        }
    }

    /* Apply convergence tests */
    if (idir == 1) {
        /* Run convergence test in forward direction
         * First apply standard test to bottom of matrix */
        if (fabsf(E[m - 2]) <= fabsf(tol) * fabsf(D[m - 1]) ||
            (tol < ZERO && fabsf(E[m - 2]) <= thresh)) {
            E[m - 2] = ZERO;
            goto L60;
        }

        if (tol >= ZERO) {
            /* If relative accuracy desired,
             * apply convergence criterion forward */
            mu = fabsf(D[ll]);
            smin = mu;
            for (lll = ll; lll <= m - 2; lll++) {
                if (fabsf(E[lll]) <= tol * mu) {
                    E[lll] = ZERO;
                    goto L60;
                }
                mu = fabsf(D[lll + 1]) * (mu / (mu + fabsf(E[lll])));
                smin = (smin < mu) ? smin : mu;
            }
        }
    } else {
        /* Run convergence test in backward direction
         * First apply standard test to top of matrix */
        if (fabsf(E[ll]) <= fabsf(tol) * fabsf(D[ll]) ||
            (tol < ZERO && fabsf(E[ll]) <= thresh)) {
            E[ll] = ZERO;
            goto L60;
        }

        if (tol >= ZERO) {
            /* If relative accuracy desired,
             * apply convergence criterion backward */
            mu = fabsf(D[m - 1]);
            smin = mu;
            for (lll = m - 2; lll >= ll; lll--) {
                if (fabsf(E[lll]) <= tol * mu) {
                    E[lll] = ZERO;
                    goto L60;
                }
                mu = fabsf(D[lll]) * (mu / (mu + fabsf(E[lll])));
                smin = (smin < mu) ? smin : mu;
            }
        }
    }
    oldll = ll;
    oldm = m;

    /* Compute shift.  First, test if shifting would ruin relative
     * accuracy, and if so set the shift to zero. */
    if (tol >= ZERO && n * tol * (smin / smax) <= (eps > HNDRTH * tol ? eps : HNDRTH * tol)) {
        /* Use a zero shift to avoid loss of relative accuracy */
        shift = ZERO;
    } else {
        /* Compute the shift from 2-by-2 block at end of matrix */
        if (idir == 1) {
            sll = fabsf(D[ll]);
            slas2(D[m - 2], E[m - 2], D[m - 1], &shift, &r);
        } else {
            sll = fabsf(D[m - 1]);
            slas2(D[ll], E[ll], D[ll + 1], &shift, &r);
        }

        /* Test if shift negligible, and if so set to zero */
        if (sll > ZERO) {
            if ((shift / sll) * (shift / sll) < eps) {
                shift = ZERO;
            }
        }
    }

    /* Increment iteration count */
    iter = iter + m - ll;

    /* If SHIFT = 0, do simplified QR iteration */
    if (shift == ZERO) {
        if (idir == 1) {
            /* Chase bulge from top to bottom
             * Save cosines and sines for later singular vector updates */
            cs = ONE;
            oldcs = ONE;
            for (i = ll; i <= m - 2; i++) {
                slartg(D[i] * cs, E[i], &cs, &sn, &r);
                if (i > ll) {
                    E[i - 1] = oldsn * r;
                }
                slartg(oldcs * r, D[i + 1] * sn, &oldcs, &oldsn, &D[i]);
                work[i - ll] = cs;
                work[i - ll + nm1] = sn;
                work[i - ll + nm12] = oldcs;
                work[i - ll + nm13] = oldsn;
            }
            h = D[m - 1] * cs;
            D[m - 1] = h * oldcs;
            E[m - 2] = h * oldsn;

            /* Update singular vectors */
            if (ncvt > 0) {
                slasr("L", "V", "F", m - ll, ncvt, work,
                      &work[nm1], &VT[ll], ldvt);
            }
            if (nru > 0) {
                slasr("R", "V", "F", nru, m - ll,
                      &work[nm12], &work[nm13], &U[ll * ldu], ldu);
            }
            if (ncc > 0) {
                slasr("L", "V", "F", m - ll, ncc,
                      &work[nm12], &work[nm13], &C[ll], ldc);
            }

            /* Test convergence */
            if (fabsf(E[m - 2]) <= thresh) {
                E[m - 2] = ZERO;
            }
        } else {
            /* Chase bulge from bottom to top
             * Save cosines and sines for later singular vector updates */
            cs = ONE;
            oldcs = ONE;
            for (i = m - 1; i >= ll + 1; i--) {
                slartg(D[i] * cs, E[i - 1], &cs, &sn, &r);
                if (i < m - 1) {
                    E[i] = oldsn * r;
                }
                slartg(oldcs * r, D[i - 1] * sn, &oldcs, &oldsn, &D[i]);
                work[i - ll - 1] = cs;
                work[i - ll - 1 + nm1] = -sn;
                work[i - ll - 1 + nm12] = oldcs;
                work[i - ll - 1 + nm13] = -oldsn;
            }
            h = D[ll] * cs;
            D[ll] = h * oldcs;
            E[ll] = h * oldsn;

            /* Update singular vectors */
            if (ncvt > 0) {
                slasr("L", "V", "B", m - ll, ncvt,
                      &work[nm12], &work[nm13], &VT[ll], ldvt);
            }
            if (nru > 0) {
                slasr("R", "V", "B", nru, m - ll, work,
                      &work[nm1], &U[ll * ldu], ldu);
            }
            if (ncc > 0) {
                slasr("L", "V", "B", m - ll, ncc, work,
                      &work[nm1], &C[ll], ldc);
            }

            /* Test convergence */
            if (fabsf(E[ll]) <= thresh) {
                E[ll] = ZERO;
            }
        }
    } else {
        /* Use nonzero shift */
        if (idir == 1) {
            /* Chase bulge from top to bottom
             * Save cosines and sines for later singular vector updates */
            f = (fabsf(D[ll]) - shift) *
                ((D[ll] >= ZERO ? ONE : NEGONE) + shift / D[ll]);
            g = E[ll];
            for (i = ll; i <= m - 2; i++) {
                slartg(f, g, &cosr, &sinr, &r);
                if (i > ll) {
                    E[i - 1] = r;
                }
                f = cosr * D[i] + sinr * E[i];
                E[i] = cosr * E[i] - sinr * D[i];
                g = sinr * D[i + 1];
                D[i + 1] = cosr * D[i + 1];
                slartg(f, g, &cosl, &sinl, &r);
                D[i] = r;
                f = cosl * E[i] + sinl * D[i + 1];
                D[i + 1] = cosl * D[i + 1] - sinl * E[i];
                if (i < m - 2) {
                    g = sinl * E[i + 1];
                    E[i + 1] = cosl * E[i + 1];
                }
                work[i - ll] = cosr;
                work[i - ll + nm1] = sinr;
                work[i - ll + nm12] = cosl;
                work[i - ll + nm13] = sinl;
            }
            E[m - 2] = f;

            /* Update singular vectors */
            if (ncvt > 0) {
                slasr("L", "V", "F", m - ll, ncvt, work,
                      &work[nm1], &VT[ll], ldvt);
            }
            if (nru > 0) {
                slasr("R", "V", "F", nru, m - ll,
                      &work[nm12], &work[nm13], &U[ll * ldu], ldu);
            }
            if (ncc > 0) {
                slasr("L", "V", "F", m - ll, ncc,
                      &work[nm12], &work[nm13], &C[ll], ldc);
            }

            /* Test convergence */
            if (fabsf(E[m - 2]) <= thresh) {
                E[m - 2] = ZERO;
            }
        } else {
            /* Chase bulge from bottom to top
             * Save cosines and sines for later singular vector updates */
            f = (fabsf(D[m - 1]) - shift) *
                ((D[m - 1] >= ZERO ? ONE : NEGONE) + shift / D[m - 1]);
            g = E[m - 2];
            for (i = m - 1; i >= ll + 1; i--) {
                slartg(f, g, &cosr, &sinr, &r);
                if (i < m - 1) {
                    E[i] = r;
                }
                f = cosr * D[i] + sinr * E[i - 1];
                E[i - 1] = cosr * E[i - 1] - sinr * D[i];
                g = sinr * D[i - 1];
                D[i - 1] = cosr * D[i - 1];
                slartg(f, g, &cosl, &sinl, &r);
                D[i] = r;
                f = cosl * E[i - 1] + sinl * D[i - 1];
                D[i - 1] = cosl * D[i - 1] - sinl * E[i - 1];
                if (i > ll + 1) {
                    g = sinl * E[i - 2];
                    E[i - 2] = cosl * E[i - 2];
                }
                work[i - ll - 1] = cosr;
                work[i - ll - 1 + nm1] = -sinr;
                work[i - ll - 1 + nm12] = cosl;
                work[i - ll - 1 + nm13] = -sinl;
            }
            E[ll] = f;

            /* Test convergence */
            if (fabsf(E[ll]) <= thresh) {
                E[ll] = ZERO;
            }

            /* Update singular vectors if desired */
            if (ncvt > 0) {
                slasr("L", "V", "B", m - ll, ncvt,
                      &work[nm12], &work[nm13], &VT[ll], ldvt);
            }
            if (nru > 0) {
                slasr("R", "V", "B", nru, m - ll, work,
                      &work[nm1], &U[ll * ldu], ldu);
            }
            if (ncc > 0) {
                slasr("L", "V", "B", m - ll, ncc, work,
                      &work[nm1], &C[ll], ldc);
            }
        }
    }

    /* QR iteration finished, go back and check convergence */
    goto L60;

    /* All singular values converged, so make them positive */
L160:
    for (i = 0; i < n; i++) {
        if (D[i] == ZERO) {
            /* Avoid -ZERO */
            D[i] = ZERO;
        }
        if (D[i] < ZERO) {
            D[i] = -D[i];

            /* Change sign of singular vectors, if desired */
            if (ncvt > 0) {
                cblas_sscal(ncvt, NEGONE, &VT[i], ldvt);
            }
        }
    }

    /* Sort the singular values into decreasing order (insertion sort on
     * singular values, but only one transposition per singular vector) */
    for (i = 0; i < n - 1; i++) {
        /* Scan for smallest D(I) */
        isub = 0;
        smin = D[0];
        for (j = 1; j < n - i; j++) {
            if (D[j] <= smin) {
                isub = j;
                smin = D[j];
            }
        }
        if (isub != n - 1 - i) {
            /* Swap singular values and vectors */
            D[isub] = D[n - 1 - i];
            D[n - 1 - i] = smin;
            if (ncvt > 0) {
                cblas_sswap(ncvt, &VT[isub], ldvt, &VT[n - 1 - i], ldvt);
            }
            if (nru > 0) {
                cblas_sswap(nru, &U[isub * ldu], 1, &U[(n - 1 - i) * ldu], 1);
            }
            if (ncc > 0) {
                cblas_sswap(ncc, &C[isub], ldc, &C[n - 1 - i], ldc);
            }
        }
    }
    return;

    /* Maximum number of iterations exceeded, failure to converge */
L200:
    *info = 0;
    for (i = 0; i < n - 1; i++) {
        if (E[i] != ZERO) {
            (*info)++;
        }
    }
    return;
}
