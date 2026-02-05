/**
 * @file dbdsvdx.c
 * @brief DBDSVDX computes the singular value decomposition (SVD) of a real
 *        N-by-N bidiagonal matrix using eigenvalues of an associated tridiagonal.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

static const double ZERO = 0.0;
static const double ONE = 1.0;
static const double TEN = 10.0;
static const double HNDRD = 100.0;
static const double MEIGTH = -0.125;
static const double FUDGE = 2.0;

/**
 * DBDSVDX computes the singular value decomposition (SVD) of a real
 * N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
 * where S is a diagonal matrix with non-negative diagonal elements
 * (the singular values of B), and U and VT are orthogonal matrices
 * of left and right singular vectors, respectively.
 *
 * Given an upper bidiagonal B with diagonal D = [ d_1 d_2 ... d_N ]
 * and superdiagonal E = [ e_1 e_2 ... e_N-1 ], DBDSVDX computes the
 * singular value decomposition of B through the eigenvalues and
 * eigenvectors of the N*2-by-N*2 tridiagonal matrix
 *
 *        |  0  d_1                |
 *        | d_1  0  e_1            |
 *  TGK = |     e_1  0  d_2        |
 *        |         d_2  .   .     |
 *        |              .   .   . |
 *
 * @param[in]     uplo    = 'U': B is upper bidiagonal.
 *                         = 'L': B is lower bidiagonal.
 * @param[in]     jobz    = 'N': Compute singular values only.
 *                         = 'V': Compute singular values and singular vectors.
 * @param[in]     range   = 'A': all singular values will be found.
 *                         = 'V': all singular values in [VL,VU) will be found.
 *                         = 'I': the IL-th through IU-th singular values.
 * @param[in]     n       The order of the bidiagonal matrix. n >= 0.
 * @param[in]     D       Array of dimension n. Diagonal elements.
 * @param[in]     E       Array of dimension max(1,n-1). Superdiagonal elements.
 * @param[in]     vl      If range='V', lower bound of interval for singular values.
 * @param[in]     vu      If range='V', upper bound of interval. vu > vl.
 * @param[in]     il      If range='I', index of smallest singular value (1-based).
 * @param[in]     iu      If range='I', index of largest singular value (1-based).
 * @param[out]    ns      Number of singular values found. 0 <= ns <= n.
 * @param[out]    S       Array (n). First ns elements contain selected singular values.
 * @param[out]    Z       Array (ldz, K). If jobz='V', singular vectors stored as
 *                        Z = [U; V] with U in rows 1..n and V in rows n+1..2*n.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, ldz >= 2*n if jobz='V'.
 * @param[out]    work    Array of dimension 14*n.
 * @param[out]    iwork   Integer array of dimension 12*n.
 * @param[out]    info    = 0: success.
 *                        < 0: illegal argument.
 *                        > 0: i eigenvectors failed to converge in DSTEVX.
 */
void dbdsvdx(const char* uplo, const char* jobz, const char* range, const int n,
             double* const restrict D, double* const restrict E,
             const double vl, const double vu, const int il, const int iu,
             int* ns, double* const restrict S, double* const restrict Z,
             const int ldz, double* const restrict work, int* const restrict iwork,
             int* info)
{
    int allsv, indsv, lower, split, sveq0, valsv, wantz;
    int i, icolz, idbeg, idend, idptr, idtgk, ieptr, ietgk;
    int iifail, iiwork, iltgk, irowu, irowv, irowz, isbeg;
    int isplt, itemp, iutgk, j, k, ns_local, nsl, ntgk;
    int nru, nrv;
    double abstol, emin, eps, mu, nrmu, nrmv, ortol, smax, smin;
    double sqrt2, thresh, tol, ulp, vltgk, vutgk, zjtji;
    char rngvx = 'I';

    /* Test the input parameters */
    *info = 0;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    allsv = (range[0] == 'A' || range[0] == 'a');
    valsv = (range[0] == 'V' || range[0] == 'v');
    indsv = (range[0] == 'I' || range[0] == 'i');

    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !lower) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!allsv && !valsv && !indsv) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (n > 0) {
        if (valsv) {
            if (vl < ZERO) {
                *info = -7;
            } else if (vu <= vl) {
                *info = -8;
            }
        } else if (indsv) {
            if (il < 1 || il > (n > 1 ? n : 1)) {
                *info = -9;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n * 2)) {
            *info = -14;
        }
    }

    if (*info != 0) {
        xerbla("DBDSVDX", -(*info));
        return;
    }

    /* Quick return if possible (N <= 1) */
    *ns = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (allsv || indsv) {
            *ns = 1;
            S[0] = fabs(D[0]);
        } else {
            if (vl < fabs(D[0]) && vu >= fabs(D[0])) {
                *ns = 1;
                S[0] = fabs(D[0]);
            }
        }
        if (wantz) {
            Z[0] = (D[0] >= ZERO) ? ONE : -ONE;
            Z[1] = ONE;
        }
        return;
    }

    abstol = 2.0 * dlamch("S");  /* 2 * Safe minimum */
    ulp = dlamch("P");           /* Precision */
    eps = dlamch("E");           /* Epsilon */
    sqrt2 = sqrt(2.0);
    ortol = sqrt(ulp);

    /* Criterion for splitting from DBDSQR */
    tol = fmax(TEN, fmin(HNDRD, pow(eps, MEIGTH))) * eps;

    /* Compute approximate maximum, minimum singular values */
    i = cblas_idamax(n, D, 1);
    smax = fabs(D[i]);
    if (n > 1) {
        i = cblas_idamax(n - 1, E, 1);
        smax = fmax(smax, fabs(E[i]));
    }

    /* Compute threshold for neglecting D's and E's */
    smin = fabs(D[0]);
    if (smin != ZERO) {
        mu = smin;
        for (i = 1; i < n; i++) {
            mu = fabs(D[i]) * (mu / (mu + fabs(E[i - 1])));
            smin = fmin(smin, mu);
            if (smin == ZERO) break;
        }
    }
    smin = smin / sqrt((double)n);
    thresh = tol * smin;

    /* Check for zeros in D and E (splits), i.e., submatrices */
    for (i = 0; i < n - 1; i++) {
        if (fabs(D[i]) <= thresh) D[i] = ZERO;
        if (fabs(E[i]) <= thresh) E[i] = ZERO;
    }
    if (fabs(D[n - 1]) <= thresh) D[n - 1] = ZERO;

    /* Pointers for arrays used by DSTEVX (0-based) */
    idtgk = 0;
    ietgk = idtgk + 2 * n;
    itemp = ietgk + 2 * n;
    iifail = 0;
    iiwork = iifail + 2 * n;

    /* Set RNGVX, which corresponds to RANGE for DSTEVX in TGK mode */
    iltgk = 0;
    iutgk = 0;
    vltgk = ZERO;
    vutgk = ZERO;

    if (allsv) {
        /* All singular values. Use RNGVX='I' with indices set later. */
        rngvx = 'I';
        if (wantz) {
            dlaset("F", 2 * n, n + 1, ZERO, ZERO, Z, ldz);
        }
    } else if (valsv) {
        /* Find singular values in [VL, VU). Swap and negate for negative eigenvalues. */
        rngvx = 'V';
        vltgk = -vu;
        vutgk = -vl;
        /* Zero out WORK arrays for TGK diagonal */
        for (i = 0; i < 2 * n; i++) {
            work[idtgk + i] = ZERO;
        }
        /* Copy D to odd positions (1, 3, 5, ...) of IETGK */
        cblas_dcopy(n, D, 1, &work[ietgk], 2);
        /* Copy E to even positions (2, 4, 6, ...) of IETGK */
        if (n > 1) {
            cblas_dcopy(n - 1, E, 1, &work[ietgk + 1], 2);
        }
        dstevx("N", "V", 2 * n, &work[idtgk], &work[ietgk], vltgk, vutgk,
               iltgk, iltgk, abstol, &ns_local, S, Z, ldz, &work[itemp],
               &iwork[iiwork], &iwork[iifail], info);
        if (ns_local == 0) {
            *ns = 0;
            return;
        } else {
            if (wantz) {
                dlaset("F", 2 * n, ns_local, ZERO, ZERO, Z, ldz);
            }
        }
    } else if (indsv) {
        /* Find IL-th through IU-th singular values.
         * Map indices to values by finding boundary eigenvalues. */
        iltgk = il;  /* 1-based in Fortran, but dstevx expects 1-based too */
        iutgk = iu;
        rngvx = 'V';

        /* Zero out WORK arrays for TGK diagonal */
        for (i = 0; i < 2 * n; i++) {
            work[idtgk + i] = ZERO;
        }
        /* Copy D to odd positions, E to even positions */
        cblas_dcopy(n, D, 1, &work[ietgk], 2);
        if (n > 1) {
            cblas_dcopy(n - 1, E, 1, &work[ietgk + 1], 2);
        }

        /* Find the IL-th eigenvalue (for lower bound) */
        dstevx("N", "I", 2 * n, &work[idtgk], &work[ietgk], vltgk, vltgk,
               iltgk, iltgk, abstol, &ns_local, S, Z, ldz, &work[itemp],
               &iwork[iiwork], &iwork[iifail], info);
        vltgk = S[0] - FUDGE * smax * ulp * n;

        /* Reset work arrays */
        for (i = 0; i < 2 * n; i++) {
            work[idtgk + i] = ZERO;
        }
        cblas_dcopy(n, D, 1, &work[ietgk], 2);
        if (n > 1) {
            cblas_dcopy(n - 1, E, 1, &work[ietgk + 1], 2);
        }

        /* Find the IU-th eigenvalue (for upper bound) */
        dstevx("N", "I", 2 * n, &work[idtgk], &work[ietgk], vutgk, vutgk,
               iutgk, iutgk, abstol, &ns_local, S, Z, ldz, &work[itemp],
               &iwork[iiwork], &iwork[iifail], info);
        vutgk = S[0] + FUDGE * smax * ulp * n;
        vutgk = fmin(vutgk, ZERO);

        /* If VLTGK == VUTGK, DSTEVX returns error, so adjust slightly */
        if (vltgk == vutgk) {
            vltgk = vltgk - tol;
        }

        if (wantz) {
            dlaset("F", 2 * n, iu - il + 1, ZERO, ZERO, Z, ldz);
        }
    }

    /* Initialize variables and pointers for S, Z, and WORK */
    ns_local = 0;
    nru = 0;
    nrv = 0;
    idbeg = 0;  /* 0-based start into D */
    isbeg = 0;  /* 0-based start into S */
    irowz = 0;  /* 0-based row offset in Z */
    icolz = 0;  /* 0-based column offset in Z */
    irowu = 1;  /* rows for U start at odd indices in Z (0-based: row 1, 3, 5...) */
    irowv = 0;  /* rows for V start at even indices in Z (0-based: row 0, 2, 4...) */
    split = 0;
    sveq0 = 0;


    /* Initialize S to zero */
    for (i = 0; i < n; i++) {
        S[i] = ZERO;
    }

    /* Form the tridiagonal TGK matrix */
    work[ietgk + 2 * n - 1] = ZERO;
    for (i = 0; i < 2 * n; i++) {
        work[idtgk + i] = ZERO;
    }
    /* D goes to odd positions, E to even */
    cblas_dcopy(n, D, 1, &work[ietgk], 2);
    if (n > 1) {
        cblas_dcopy(n - 1, E, 1, &work[ietgk + 1], 2);
    }

    /* Check for splits in two levels: outer level in E, inner level in D */
    for (ieptr = 1; ieptr < 2 * n; ieptr += 2) {
        if (work[ietgk + ieptr] == ZERO) {
            /* Split in E (this piece of B is square) or bottom of matrix */
            isplt = idbeg;
            idend = ieptr - 1;  /* Last index of D in this subproblem */

            for (idptr = idbeg; idptr <= idend; idptr += 2) {
                if (work[ietgk + idptr] == ZERO) {
                    /* Split in D (rectangular submatrix) */
                    if (idptr == idbeg) {
                        /* D=0 at the top */
                        sveq0 = 1;
                        if (idbeg == idend) {
                            nru = 1;
                            nrv = 1;
                        }
                    } else if (idptr == idend) {
                        /* D=0 at the bottom */
                        sveq0 = 1;
                        nru = (idend - isplt) / 2 + 1;
                        nrv = nru;
                        if (isplt != idbeg) {
                            nru = nru + 1;
                        }
                    } else {
                        if (isplt == idbeg) {
                            /* Split: top rectangular submatrix */
                            nru = (idptr - idbeg) / 2;
                            nrv = nru + 1;
                        } else {
                            /* Split: middle square submatrix */
                            nru = (idptr - isplt) / 2 + 1;
                            nrv = nru;
                        }
                    }
                } else if (idptr == idend) {
                    /* Last entry of D in the active submatrix */
                    if (isplt == idbeg) {
                        /* No split (trivial case) */
                        nru = (idend - idbeg) / 2 + 1;
                        nrv = nru;
                    } else {
                        /* Split: bottom rectangular submatrix */
                        nrv = (idend - isplt) / 2 + 1;
                        nru = nrv + 1;
                    }
                }

                ntgk = nru + nrv;


                if (ntgk > 0) {
                    /* Compute eigenvalues/vectors of the active submatrix */
                    iltgk = 1;
                    iutgk = ntgk / 2;
                    if (allsv || vutgk == ZERO) {
                        if (sveq0 || smin < eps || (ntgk % 2) > 0) {
                            /* Special case: eigenvalue equal to zero or very small */
                            iutgk = iutgk + 1;
                        }
                    }


                    char rngvx_str[2];
                    rngvx_str[0] = rngvx;
                    rngvx_str[1] = '\0';

                    dstevx(wantz ? "V" : "N", rngvx_str, ntgk,
                           &work[idtgk + isplt], &work[ietgk + isplt],
                           vltgk, vutgk, iltgk, iutgk, abstol, &nsl,
                           &S[isbeg], &Z[irowz + icolz * ldz], ldz,
                           &work[itemp], &iwork[iiwork], &iwork[iifail], info);


                    if (*info != 0) {
                        return;
                    }

                    /* Find absolute value of maximum eigenvalue.
                     * Note: Fortran uses EMIN = ABS(MAXVAL(S(...))), which finds
                     * the algebraically largest value, then takes its absolute value.
                     * This is NOT the same as max(abs(values))!
                     * For eigenvalues [-1.65e-138, -1.93e-308, 0], MAXVAL = 0,
                     * so EMIN = ABS(0) = 0, triggering eigenvector concatenation. */
                    {
                        double maxval = S[isbeg];
                        for (i = 1; i < nsl; i++) {
                            if (S[isbeg + i] > maxval) {
                                maxval = S[isbeg + i];
                            }
                        }
                        emin = fabs(maxval);
                    }

                    if (nsl > 0 && wantz) {
                        /* Normalize u and v, changing sign of v */
                        if (nsl > 1 && vutgk == ZERO && (ntgk % 2) == 0 &&
                            emin == ZERO && !split) {
                            /* D=0 at top or bottom: concatenate eigenvectors
                             * for the two smallest eigenvalues */
                            for (i = 0; i < ntgk; i++) {
                                Z[irowz + i + (icolz + nsl - 2) * ldz] +=
                                    Z[irowz + i + (icolz + nsl - 1) * ldz];
                                Z[irowz + i + (icolz + nsl - 1) * ldz] = ZERO;
                            }
                        }

                        /* Normalize U vectors (rows irowu, irowu+2, ...) */
                        for (i = 0; i < nsl && i < nru; i++) {
                            nrmu = cblas_dnrm2(nru, &Z[irowu + (icolz + i) * ldz], 2);
                            if (nrmu == ZERO) {
                                *info = 2 * n + 1;
                                return;
                            }
                            cblas_dscal(nru, ONE / nrmu, &Z[irowu + (icolz + i) * ldz], 2);
                            if (nrmu != ONE && fabs(nrmu - ortol) * sqrt2 > ONE) {
                                /* Reorthogonalize */
                                for (j = 0; j < i; j++) {
                                    zjtji = -cblas_ddot(nru, &Z[irowu + (icolz + j) * ldz], 2,
                                                        &Z[irowu + (icolz + i) * ldz], 2);
                                    cblas_daxpy(nru, zjtji, &Z[irowu + (icolz + j) * ldz], 2,
                                                &Z[irowu + (icolz + i) * ldz], 2);
                                }
                                nrmu = cblas_dnrm2(nru, &Z[irowu + (icolz + i) * ldz], 2);
                                cblas_dscal(nru, ONE / nrmu, &Z[irowu + (icolz + i) * ldz], 2);
                            }
                        }

                        /* Normalize V vectors (rows irowv, irowv+2, ...) with sign change */
                        for (i = 0; i < nsl && i < nrv; i++) {
                            nrmv = cblas_dnrm2(nrv, &Z[irowv + (icolz + i) * ldz], 2);
                            if (nrmv == ZERO) {
                                *info = 2 * n + 1;
                                return;
                            }
                            cblas_dscal(nrv, -ONE / nrmv, &Z[irowv + (icolz + i) * ldz], 2);
                            if (nrmv != ONE && fabs(nrmv - ortol) * sqrt2 > ONE) {
                                /* Reorthogonalize */
                                for (j = 0; j < i; j++) {
                                    zjtji = -cblas_ddot(nrv, &Z[irowv + (icolz + j) * ldz], 2,
                                                        &Z[irowv + (icolz + i) * ldz], 2);
                                    cblas_daxpy(nru, zjtji, &Z[irowv + (icolz + j) * ldz], 2,
                                                &Z[irowv + (icolz + i) * ldz], 2);
                                }
                                nrmv = cblas_dnrm2(nrv, &Z[irowv + (icolz + i) * ldz], 2);
                                cblas_dscal(nrv, ONE / nrmv, &Z[irowv + (icolz + i) * ldz], 2);
                            }
                        }

                        if (vutgk == ZERO && idptr < idend && (ntgk % 2) > 0) {
                            /* D=0 in the middle: save eigenvector for later */
                            split = 1;
                            for (i = 0; i < ntgk; i++) {
                                Z[irowz + i + n * ldz] =
                                    Z[irowz + i + (ns_local + nsl - 1) * ldz];
                                Z[irowz + i + (ns_local + nsl - 1) * ldz] = ZERO;
                            }
                        }
                    } /* WANTZ */

                    nsl = nsl < nru ? nsl : nru;
                    sveq0 = 0;

                    /* Absolute values of eigenvalues of TGK */
                    for (i = 0; i < nsl; i++) {
                        S[isbeg + i] = fabs(S[isbeg + i]);
                    }

                    /* Update pointers */
                    isbeg = isbeg + nsl;
                    irowz = irowz + ntgk;
                    icolz = icolz + nsl;
                    /* Fortran: IROWU = IROWZ, IROWV = IROWZ + 1
                     * Converting 1-based to 0-based: irowu = irowz (new), irowv = irowz + 1 (new)
                     * But we store in 0-based, so irowu = irowz, irowv = irowz + 1 */
                    irowu = irowz;
                    irowv = irowz + 1;
                    isplt = idptr + 2;
                    ns_local = ns_local + nsl;
                    nru = 0;
                    nrv = 0;
                } /* NTGK > 0 */

                if (irowz < 2 * n && wantz) {
                    for (i = 0; i < irowz; i++) {
                        Z[i + icolz * ldz] = ZERO;
                    }
                }
            } /* IDPTR loop */

            if (split && wantz) {
                /* Bring back eigenvector corresponding to eigenvalue zero */
                for (i = idbeg; i <= idend - ntgk + 1; i++) {
                    Z[i + (isbeg - 1) * ldz] += Z[i + n * ldz];
                    Z[i + n * ldz] = ZERO;
                }
            }
            irowv = irowv - 1;
            irowu = irowu + 1;
            idbeg = ieptr + 1;
            sveq0 = 0;
            split = 0;
        } /* Check for split in E */
    } /* IEPTR loop */

    *ns = ns_local;

    /* Sort singular values into decreasing order (insertion sort) */
    for (i = 0; i < ns_local - 1; i++) {
        k = 0;
        smin = S[0];
        for (j = 1; j < ns_local - i; j++) {
            if (S[j] <= smin) {
                k = j;
                smin = S[j];
            }
        }
        if (k != ns_local - 1 - i) {
            S[k] = S[ns_local - 1 - i];
            S[ns_local - 1 - i] = smin;
            if (wantz) {
                cblas_dswap(2 * n, &Z[k * ldz], 1, &Z[(ns_local - 1 - i) * ldz], 1);
            }
        }
    }

    /* If RANGE='I', check for singular values/vectors to be discarded */
    if (indsv) {
        k = iu - il + 1;
        if (k < ns_local) {
            for (i = k; i < ns_local; i++) {
                S[i] = ZERO;
            }
            if (wantz) {
                for (i = k; i < ns_local; i++) {
                    for (j = 0; j < 2 * n; j++) {
                        Z[j + i * ldz] = ZERO;
                    }
                }
            }
            *ns = k;
        }
    }

    /* Reorder Z: U = Z(1:N,:), V = Z(N+1:2*N,:).
     * If B is lower bidiagonal, swap U and V. */
    if (wantz) {
        for (i = 0; i < *ns; i++) {
            /* Copy column to work, then redistribute */
            cblas_dcopy(2 * n, &Z[i * ldz], 1, work, 1);
            if (lower) {
                /* V from even rows, U from odd rows */
                for (j = 0; j < n; j++) {
                    Z[n + j + i * ldz] = work[2 * j + 1];  /* V */
                    Z[j + i * ldz] = work[2 * j];          /* U */
                }
            } else {
                /* U from odd rows, V from even rows */
                for (j = 0; j < n; j++) {
                    Z[j + i * ldz] = work[2 * j + 1];      /* U */
                    Z[n + j + i * ldz] = work[2 * j];      /* V */
                }
            }
        }
    }
}
