/**
 * @file dbdsdc.c
 * @brief DBDSDC computes the singular value decomposition (SVD) of a real
 *        N-by-N (upper or lower) bidiagonal matrix using divide and conquer.
 */

#include "semicolon_lapack_double.h"
#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;
static const f64 TWO = 2.0;

/* SMLSIZ: maximum size of subproblems at bottom of DC tree.
 * From ilaenv.f ISPEC=9, the default is 25. */
static const INT SMLSIZ = 25;

/**
 * DBDSDC computes the singular value decomposition (SVD) of a real
 * N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
 * using a divide and conquer method, where S is a diagonal matrix
 * with non-negative diagonal elements (the singular values of B), and
 * U and VT are orthogonal matrices of left and right singular vectors,
 * respectively. DBDSDC can be used to compute all singular values,
 * and optionally, singular vectors or singular vectors in compact form.
 *
 * The code currently calls DLASDQ if singular values only are desired.
 * However, it can be slightly modified to compute singular values
 * using the divide and conquer method.
 *
 * @param[in]     uplo    = 'U': B is upper bidiagonal.
 *                         = 'L': B is lower bidiagonal.
 * @param[in]     compq   = 'N': Compute singular values only.
 *                         = 'P': Compute singular values and vectors in compact form.
 *                         = 'I': Compute singular values and singular vectors.
 * @param[in]     n       The order of the matrix B. n >= 0.
 * @param[in,out] D       Array of dimension n. On entry, diagonal elements.
 *                        On exit, singular values of B.
 * @param[in,out] E       Array of dimension n-1. Off-diagonal elements.
 *                        On exit, E has been destroyed.
 * @param[out]    U       Array (ldu, n). If compq='I', left singular vectors.
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    VT      Array (ldvt, n). If compq='I', right singular vectors.
 * @param[in]     ldvt    Leading dimension of VT.
 * @param[out]    Q       Array. If compq='P', compact form of singular vectors.
 * @param[out]    IQ      Integer array. If compq='P', compact form data.
 * @param[out]    work    Array of dimension:
 *                        if compq='N': 4*n
 *                        if compq='P': 6*n
 *                        if compq='I': 3*n^2 + 4*n
 * @param[out]    IWORK   Integer array of dimension 8*n.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dbdsdc(const char* uplo, const char* compq, const INT n,
            f64* restrict D, f64* restrict E,
            f64* restrict U, const INT ldu,
            f64* restrict VT, const INT ldvt,
            f64* restrict Q, INT* restrict IQ,
            f64* restrict work, INT* restrict IWORK, INT* info)
{
    INT difl = 0, difr = 0, givcol = 0, givnum = 0, givptr, i, ic = 0, icompq, ierr;
    INT ii, is = 0, iu, iuplo, ivt, j, k, kk, mlvl, nm1, nsize, perm;
    INT poles = 0, qstart, smlszp, sqre, start, wstart, z = 0;
    f64 cs, eps, orgnrm, p, r, sn;

    /* Test the input parameters */
    *info = 0;

    iuplo = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        iuplo = 1;
    }
    if (uplo[0] == 'L' || uplo[0] == 'l') {
        iuplo = 2;
    }

    if (compq[0] == 'N' || compq[0] == 'n') {
        icompq = 0;
    } else if (compq[0] == 'P' || compq[0] == 'p') {
        icompq = 1;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        icompq = 2;
    } else {
        icompq = -1;
    }

    if (iuplo == 0) {
        *info = -1;
    } else if (icompq < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if ((ldu < 1) || (icompq == 2 && ldu < n)) {
        *info = -7;
    } else if ((ldvt < 1) || (icompq == 2 && ldvt < n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DBDSDC", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* smlsiz = 25 from ilaenv(9, 'DBDSDC', ...) */
    if (n == 1) {
        if (icompq == 1) {
            Q[0] = (D[0] >= ZERO) ? ONE : -ONE;
            Q[SMLSIZ * n] = ONE;
        } else if (icompq == 2) {
            U[0] = (D[0] >= ZERO) ? ONE : -ONE;
            VT[0] = ONE;
        }
        D[0] = fabs(D[0]);
        return;
    }
    nm1 = n - 1;

    /* If matrix lower bidiagonal, rotate to be upper bidiagonal
     * by applying Givens rotations on the left */
    wstart = 0;   /* 0-based index into work */
    qstart = 2;   /* qstart-1 = offset in Q array for starting matrices (0-based: 2) */

    if (icompq == 1) {
        cblas_dcopy(n, D, 1, Q, 1);
        cblas_dcopy(n - 1, E, 1, &Q[n], 1);
    }

    if (iuplo == 2) {
        qstart = 4;
        if (icompq == 2) {
            wstart = 2 * n - 2;  /* 0-based */
        }
        for (i = 0; i < n - 1; i++) {
            dlartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            if (icompq == 1) {
                Q[i + 2 * n] = cs;
                Q[i + 3 * n] = sn;
            } else if (icompq == 2) {
                work[i] = cs;
                work[nm1 + i] = -sn;
            }
        }
    }

    /* If ICOMPQ = 0, use DLASDQ to compute the singular values */
    if (icompq == 0) {
        /* Ignore WSTART, instead using WORK(1), since the two vectors
         * for CS and -SN above are added only if ICOMPQ == 2,
         * and adding them exceeds documented WORK size of 4*n. */
        dlasdq("U", 0, n, 0, 0, 0, D, E, NULL, 1, NULL, 1, NULL, 1, work, info);
        goto L40;
    }

    /* If N is smaller than the minimum divide size SMLSIZ, then solve
     * the problem with another solver. */
    if (n <= SMLSIZ) {
        if (icompq == 2) {
            dlaset("A", n, n, ZERO, ONE, U, ldu);
            dlaset("A", n, n, ZERO, ONE, VT, ldvt);
            dlasdq("U", 0, n, n, n, 0, D, E, VT, ldvt, U, ldu, NULL, 1,
                   &work[wstart], info);
        } else if (icompq == 1) {
            iu = 0;           /* 0-based */
            ivt = iu + n;
            /* Q(IU + (QSTART-1)*N) -> Q[iu + (qstart-1)*n] but Fortran is 1-based
             * In Fortran: Q(IU + (QSTART-1)*N) with IU=1, QSTART=3 -> Q(1 + 2*N)
             * In C (0-based): iu=0, qstart=2, so Q[0 + 2*n] = Q[2*n]
             */
            dlaset("A", n, n, ZERO, ONE, &Q[iu + qstart * n], n);
            dlaset("A", n, n, ZERO, ONE, &Q[ivt + qstart * n], n);
            dlasdq("U", 0, n, n, n, 0, D, E,
                   &Q[ivt + qstart * n], n,
                   &Q[iu + qstart * n], n,
                   NULL, 1,
                   &work[wstart], info);
        }
        goto L40;
    }

    if (icompq == 2) {
        dlaset("A", n, n, ZERO, ONE, U, ldu);
        dlaset("A", n, n, ZERO, ONE, VT, ldvt);
    }

    /* Scale */
    orgnrm = dlanst("M", n, D, E);
    if (orgnrm == ZERO) {
        return;
    }
    dlascl("G", 0, 0, orgnrm, ONE, n, 1, D, n, &ierr);
    dlascl("G", 0, 0, orgnrm, ONE, nm1, 1, E, nm1, &ierr);

    eps = 0.9 * dlamch("Epsilon");

    mlvl = (INT)(log((f64)n / (f64)(SMLSIZ + 1)) / log(TWO)) + 1;
    smlszp = SMLSIZ + 1;

    if (icompq == 1) {
        /* Workspace indices for compact form storage (0-based into Q) */
        iu = 0;
        ivt = iu + SMLSIZ;
        difl = ivt + smlszp;
        difr = difl + mlvl;
        z = difr + mlvl * 2;
        ic = z + mlvl;
        is = ic + 1;
        poles = is + 1;
        givnum = poles + 2 * mlvl;

        /* Integer workspace indices (0-based into IQ) */
        k = 0;
        givptr = 1;
        perm = 2;
        givcol = perm + mlvl;
    }

    for (i = 0; i < n; i++) {
        if (fabs(D[i]) < eps) {
            D[i] = (D[i] >= ZERO) ? eps : -eps;
        }
    }

    start = 0;  /* 0-based */
    sqre = 0;

    for (i = 0; i < nm1; i++) {
        if (fabs(E[i]) < eps || i == nm1 - 1) {
            /* Subproblem found. First determine its size and then
             * apply divide and conquer on it. */
            if (i < nm1 - 1) {
                /* A subproblem with E[i] small for i < nm1-1 */
                nsize = i - start + 1;
            } else if (fabs(E[i]) >= eps) {
                /* A subproblem with E[nm1-1] not too small but i = nm1-1 */
                nsize = n - start;
            } else {
                /* A subproblem with E[nm1-1] small. This implies an
                 * 1-by-1 subproblem at D[n-1]. Solve this 1-by-1 problem first. */
                nsize = i - start + 1;
                if (icompq == 2) {
                    U[(n - 1) + (n - 1) * ldu] = (D[n - 1] >= ZERO) ? ONE : -ONE;
                    VT[(n - 1) + (n - 1) * ldvt] = ONE;
                } else if (icompq == 1) {
                    /* Q(N + (QSTART-1)*N) in Fortran -> Q[(n-1) + qstart*n] in C */
                    Q[(n - 1) + qstart * n] = (D[n - 1] >= ZERO) ? ONE : -ONE;
                    Q[(n - 1) + (SMLSIZ + qstart) * n] = ONE;
                }
                D[n - 1] = fabs(D[n - 1]);
            }

            if (icompq == 2) {
                dlasd0(nsize, sqre, &D[start], &E[start],
                       &U[start + start * ldu], ldu,
                       &VT[start + start * ldvt], ldvt,
                       SMLSIZ, IWORK, &work[wstart], info);
            } else {
                /* Call dlasda for compact form. */
                dlasda(icompq, SMLSIZ, nsize, sqre, &D[start], &E[start],
                       &Q[start + (iu + qstart) * n], n,
                       &Q[start + (ivt + qstart) * n],
                       &IQ[start + k * n],
                       &Q[start + (difl + qstart) * n],
                       &Q[start + (difr + qstart) * n],
                       &Q[start + (z + qstart) * n],
                       &Q[start + (poles + qstart) * n],
                       &IQ[start + givptr * n],
                       &IQ[start + givcol * n], n,
                       &IQ[start + perm * n],
                       &Q[start + (givnum + qstart) * n],
                       &Q[start + (ic + qstart) * n],
                       &Q[start + (is + qstart) * n],
                       &work[wstart], IWORK, info);
            }
            if (*info != 0) {
                return;
            }
            start = i + 1;
        }
    }

    /* Unscale */
    dlascl("G", 0, 0, ONE, orgnrm, n, 1, D, n, &ierr);

L40:
    /* Use Selection Sort to minimize swaps of singular vectors */
    for (ii = 1; ii < n; ii++) {
        i = ii - 1;
        kk = i;
        p = D[i];
        for (j = ii; j < n; j++) {
            if (D[j] > p) {
                kk = j;
                p = D[j];
            }
        }
        if (kk != i) {
            D[kk] = D[i];
            D[i] = p;
            if (icompq == 1) {
                IQ[i] = kk;
            } else if (icompq == 2) {
                cblas_dswap(n, &U[i * ldu], 1, &U[kk * ldu], 1);
                cblas_dswap(n, &VT[i], ldvt, &VT[kk], ldvt);
            }
        } else if (icompq == 1) {
            IQ[i] = i;
        }
    }

    /* If ICOMPQ = 1, use IQ(N-1) as the indicator for UPLO */
    if (icompq == 1) {
        if (iuplo == 1) {
            IQ[n - 1] = 1;
        } else {
            IQ[n - 1] = 0;
        }
    }

    /* If B is lower bidiagonal, update U by those Givens rotations
     * which rotated B to be upper bidiagonal */
    if (iuplo == 2 && icompq == 2) {
        dlasr("L", "V", "B", n, n, work, &work[n - 1], U, ldu);
    }
}
