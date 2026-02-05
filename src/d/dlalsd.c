/**
 * @file dlalsd.c
 * @brief DLALSD uses the singular value decomposition of A to solve
 *        the least squares problem.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

static const double ONE = 1.0;
static const double ZERO = 0.0;
static const double TWO = 2.0;

/**
 * DLALSD uses the singular value decomposition of A to solve the least
 * squares problem of finding X to minimize the Euclidean norm of each
 * column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
 * are N-by-NRHS. The solution X overwrites B.
 *
 * The singular values of A smaller than RCOND times the largest
 * singular value are treated as zero in solving the least squares
 * problem; in this case a minimum norm solution is returned.
 * The actual singular values are returned in D in ascending order.
 *
 * @param[in]     uplo     = 'U': D and E define an upper bidiagonal matrix.
 *                          = 'L': D and E define a lower bidiagonal matrix.
 * @param[in]     smlsiz   The maximum size of subproblems at the bottom of the tree.
 * @param[in]     n        The dimension of the bidiagonal matrix. n >= 0.
 * @param[in]     nrhs     The number of columns of B. nrhs >= 1.
 * @param[in,out] D        Array of dimension n. On entry, the main diagonal.
 *                          On exit, the singular values in ascending order.
 * @param[in,out] E        Array of dimension n-1. On entry, the super-diagonal.
 *                          On exit, E has been destroyed.
 * @param[in,out] B        Array of dimension (ldb, nrhs).
 *                          On input, the right hand sides.
 *                          On output, the solution X.
 * @param[in]     ldb      The leading dimension of B. ldb >= max(1, n).
 * @param[in]     rcond    The singular values less than or equal to rcond*max(S)
 *                          are treated as zero. If rcond < 0, machine precision is used.
 * @param[out]    rank     The number of singular values > rcond*max(S).
 * @param[out]    work     Array of dimension at least
 *                          (9*n + 2*n*smlsiz + 8*n*nlvl + n*nrhs + (smlsiz+1)^2).
 * @param[out]    iwork    Integer array of dimension at least (3*n*nlvl + 11*n).
 * @param[out]    info     = 0: successful exit.
 *                          < 0: if info = -i, the i-th argument had illegal value.
 *                          > 0: The algorithm failed to compute a singular value.
 */
void dlalsd(const char* uplo, const int smlsiz, const int n, const int nrhs,
            double* const restrict D, double* const restrict E,
            double* const restrict B, const int ldb, const double rcond,
            int* rank, double* const restrict work, int* const restrict iwork,
            int* info)
{
    int bx, bxst, c_idx, difl_idx, difr_idx, givcol, givnum;
    int givptr, i, icmpq1, icmpq2, iwk, j, k_idx, nlvl;
    int nm1, nsize, nsub, nwork, perm, poles, s_idx, sizei;
    int smlszp, sqre, st, st1, u_idx, vt_idx, z_idx;
    double cs, eps, orgnrm, r, rcnd, sn, tol;

    *info = 0;

    if (n < 0) {
        *info = -3;
    } else if (nrhs < 1) {
        *info = -4;
    } else if (ldb < 1 || ldb < n) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("DLALSD", -(*info));
        return;
    }

    eps = dlamch("Epsilon");

    /* Set up the tolerance. */
    if (rcond <= ZERO || rcond >= ONE) {
        rcnd = eps;
    } else {
        rcnd = rcond;
    }

    *rank = 0;

    /* Quick return if possible. */
    if (n == 0) {
        return;
    } else if (n == 1) {
        if (D[0] == ZERO) {
            dlaset("A", 1, nrhs, ZERO, ZERO, B, ldb);
        } else {
            *rank = 1;
            dlascl("G", 0, 0, D[0], ONE, 1, nrhs, B, ldb, info);
            D[0] = fabs(D[0]);
        }
        return;
    }

    /* Rotate the matrix if it is lower bidiagonal. */
    if (uplo[0] == 'L' || uplo[0] == 'l') {
        /* DO 10 I = 1, N - 1 (Fortran) -> i = 0..n-2 (C) */
        for (i = 0; i < n - 1; i++) {
            dlartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            if (nrhs == 1) {
                cblas_drot(1, &B[i], 1, &B[i + 1], 1, cs, sn);
            } else {
                work[i * 2] = cs;
                work[i * 2 + 1] = sn;
            }
        }
        if (nrhs > 1) {
            /* DO 30 I = 1, NRHS (Fortran) -> i = 0..nrhs-1 (C) */
            for (i = 0; i < nrhs; i++) {
                /* DO 20 J = 1, N - 1 (Fortran) -> j = 0..n-2 (C) */
                for (j = 0; j < n - 1; j++) {
                    cs = work[j * 2];
                    sn = work[j * 2 + 1];
                    cblas_drot(1, &B[j + i * ldb], 1, &B[j + 1 + i * ldb], 1, cs, sn);
                }
            }
        }
    }

    /* Scale. */
    nm1 = n - 1;
    orgnrm = dlanst("M", n, D, E);
    if (orgnrm == ZERO) {
        dlaset("A", n, nrhs, ZERO, ZERO, B, ldb);
        return;
    }

    dlascl("G", 0, 0, orgnrm, ONE, n, 1, D, n, info);
    dlascl("G", 0, 0, orgnrm, ONE, nm1, 1, E, nm1, info);

    /* If N is smaller than the minimum divide size SMLSIZ, then solve
     * the problem with another solver.
     */
    if (n <= smlsiz) {
        nwork = 1 + n * n;
        dlaset("A", n, n, ZERO, ONE, work, n);
        /* NRU=0: U not referenced */
        dlasdq("U", 0, n, n, 0, nrhs, D, E, work, n, NULL, n, B, ldb, &work[nwork], info);
        if (*info != 0) {
            return;
        }
        tol = rcnd * fabs(D[cblas_idamax(n, D, 1)]);
        /* DO 40 I = 1, N (Fortran) -> i = 0..n-1 (C) */
        for (i = 0; i < n; i++) {
            if (D[i] <= tol) {
                dlaset("A", 1, nrhs, ZERO, ZERO, &B[i], ldb);
            } else {
                dlascl("G", 0, 0, D[i], ONE, 1, nrhs, &B[i], ldb, info);
                *rank = *rank + 1;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, n, ONE, work, n, B, ldb, ZERO, &work[nwork], n);
        dlacpy("A", n, nrhs, &work[nwork], n, B, ldb);

        /* Unscale. */
        dlascl("G", 0, 0, ONE, orgnrm, n, 1, D, n, info);
        dlasrt("D", n, D, info);
        dlascl("G", 0, 0, orgnrm, ONE, n, nrhs, B, ldb, info);

        return;
    }

    /* Book-keeping and setting up some constants. */
    nlvl = (int)(log((double)n / (double)(smlsiz + 1)) / log(TWO)) + 1;

    smlszp = smlsiz + 1;

    u_idx = 0;
    vt_idx = smlsiz * n;
    difl_idx = vt_idx + smlszp * n;
    difr_idx = difl_idx + nlvl * n;
    z_idx = difr_idx + nlvl * n * 2;
    c_idx = z_idx + nlvl * n;
    s_idx = c_idx + n;
    poles = s_idx + n;
    givnum = poles + 2 * nlvl * n;
    bx = givnum + 2 * nlvl * n;
    nwork = bx + n * nrhs;

    sizei = n;  /* Fortran: SIZEI = 1 + N, but our indices are shifted by 1 for 0-based */
    k_idx = sizei + n;
    givptr = k_idx + n;
    perm = givptr + n;
    givcol = perm + nlvl * n;
    iwk = givcol + nlvl * n * 2;

    st = 0;
    sqre = 0;
    icmpq1 = 1;
    icmpq2 = 0;
    nsub = 0;

    /* DO 50 I = 1, N (Fortran) -> i = 0..n-1 (C) */
    for (i = 0; i < n; i++) {
        if (fabs(D[i]) < eps) {
            D[i] = copysign(eps, D[i]);
        }
    }

    /* DO 60 I = 1, NM1 (Fortran) -> i = 0..nm1-1 (C) */
    for (i = 0; i < nm1; i++) {
        if (fabs(E[i]) < eps || i == nm1 - 1) {
            nsub = nsub + 1;
            iwork[nsub - 1] = st;

            /* Subproblem found. First determine its size and then
             * apply divide and conquer on it.
             */
            if (i < nm1 - 1) {
                /* A subproblem with E[i] small for i < nm1-1. */
                nsize = i - st + 1;
                iwork[sizei + nsub - 1] = nsize;
            } else if (fabs(E[i]) >= eps) {
                /* A subproblem with E[nm1-1] not too small but i = nm1-1. */
                nsize = n - st;
                iwork[sizei + nsub - 1] = nsize;
            } else {
                /* A subproblem with E[nm1-1] small. This implies an
                 * 1-by-1 subproblem at D[n-1], which is not solved explicitly.
                 */
                nsize = i - st + 1;
                iwork[sizei + nsub - 1] = nsize;
                nsub = nsub + 1;
                iwork[nsub - 1] = n - 1;
                iwork[sizei + nsub - 1] = 1;
                cblas_dcopy(nrhs, &B[n - 1], ldb, &work[bx + n - 1], n);
            }
            st1 = st;
            if (nsize == 1) {
                /* This is a 1-by-1 subproblem and is not solved explicitly. */
                cblas_dcopy(nrhs, &B[st], ldb, &work[bx + st1], n);
            } else if (nsize <= smlsiz) {
                /* This is a small subproblem and is solved by DLASDQ. */
                dlaset("A", nsize, nsize, ZERO, ONE, &work[vt_idx + st1], n);
                /* NRU=0: U not referenced */
                dlasdq("U", 0, nsize, nsize, 0, nrhs, &D[st], &E[st],
                       &work[vt_idx + st1], n, NULL, n,
                       &B[st], ldb, &work[nwork], info);
                if (*info != 0) {
                    return;
                }
                dlacpy("A", nsize, nrhs, &B[st], ldb, &work[bx + st1], n);
            } else {
                /* A large problem. Solve it using divide and conquer. */
                dlasda(icmpq1, smlsiz, nsize, sqre, &D[st], &E[st],
                       &work[u_idx + st1], n, &work[vt_idx + st1],
                       &iwork[k_idx + st1], &work[difl_idx + st1],
                       &work[difr_idx + st1], &work[z_idx + st1],
                       &work[poles + st1], &iwork[givptr + st1],
                       &iwork[givcol + st1], n, &iwork[perm + st1],
                       &work[givnum + st1], &work[c_idx + st1],
                       &work[s_idx + st1], &work[nwork], &iwork[iwk], info);
                if (*info != 0) {
                    return;
                }
                bxst = bx + st1;
                dlalsa(icmpq2, smlsiz, nsize, nrhs, &B[st], ldb,
                       &work[bxst], n, &work[u_idx + st1], n,
                       &work[vt_idx + st1], &iwork[k_idx + st1],
                       &work[difl_idx + st1], &work[difr_idx + st1],
                       &work[z_idx + st1], &work[poles + st1],
                       &iwork[givptr + st1], &iwork[givcol + st1], n,
                       &iwork[perm + st1], &work[givnum + st1],
                       &work[c_idx + st1], &work[s_idx + st1], &work[nwork],
                       &iwork[iwk], info);
                if (*info != 0) {
                    return;
                }
            }
            st = i + 1;
        }
    }

    /* Apply the singular values and treat the tiny ones as zero. */
    tol = rcnd * fabs(D[cblas_idamax(n, D, 1)]);

    /* DO 70 I = 1, N (Fortran) -> i = 0..n-1 (C) */
    for (i = 0; i < n; i++) {
        /* Some of the elements in D can be negative because 1-by-1
         * subproblems were not solved explicitly.
         */
        if (fabs(D[i]) <= tol) {
            dlaset("A", 1, nrhs, ZERO, ZERO, &work[bx + i], n);
        } else {
            *rank = *rank + 1;
            dlascl("G", 0, 0, D[i], ONE, 1, nrhs, &work[bx + i], n, info);
        }
        D[i] = fabs(D[i]);
    }

    /* Now apply back the right singular vectors. */
    icmpq2 = 1;
    /* DO 80 I = 1, NSUB (Fortran) -> i = 0..nsub-1 (C) */
    for (i = 0; i < nsub; i++) {
        st = iwork[i];
        st1 = st;
        nsize = iwork[sizei + i];
        bxst = bx + st1;
        if (nsize == 1) {
            cblas_dcopy(nrhs, &work[bxst], n, &B[st], ldb);
        } else if (nsize <= smlsiz) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        nsize, nrhs, nsize, ONE, &work[vt_idx + st1], n,
                        &work[bxst], n, ZERO, &B[st], ldb);
        } else {
            dlalsa(icmpq2, smlsiz, nsize, nrhs, &work[bxst], n,
                   &B[st], ldb, &work[u_idx + st1], n,
                   &work[vt_idx + st1], &iwork[k_idx + st1],
                   &work[difl_idx + st1], &work[difr_idx + st1],
                   &work[z_idx + st1], &work[poles + st1],
                   &iwork[givptr + st1], &iwork[givcol + st1], n,
                   &iwork[perm + st1], &work[givnum + st1],
                   &work[c_idx + st1], &work[s_idx + st1], &work[nwork],
                   &iwork[iwk], info);
            if (*info != 0) {
                return;
            }
        }
    }

    /* Unscale and sort the singular values. */
    dlascl("G", 0, 0, ONE, orgnrm, n, 1, D, n, info);
    dlasrt("D", n, D, info);
    dlascl("G", 0, 0, orgnrm, ONE, n, nrhs, B, ldb, info);
}
