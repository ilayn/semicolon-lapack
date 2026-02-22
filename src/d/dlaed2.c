/**
 * @file dlaed2.c
 * @brief DLAED2 merges eigenvalues and deflates the secular equation.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"


/**
 * DLAED2 merges the two sets of eigenvalues together into a single
 * sorted set. Then it tries to deflate the size of the problem.
 * There are two ways in which deflation can occur: when two or more
 * eigenvalues are close together or if there is a tiny entry in the
 * Z vector. For each such occurrence the order of the related secular
 * equation problem is reduced by one.
 *
 * @param[out]    K       The number of non-deflated eigenvalues, and the order
 *                        of the related secular equation. 0 <= K <= N.
 * @param[in]     n       The dimension of the symmetric tridiagonal matrix.
 *                        N >= 0.
 * @param[in]     n1      The location of the last eigenvalue in the leading
 *                        sub-matrix. min(1,N) <= N1 <= N/2.
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, D contains the eigenvalues of the two
 *                        submatrices to be combined.
 *                        On exit, D contains the trailing (N-K) updated
 *                        eigenvalues (those which were deflated) sorted into
 *                        increasing order.
 * @param[in,out] Q       Double precision array, dimension (LDQ, N).
 *                        On entry, Q contains the eigenvectors of two
 *                        submatrices in the two square blocks with corners at
 *                        (0,0), (N1-1,N1-1) and (N1,N1), (N-1,N-1).
 *                        On exit, Q contains the trailing (N-K) updated
 *                        eigenvectors (those which were deflated) in its last
 *                        N-K columns.
 * @param[in]     ldq     The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[in,out] indxq   Integer array, dimension (N).
 *                        The permutation which separately sorts the two
 *                        sub-problems in D into ascending order. Note that
 *                        elements in the second half of this permutation must
 *                        first have N1 added to their values. Destroyed on exit.
 * @param[in,out] rho     On entry, the off-diagonal element associated with the
 *                        rank-1 cut which originally split the two submatrices
 *                        which are now being recombined.
 *                        On exit, rho has been modified to the value required by
 *                        DLAED3.
 * @param[in]     Z       Double precision array, dimension (N).
 *                        On entry, Z contains the updating vector (the last row
 *                        of the first sub-eigenvector matrix and the first row
 *                        of the second sub-eigenvector matrix).
 *                        On exit, the contents of Z have been destroyed by the
 *                        updating process.
 * @param[out]    dlambda Double precision array, dimension (N).
 *                        A copy of the first K eigenvalues which will be used
 *                        by DLAED3 to form the secular equation.
 * @param[out]    W       Double precision array, dimension (N).
 *                        The first K values of the final deflation-altered
 *                        z-vector which will be passed to DLAED3.
 * @param[out]    Q2      Double precision array, dimension (N1**2+(N-N1)**2).
 *                        A copy of the first K eigenvectors which will be used
 *                        by DLAED3 in a matrix multiply (DGEMM) to solve for
 *                        the new eigenvectors.
 * @param[out]    indx    Integer array, dimension (N).
 *                        The permutation used to sort the contents of DLAMBDA
 *                        into ascending order.
 * @param[out]    indxc   Integer array, dimension (N).
 *                        The permutation used to arrange the columns of the
 *                        deflated Q matrix into three groups: the first group
 *                        contains non-zero elements only at and above N1, the
 *                        second contains non-zero elements only below N1, and
 *                        the third is dense.
 * @param[out]    indxp   Integer array, dimension (N).
 *                        The permutation used to place deflated values of D at
 *                        the end of the array. INDXP(0:K-1) points to the
 *                        nondeflated D-values and INDXP(K:N-1) points to the
 *                        deflated eigenvalues.
 * @param[out]    coltyp  Integer array, dimension (N).
 *                        During execution, a label which will indicate which of
 *                        the following types a column in the Q2 matrix is:
 *                        1: non-zero in the upper half only;
 *                        2: dense;
 *                        3: non-zero in the lower half only;
 *                        4: deflated.
 *                        On exit, COLTYP(i) is the number of columns of type i,
 *                        for i=0 to 3 only.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value.
 */
void dlaed2(INT* K, const INT n, const INT n1, f64* D, f64* Q,
            const INT ldq, INT* indxq, f64* rho, f64* Z,
            f64* dlambda, f64* W, f64* Q2, INT* indx,
            INT* indxc, INT* indxp, INT* coltyp, INT* info)
{
    const f64 MONE = -1.0;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 EIGHT = 8.0;

    INT ctot[4], psm[4];
    INT ct, i, imax, iq1, iq2, j, jmax, js, k2, n2, nj, pj = 0;
    f64 c, eps, s, t, tau, tol;

    /* Test the input parameters. */
    *info = 0;

    if (n < 0) {
        *info = -2;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -6;
    } else if ((1 < n / 2 ? 1 : n / 2) > n1 || n / 2 < n1) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("DLAED2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    n2 = n - n1;

    if (*rho < ZERO) {
        cblas_dscal(n2, MONE, &Z[n1], 1);
    }

    /* Normalize z so that norm(z) = 1. Since z is the concatenation of
       two normalized vectors, norm2(z) = sqrt(2). */
    t = ONE / sqrt(TWO);
    cblas_dscal(n, t, Z, 1);

    /* RHO = ABS( norm(z)**2 * RHO ) */
    *rho = fabs(TWO * (*rho));

    /* Sort the eigenvalues into increasing order */
    for (i = n1; i < n; i++) {
        indxq[i] = indxq[i] + n1;
    }

    /* Re-integrate the deflated parts from the last pass */
    for (i = 0; i < n; i++) {
        dlambda[i] = D[indxq[i]];
    }
    dlamrg(n1, n2, dlambda, 1, 1, indxc);
    for (i = 0; i < n; i++) {
        indx[i] = indxq[indxc[i]];
    }

    /* Calculate the allowable deflation tolerance */
    imax = cblas_idamax(n, Z, 1);
    jmax = cblas_idamax(n, D, 1);
    eps = dlamch("Epsilon");
    tol = EIGHT * eps * (fabs(D[jmax]) > fabs(Z[imax]) ? fabs(D[jmax]) : fabs(Z[imax]));

    /* If the rank-1 modifier is small enough, no more needs to be done
       except to reorganize Q so that its columns correspond with the
       elements in D. */
    if ((*rho) * fabs(Z[imax]) <= tol) {
        *K = 0;
        iq2 = 0;
        for (j = 0; j < n; j++) {
            i = indx[j];
            cblas_dcopy(n, &Q[0 + i * ldq], 1, &Q2[iq2], 1);
            dlambda[j] = D[i];
            iq2 = iq2 + n;
        }
        dlacpy("A", n, n, Q2, n, Q, ldq);
        cblas_dcopy(n, dlambda, 1, D, 1);
        goto L190;
    }

    /* If there are multiple eigenvalues then the problem deflates. Here
       the number of equal eigenvalues are found. As each equal
       eigenvalue is found, an elementary reflector is computed to rotate
       the corresponding eigensubspace so that the corresponding
       components of Z are zero in this new basis. */
    for (i = 0; i < n1; i++) {
        coltyp[i] = 1;
    }
    for (i = n1; i < n; i++) {
        coltyp[i] = 3;
    }

    *K = 0;
    k2 = n;
    for (j = 0; j < n; j++) {
        nj = indx[j];
        if ((*rho) * fabs(Z[nj]) <= tol) {
            /* Deflate due to small z component. */
            k2 = k2 - 1;
            coltyp[nj] = 4;
            indxp[k2] = nj;
            if (j == n - 1)
                goto L100;
        } else {
            pj = nj;
            goto L80;
        }
    }
    /* If we reach here, all eigenvalues were deflated via small Z */
    goto L100;
L80:
    j = j + 1;
    if (j > n - 1) {
        goto L100;
    }
    nj = indx[j];
    if ((*rho) * fabs(Z[nj]) <= tol) {
        /* Deflate due to small z component. */
        k2 = k2 - 1;
        coltyp[nj] = 4;
        indxp[k2] = nj;
    } else {
        /* Check if eigenvalues are close enough to allow deflation. */
        s = Z[pj];
        c = Z[nj];

        /* Find sqrt(a**2+b**2) without overflow or
           destructive underflow. */
        tau = dlapy2(c, s);
        t = D[nj] - D[pj];
        c = c / tau;
        s = -s / tau;
        if (fabs(t * c * s) <= tol) {
            /* Deflation is possible. */
            Z[nj] = tau;
            Z[pj] = ZERO;
            if (coltyp[nj] != coltyp[pj])
                coltyp[nj] = 2;
            coltyp[pj] = 4;
            cblas_drot(n, &Q[0 + pj * ldq], 1, &Q[0 + nj * ldq], 1, c, s);
            t = D[pj] * c * c + D[nj] * s * s;
            D[nj] = D[pj] * s * s + D[nj] * c * c;
            D[pj] = t;
            k2 = k2 - 1;
            i = 1;
L90:
            if (k2 + i <= n - 1) {
                if (D[pj] < D[indxp[k2 + i]]) {
                    indxp[k2 + i - 1] = indxp[k2 + i];
                    indxp[k2 + i] = pj;
                    i = i + 1;
                    goto L90;
                } else {
                    indxp[k2 + i - 1] = pj;
                }
            } else {
                indxp[k2 + i - 1] = pj;
            }
            pj = nj;
        } else {
            *K = *K + 1;
            dlambda[*K - 1] = D[pj];
            W[*K - 1] = Z[pj];
            indxp[*K - 1] = pj;
            pj = nj;
        }
    }
    goto L80;
L100:

    /* Record the last eigenvalue. */
    *K = *K + 1;
    dlambda[*K - 1] = D[pj];
    W[*K - 1] = Z[pj];
    indxp[*K - 1] = pj;

    /* Count up the total number of the various types of columns, then
       form a permutation which positions the four column types into
       four uniform groups (although one or more of these groups may be
       empty). */
    for (j = 0; j < 4; j++) {
        ctot[j] = 0;
    }
    for (j = 0; j < n; j++) {
        ct = coltyp[j];
        ctot[ct - 1] = ctot[ct - 1] + 1;
    }

    /* PSM(*) = Position in SubMatrix (of types 1 through 4) */
    psm[0] = 0;
    psm[1] = ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    *K = n - ctot[3];

    /* Fill out the INDXC array so that the permutation which it induces
       will place all type-1 columns first, all type-2 columns next,
       then all type-3's, and finally all type-4's. */
    for (j = 0; j < n; j++) {
        js = indxp[j];
        ct = coltyp[js];
        indx[psm[ct - 1]] = js;
        indxc[psm[ct - 1]] = j;
        psm[ct - 1] = psm[ct - 1] + 1;
    }

    /* Sort the eigenvalues and corresponding eigenvectors into DLAMBDA
       and Q2 respectively. The eigenvalues/vectors which were not
       deflated go into the first K slots of DLAMBDA and Q2 respectively,
       while those which were deflated go into the last N - K slots. */
    i = 0;
    iq1 = 0;
    iq2 = (ctot[0] + ctot[1]) * n1;
    for (j = 0; j < ctot[0]; j++) {
        js = indx[i];
        cblas_dcopy(n1, &Q[0 + js * ldq], 1, &Q2[iq1], 1);
        Z[i] = D[js];
        i = i + 1;
        iq1 = iq1 + n1;
    }

    for (j = 0; j < ctot[1]; j++) {
        js = indx[i];
        cblas_dcopy(n1, &Q[0 + js * ldq], 1, &Q2[iq1], 1);
        cblas_dcopy(n2, &Q[n1 + js * ldq], 1, &Q2[iq2], 1);
        Z[i] = D[js];
        i = i + 1;
        iq1 = iq1 + n1;
        iq2 = iq2 + n2;
    }

    for (j = 0; j < ctot[2]; j++) {
        js = indx[i];
        cblas_dcopy(n2, &Q[n1 + js * ldq], 1, &Q2[iq2], 1);
        Z[i] = D[js];
        i = i + 1;
        iq2 = iq2 + n2;
    }

    iq1 = iq2;
    for (j = 0; j < ctot[3]; j++) {
        js = indx[i];
        cblas_dcopy(n, &Q[0 + js * ldq], 1, &Q2[iq2], 1);
        iq2 = iq2 + n;
        Z[i] = D[js];
        i = i + 1;
    }

    /* The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively. */
    if (*K < n) {
        dlacpy("A", n, ctot[3], &Q2[iq1], n, &Q[0 + (*K) * ldq], ldq);
        cblas_dcopy(n - *K, &Z[*K], 1, &D[*K], 1);
    }

    /* Copy CTOT into COLTYP for referencing in DLAED3. */
    for (j = 0; j < 4; j++) {
        coltyp[j] = ctot[j];
    }

L190:
    return;
}
