/**
 * @file slaed2.c
 * @brief SLAED2 merges eigenvalues and deflates the secular equation.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"


/**
 * SLAED2 merges the two sets of eigenvalues together into a single
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
 *                        SLAED3.
 * @param[in]     Z       Double precision array, dimension (N).
 *                        On entry, Z contains the updating vector (the last row
 *                        of the first sub-eigenvector matrix and the first row
 *                        of the second sub-eigenvector matrix).
 *                        On exit, the contents of Z have been destroyed by the
 *                        updating process.
 * @param[out]    dlambda Double precision array, dimension (N).
 *                        A copy of the first K eigenvalues which will be used
 *                        by SLAED3 to form the secular equation.
 * @param[out]    W       Double precision array, dimension (N).
 *                        The first K values of the final deflation-altered
 *                        z-vector which will be passed to SLAED3.
 * @param[out]    Q2      Double precision array, dimension (N1**2+(N-N1)**2).
 *                        A copy of the first K eigenvectors which will be used
 *                        by SLAED3 in a matrix multiply (DGEMM) to solve for
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
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal
 *                        value.
 */
void slaed2(int* K, const int n, const int n1, float* D, float* Q,
            const int ldq, int* indxq, float* rho, float* Z,
            float* dlambda, float* W, float* Q2, int* indx,
            int* indxc, int* indxp, int* coltyp, int* info)
{
    const float MONE = -1.0f;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float EIGHT = 8.0f;

    int ctot[4], psm[4];
    int ct, i, imax, iq1, iq2, j, jmax, js, k2, n2, nj, pj = 0;
    float c, eps, s, t, tau, tol;

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
        xerbla("SLAED2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    n2 = n - n1;

    if (*rho < ZERO) {
        cblas_sscal(n2, MONE, &Z[n1], 1);
    }

    /* Normalize z so that norm(z) = 1. Since z is the concatenation of
       two normalized vectors, norm2(z) = sqrt(2). */
    t = ONE / sqrtf(TWO);
    cblas_sscal(n, t, Z, 1);

    /* RHO = ABS( norm(z)**2 * RHO ) */
    *rho = fabsf(TWO * (*rho));

    /* Sort the eigenvalues into increasing order */
    for (i = n1; i < n; i++) {
        indxq[i] = indxq[i] + n1;
    }

    /* Re-integrate the deflated parts from the last pass */
    for (i = 0; i < n; i++) {
        dlambda[i] = D[indxq[i]];
    }
    slamrg(n1, n2, dlambda, 1, 1, indxc);
    for (i = 0; i < n; i++) {
        indx[i] = indxq[indxc[i]];
    }

    /* Calculate the allowable deflation tolerance */
    imax = cblas_isamax(n, Z, 1);
    jmax = cblas_isamax(n, D, 1);
    eps = slamch("Epsilon");
    tol = EIGHT * eps * (fabsf(D[jmax]) > fabsf(Z[imax]) ? fabsf(D[jmax]) : fabsf(Z[imax]));

    /* If the rank-1 modifier is small enough, no more needs to be done
       except to reorganize Q so that its columns correspond with the
       elements in D. */
    if ((*rho) * fabsf(Z[imax]) <= tol) {
        *K = 0;
        iq2 = 0;
        for (j = 0; j < n; j++) {
            i = indx[j];
            cblas_scopy(n, &Q[0 + i * ldq], 1, &Q2[iq2], 1);
            dlambda[j] = D[i];
            iq2 = iq2 + n;
        }
        slacpy("A", n, n, Q2, n, Q, ldq);
        cblas_scopy(n, dlambda, 1, D, 1);
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
        if ((*rho) * fabsf(Z[nj]) <= tol) {
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
    if ((*rho) * fabsf(Z[nj]) <= tol) {
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
        tau = slapy2(c, s);
        t = D[nj] - D[pj];
        c = c / tau;
        s = -s / tau;
        if (fabsf(t * c * s) <= tol) {
            /* Deflation is possible. */
            Z[nj] = tau;
            Z[pj] = ZERO;
            if (coltyp[nj] != coltyp[pj])
                coltyp[nj] = 2;
            coltyp[pj] = 4;
            cblas_srot(n, &Q[0 + pj * ldq], 1, &Q[0 + nj * ldq], 1, c, s);
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
        cblas_scopy(n1, &Q[0 + js * ldq], 1, &Q2[iq1], 1);
        Z[i] = D[js];
        i = i + 1;
        iq1 = iq1 + n1;
    }

    for (j = 0; j < ctot[1]; j++) {
        js = indx[i];
        cblas_scopy(n1, &Q[0 + js * ldq], 1, &Q2[iq1], 1);
        cblas_scopy(n2, &Q[n1 + js * ldq], 1, &Q2[iq2], 1);
        Z[i] = D[js];
        i = i + 1;
        iq1 = iq1 + n1;
        iq2 = iq2 + n2;
    }

    for (j = 0; j < ctot[2]; j++) {
        js = indx[i];
        cblas_scopy(n2, &Q[n1 + js * ldq], 1, &Q2[iq2], 1);
        Z[i] = D[js];
        i = i + 1;
        iq2 = iq2 + n2;
    }

    iq1 = iq2;
    for (j = 0; j < ctot[3]; j++) {
        js = indx[i];
        cblas_scopy(n, &Q[0 + js * ldq], 1, &Q2[iq2], 1);
        iq2 = iq2 + n;
        Z[i] = D[js];
        i = i + 1;
    }

    /* The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively. */
    if (*K < n) {
        slacpy("A", n, ctot[3], &Q2[iq1], n, &Q[0 + (*K) * ldq], ldq);
        cblas_scopy(n - *K, &Z[*K], 1, &D[*K], 1);
    }

    /* Copy CTOT into COLTYP for referencing in SLAED3. */
    for (j = 0; j < 4; j++) {
        coltyp[j] = ctot[j];
    }

L190:
    return;
}
