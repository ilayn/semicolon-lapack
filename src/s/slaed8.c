/**
 * @file slaed8.c
 * @brief SLAED8 merges eigenvalues and deflates the secular equation.
 *        Used when the original matrix is dense.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"


/**
 * SLAED8 merges the two sets of eigenvalues together into a single
 * sorted set. Then it tries to deflate the size of the problem.
 * There are two ways in which deflation can occur: when two or more
 * eigenvalues are close together or if there is a tiny element in the
 * Z vector. For each such occurrence the order of the related secular
 * equation problem is reduced by one.
 *
 * @param[in]     icompq  = 0: Compute eigenvalues only.
 *                         = 1: Compute eigenvectors of original dense symmetric
 *                              matrix also. On entry, Q contains the orthogonal
 *                              matrix used to reduce the original matrix to
 *                              tridiagonal form.
 * @param[out]    K       The number of non-deflated eigenvalues, and the order
 *                        of the related secular equation.
 * @param[in]     n       The dimension of the symmetric tridiagonal matrix.
 *                        N >= 0.
 * @param[in]     qsiz    The dimension of the orthogonal matrix used to reduce
 *                        the full matrix to tridiagonal form. QSIZ >= N if
 *                        ICOMPQ = 1.
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, the eigenvalues of the two submatrices to be
 *                        combined.
 *                        On exit, the trailing (N-K) updated eigenvalues
 *                        (those which were deflated) sorted into increasing order.
 * @param[in,out] Q       Double precision array, dimension (LDQ, N).
 *                        If ICOMPQ = 0, Q is not referenced. Otherwise,
 *                        on entry, Q contains the eigenvectors of the partially
 *                        solved system which has been previously updated in matrix
 *                        multiplies with other partially solved eigensystems.
 *                        On exit, Q contains the trailing (N-K) updated
 *                        eigenvectors (those which were deflated) in its last
 *                        N-K columns.
 * @param[in]     ldq     The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[in]     indxq   Integer array, dimension (N).
 *                        The permutation which separately sorts the two
 *                        sub-problems in D into ascending order. Note that
 *                        elements in the second half of this permutation must
 *                        first have CUTPNT added to their values in order to be
 *                        accurate.
 * @param[in,out] rho     On entry, the off-diagonal element associated with the
 *                        rank-1 cut which originally split the two submatrices
 *                        which are now being recombined.
 *                        On exit, rho has been modified to the value required by
 *                        SLAED3.
 * @param[in]     cutpnt  The location of the last eigenvalue in the leading
 *                        sub-matrix. min(1,N) <= CUTPNT <= N.
 * @param[in,out] Z       Double precision array, dimension (N).
 *                        On entry, Z contains the updating vector (the last row
 *                        of the first sub-eigenvector matrix and the first row of
 *                        the second sub-eigenvector matrix).
 *                        On exit, the contents of Z are destroyed by the updating
 *                        process.
 * @param[out]    dlambda Double precision array, dimension (N).
 *                        A copy of the first K eigenvalues which will be used by
 *                        SLAED3 to form the secular equation.
 * @param[out]    Q2      Double precision array, dimension (LDQ2, N).
 *                        If ICOMPQ = 0, Q2 is not referenced. Otherwise,
 *                        a copy of the first K eigenvectors which will be used by
 *                        SLAED7 in a matrix multiply (DGEMM) to update the new
 *                        eigenvectors.
 * @param[in]     ldq2    The leading dimension of the array Q2. LDQ2 >= max(1,N).
 * @param[out]    W       Double precision array, dimension (N).
 *                        The first k values of the final deflation-altered z-vector
 *                        and will be passed to SLAED3.
 * @param[out]    perm    Integer array, dimension (N).
 *                        The permutations (from deflation and sorting) to be applied
 *                        to each eigenblock.
 * @param[out]    givptr  The number of Givens rotations which took place in this
 *                        subproblem.
 * @param[out]    givcol  Integer array, dimension (2 * N).
 *                        Each pair of numbers indicates a pair of columns to take
 *                        place in a Givens rotation. Stored column-major with
 *                        leading dimension 2.
 * @param[out]    givnum  Double precision array, dimension (2 * N).
 *                        Each number indicates the C and S values to be used in the
 *                        corresponding Givens rotation. Stored column-major with
 *                        leading dimension 2.
 * @param[out]    indxp   Integer array, dimension (N).
 *                        The permutation used to place deflated values of D at the
 *                        end of the array. INDXP(0:K-1) points to the nondeflated
 *                        D-values and INDXP(K:N-1) points to the deflated
 *                        eigenvalues.
 * @param[out]    indx    Integer array, dimension (N).
 *                        The permutation used to sort the contents of D into
 *                        ascending order.
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void slaed8(const int icompq, int* K, const int n, const int qsiz,
            float* D, float* Q, const int ldq, int* indxq, float* rho,
            const int cutpnt, float* Z, float* dlambda, float* Q2,
            const int ldq2, float* W, int* perm, int* givptr,
            int* givcol, float* givnum, int* indxp, int* indx, int* info)
{
    const float MONE = -1.0f;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float EIGHT = 8.0f;

    int i, imax, j, jlam, jmax, jp, k2, n1, n2;
    float c, eps, s, t, tau, tol;

    /* Test the input parameters. */
    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (n < 0) {
        *info = -3;
    } else if (icompq == 1 && qsiz < n) {
        *info = -4;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (cutpnt < (1 < n ? 1 : n) || cutpnt > n) {
        *info = -10;
    } else if (ldq2 < (1 > n ? 1 : n)) {
        *info = -14;
    }
    if (*info != 0) {
        xerbla("SLAED8", -(*info));
        return;
    }

    /* Need to initialize GIVPTR to 0 here in case of quick exit
       to prevent an unspecified code behavior (usually sigfault)
       when IWORK array on entry to *stedc is not zeroed
       (or at least some IWORK entries which used in *laed7 for GIVPTR). */
    *givptr = 0;

    /* Quick return if possible */
    if (n == 0)
        return;

    n1 = cutpnt;
    n2 = n - n1;

    if (*rho < ZERO) {
        cblas_sscal(n2, MONE, &Z[n1], 1);
    }

    /* Normalize z so that norm(z) = 1 */
    t = ONE / sqrtf(TWO);
    for (j = 0; j < n; j++) {
        indx[j] = j;
    }
    cblas_sscal(n, t, Z, 1);
    *rho = fabsf(TWO * (*rho));

    /* Sort the eigenvalues into increasing order */
    for (i = cutpnt; i < n; i++) {
        indxq[i] = indxq[i] + cutpnt;
    }
    for (i = 0; i < n; i++) {
        dlambda[i] = D[indxq[i]];
        W[i] = Z[indxq[i]];
    }
    slamrg(n1, n2, dlambda, 1, 1, indx);
    for (i = 0; i < n; i++) {
        D[i] = dlambda[indx[i]];
        Z[i] = W[indx[i]];
    }

    /* Calculate the allowable deflation tolerance */
    imax = cblas_isamax(n, Z, 1);
    jmax = cblas_isamax(n, D, 1);
    eps = slamch("Epsilon");
    tol = EIGHT * eps * fabsf(D[jmax]);

    /* If the rank-1 modifier is small enough, no more needs to be done
       except to reorganize Q so that its columns correspond with the
       elements in D. */
    if ((*rho) * fabsf(Z[imax]) <= tol) {
        *K = 0;
        if (icompq == 0) {
            for (j = 0; j < n; j++) {
                perm[j] = indxq[indx[j]];
            }
        } else {
            for (j = 0; j < n; j++) {
                perm[j] = indxq[indx[j]];
                cblas_scopy(qsiz, &Q[0 + perm[j] * ldq], 1,
                            &Q2[0 + j * ldq2], 1);
            }
            slacpy("A", qsiz, n, Q2, ldq2, Q, ldq);
        }
        return;
    }

    /* If there are multiple eigenvalues then the problem deflates. Here
       the number of equal eigenvalues are found. As each equal
       eigenvalue is found, an elementary reflector is computed to rotate
       the corresponding eigensubspace so that the corresponding
       components of Z are zero in this new basis. */
    *K = 0;
    k2 = n;
    for (j = 0; j < n; j++) {
        if ((*rho) * fabsf(Z[j]) <= tol) {
            /* Deflate due to small z component. */
            k2 = k2 - 1;
            indxp[k2] = j;
            if (j == n - 1)
                goto L110;
        } else {
            jlam = j;
            goto L80;
        }
    }
L80:
    j = j + 1;
    if (j > n - 1)
        goto L100;
    if ((*rho) * fabsf(Z[j]) <= tol) {
        /* Deflate due to small z component. */
        k2 = k2 - 1;
        indxp[k2] = j;
    } else {
        /* Check if eigenvalues are close enough to allow deflation. */
        s = Z[jlam];
        c = Z[j];

        /* Find sqrt(a**2+b**2) without overflow or
           destructive underflow. */
        tau = slapy2(c, s);
        t = D[j] - D[jlam];
        c = c / tau;
        s = -s / tau;
        if (fabsf(t * c * s) <= tol) {
            /* Deflation is possible. */
            Z[j] = tau;
            Z[jlam] = ZERO;

            /* Record the appropriate Givens rotation */
            *givptr = *givptr + 1;
            givcol[0 + (*givptr - 1) * 2] = indxq[indx[jlam]];
            givcol[1 + (*givptr - 1) * 2] = indxq[indx[j]];
            givnum[0 + (*givptr - 1) * 2] = c;
            givnum[1 + (*givptr - 1) * 2] = s;
            if (icompq == 1) {
                cblas_srot(qsiz, &Q[0 + indxq[indx[jlam]] * ldq], 1,
                           &Q[0 + indxq[indx[j]] * ldq], 1, c, s);
            }
            t = D[jlam] * c * c + D[j] * s * s;
            D[j] = D[jlam] * s * s + D[j] * c * c;
            D[jlam] = t;
            k2 = k2 - 1;
            i = 0;
L90:
            if (k2 + i < n) {
                if (D[jlam] < D[indxp[k2 + i]]) {
                    indxp[k2 + i - 1] = indxp[k2 + i];
                    indxp[k2 + i] = jlam;
                    i = i + 1;
                    goto L90;
                } else {
                    indxp[k2 + i - 1] = jlam;
                }
            } else {
                indxp[k2 + i - 1] = jlam;
            }
            jlam = j;
        } else {
            *K = *K + 1;
            W[*K - 1] = Z[jlam];
            dlambda[*K - 1] = D[jlam];
            indxp[*K - 1] = jlam;
            jlam = j;
        }
    }
    goto L80;
L100:

    /* Record the last eigenvalue. */
    *K = *K + 1;
    W[*K - 1] = Z[jlam];
    dlambda[*K - 1] = D[jlam];
    indxp[*K - 1] = jlam;

L110:

    /* Sort the eigenvalues and corresponding eigenvectors into DLAMBDA
       and Q2 respectively. The eigenvalues/vectors which were not
       deflated go into the first K slots of DLAMBDA and Q2 respectively,
       while those which were deflated go into the last N - K slots. */
    if (icompq == 0) {
        for (j = 0; j < n; j++) {
            jp = indxp[j];
            dlambda[j] = D[jp];
            perm[j] = indxq[indx[jp]];
        }
    } else {
        for (j = 0; j < n; j++) {
            jp = indxp[j];
            dlambda[j] = D[jp];
            perm[j] = indxq[indx[jp]];
            cblas_scopy(qsiz, &Q[0 + perm[j] * ldq], 1,
                        &Q2[0 + j * ldq2], 1);
        }
    }

    /* The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively. */
    if (*K < n) {
        if (icompq == 0) {
            cblas_scopy(n - *K, &dlambda[*K], 1, &D[*K], 1);
        } else {
            cblas_scopy(n - *K, &dlambda[*K], 1, &D[*K], 1);
            slacpy("A", qsiz, n - *K, &Q2[0 + (*K) * ldq2], ldq2,
                   &Q[0 + (*K) * ldq], ldq);
        }
    }

    return;
}
