/**
 * @file slaed3.c
 * @brief SLAED3 finds the roots of the secular equation and updates the
 *        eigenvectors. Used when the original matrix is tridiagonal.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAED3 finds the roots of the secular equation, as defined by the
 * values in D, W, and RHO, between 0 and K-1. It makes the
 * appropriate calls to SLAED4 and then updates the eigenvectors by
 * multiplying the matrix of eigenvectors of the pair of eigensystems
 * being combined by the matrix of eigenvectors of the K-by-K system
 * which is solved here.
 *
 * @param[in]     k       The number of terms in the rational function to be
 *                        solved by SLAED4. k >= 0.
 * @param[in]     n       The number of rows and columns in the Q matrix.
 *                        n >= k (deflation may result in n > k).
 * @param[in]     n1      The location of the last eigenvalue in the leading
 *                        submatrix. min(1,n) <= n1 <= n/2.
 * @param[out]    D       Double precision array, dimension (n).
 *                        D[i] contains the updated eigenvalues for
 *                        0 <= i < k.
 * @param[out]    Q       Double precision array, dimension (ldq, n).
 *                        Initially the first k columns are used as workspace.
 *                        On output the columns 0 to k-1 contain
 *                        the updated eigenvectors.
 * @param[in]     ldq     The leading dimension of the array Q. ldq >= max(1,n).
 * @param[in]     rho     The value of the parameter in the rank one update
 *                        equation. rho >= 0 required.
 * @param[in]     dlambda Double precision array, dimension (k).
 *                        The first k elements of this array contain the old
 *                        roots of the deflated updating problem. These are the
 *                        poles of the secular equation.
 * @param[in]     Q2      Double precision array, dimension (ldq2*n).
 *                        The first k columns of this matrix contain the
 *                        non-deflated eigenvectors for the split problem.
 *                        Stored as a contiguous packed array.
 * @param[in]     indx    Integer array, dimension (n).
 *                        The permutation used to arrange the columns of the
 *                        deflated Q matrix into three groups (see SLAED2).
 *                        The rows of the eigenvectors found by SLAED4 must be
 *                        likewise permuted before the matrix multiply can take
 *                        place. (0-based)
 * @param[in]     ctot    Integer array, dimension (4).
 *                        A count of the total number of the various types of
 *                        columns in Q, as described in INDX. The fourth column
 *                        type is any column which has been deflated.
 * @param[in,out] W       Double precision array, dimension (k).
 *                        The first k elements of this array contain the
 *                        components of the deflation-adjusted updating vector.
 *                        Destroyed on output.
 * @param[out]    S       Double precision array, dimension (n1 + 1)*k.
 *                        Will contain the eigenvectors of the repaired matrix
 *                        which will be multiplied by the previously accumulated
 *                        eigenvectors to update the system.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = 1, an eigenvalue did not converge.
 */
void slaed3(const int k, const int n, const int n1, float* D,
            float* Q, const int ldq, const float rho,
            float* dlambda, float* Q2, int* indx, int* ctot,
            float* W, float* S, int* info)
{
    const float one = 1.0f;
    const float zero = 0.0f;

    int i, ii, iq2, j, n12, n2, n23;
    float temp;

    /* Test the input parameters. */
    *info = 0;

    if (k < 0) {
        *info = -1;
    } else if (n < k) {
        *info = -2;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SLAED3", -(*info));
        return;
    }

    /* Quick return if possible */
    if (k == 0)
        return;

    /* For each eigenvalue, call SLAED4 to find the root of the secular equation.
     * slaed4 takes 0-based index i (0 <= i < k). In Fortran, SLAED4 is called
     * with 1-based J, and writes into Q(1:K, J). Here we pass 0-based j and
     * point to Q column j. */
    for (j = 0; j < k; j++) {
        slaed4(k, j, dlambda, W, &Q[j * ldq], rho, &D[j], info);

        /* If the zero finder fails, the computation is terminated. */
        if (*info != 0)
            return;
    }

    if (k == 1)
        goto label_110;

    if (k == 2) {
        for (j = 0; j < k; j++) {
            W[0] = Q[0 + j * ldq];
            W[1] = Q[1 + j * ldq];
            ii = indx[0];
            Q[0 + j * ldq] = W[ii];
            ii = indx[1];
            Q[1 + j * ldq] = W[ii];
        }
        goto label_110;
    }

    /* Compute updated W. */
    /* Save W into S */
    cblas_scopy(k, W, 1, S, 1);

    /* Initialize W[i] = Q[i + i*ldq] (diagonal of Q) */
    cblas_scopy(k, Q, ldq + 1, W, 1);

    for (j = 0; j < k; j++) {
        for (i = 0; i < j; i++) {
            W[i] = W[i] * (Q[i + j * ldq] / (dlambda[i] - dlambda[j]));
        }
        for (i = j + 1; i < k; i++) {
            W[i] = W[i] * (Q[i + j * ldq] / (dlambda[i] - dlambda[j]));
        }
    }
    for (i = 0; i < k; i++) {
        W[i] = copysignf(sqrtf(-W[i]), S[i]);
    }

    /* Compute eigenvectors of the modified rank-1 modification. */
    for (j = 0; j < k; j++) {
        for (i = 0; i < k; i++) {
            S[i] = W[i] / Q[i + j * ldq];
        }
        temp = cblas_snrm2(k, S, 1);
        for (i = 0; i < k; i++) {
            ii = indx[i];
            Q[i + j * ldq] = S[ii] / temp;
        }
    }

    /* Compute the updated eigenvectors. */
label_110:

    n2 = n - n1;
    n12 = ctot[0] + ctot[1];
    n23 = ctot[1] + ctot[2];

    /* Copy rows ctot[0] .. ctot[0]+n23-1 of Q (first k columns) into S.
     * In Fortran: SLACPY('A', N23, K, Q(CTOT(1)+1, 1), LDQ, S, N23)
     * 0-based: row ctot[0], col 0 */
    slacpy("A", n23, k, &Q[ctot[0]], ldq, S, n23);

    /* IQ2 = N1*N12 in 0-based (Fortran: N1*N12 + 1, but 0-based offset) */
    iq2 = n1 * n12;
    if (n23 != 0) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n2, k, n23, one, &Q2[iq2], n2, S, n23,
                    zero, &Q[n1], ldq);
    } else {
        slaset("A", n2, k, zero, zero, &Q[n1], ldq);
    }

    slacpy("A", n12, k, Q, ldq, S, n12);
    if (n12 != 0) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n1, k, n12, one, Q2, n1, S, n12,
                    zero, Q, ldq);
    } else {
        slaset("A", n1, k, zero, zero, Q, ldq);
    }

    return;
}
