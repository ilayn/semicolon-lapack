/**
 * @file zlaed0.c
 * @brief ZLAED0 used by ZSTEDC. Computes all eigenvalues and corresponding
 *        eigenvectors of an unreduced symmetric tridiagonal matrix using the
 *        divide and conquer method.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * Using the divide and conquer method, ZLAED0 computes all eigenvalues
 * of a symmetric tridiagonal matrix which is one diagonal block of
 * those from reducing a dense or band Hermitian matrix and
 * corresponding eigenvectors of the dense or band matrix.
 *
 * @param[in]     qsiz   The dimension of the unitary matrix used to reduce
 *                        the full matrix to tridiagonal form. QSIZ >= N.
 * @param[in]     n      The dimension of the symmetric tridiagonal matrix.
 *                        N >= 0.
 * @param[in,out] D      Double precision array, dimension (N).
 *                        On entry, the diagonal elements of the tridiagonal
 *                        matrix. On exit, the eigenvalues in ascending order.
 * @param[in,out] E      Double precision array, dimension (N-1).
 *                        On entry, the off-diagonal elements of the
 *                        tridiagonal matrix. On exit, E has been destroyed.
 * @param[in,out] Q      Complex*16 array, dimension (LDQ, N).
 *                        On entry, Q must contain a QSIZ x N matrix whose
 *                        columns are unitarily orthonormal.
 * @param[in]     ldq    The leading dimension of the array Q.
 *                        LDQ >= max(1,N).
 * @param[out]    qstore Complex*16 array, dimension (LDQS, N).
 *                        Used to store parts of the eigenvector matrix when
 *                        the updating matrix multiplies take place.
 * @param[in]     ldqs   The leading dimension of the array QSTORE.
 *                        LDQS >= max(1,N).
 * @param[out]    rwork  Double precision array,
 *                        dimension (1 + 3*N + 2*N*lg(N) + 3*N**2).
 * @param[out]    iwork  Integer array,
 *                        dimension (6 + 6*N + 5*N*lg(N)).
 * @param[out]    info   = 0: successful exit.
 *                        < 0: if INFO = -i, the i-th argument had an illegal
 *                             value.
 *                        > 0: The algorithm failed to compute an eigenvalue
 *                             while working on the submatrix lying in rows
 *                             and columns INFO/(N+1) through mod(INFO,N+1).
 */
void zlaed0(const int qsiz, const int n,
            double* D, double* E, double complex* Q, const int ldq,
            double complex* qstore, const int ldqs,
            double* rwork, int* iwork, int* info)
{
    const double TWO = 2.0;

    /* SMLSIZ from ILAENV(9, ...) */
    const int SMLSIZ = 25;

    int curlvl, curprb, curr, i, igivcl, igivnm,
        igivpt, indxq, iperm, iprmpt, iq, iqptr, iwrem,
        j, k, lgn, ll, matsiz, msd2, smm1, spm1,
        spm2, submat, subpbs, tlvls;
    double temp;

    /* Test the input parameters. */
    *info = 0;

    if (qsiz < (n > 0 ? n : 0)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldqs < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("ZLAED0", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /*
     * Determine the size and placement of the submatrices, and save in
     * the leading elements of IWORK.
     */
    iwork[0] = n;
    subpbs = 1;
    tlvls = 0;
    while (iwork[subpbs - 1] > SMLSIZ) {
        for (j = subpbs - 1; j >= 0; j--) {
            iwork[2 * j + 1] = (iwork[j] + 1) / 2;
            iwork[2 * j] = iwork[j] / 2;
        }
        tlvls = tlvls + 1;
        subpbs = 2 * subpbs;
    }
    for (j = 1; j < subpbs; j++) {
        iwork[j] = iwork[j] + iwork[j - 1];
    }

    /*
     * Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
     * using rank-1 modifications (cuts).
     */
    spm1 = subpbs - 1;
    for (i = 0; i < spm1; i++) {
        submat = iwork[i];
        smm1 = submat - 1;
        D[smm1] = D[smm1] - fabs(E[smm1]);
        D[submat] = D[submat] - fabs(E[smm1]);
    }

    indxq = 4 * n + 3;

    /*
     * Set up workspaces for eigenvalues only/accumulate new vectors
     * routine
     */
    temp = log((double)n) / log(TWO);
    lgn = (int)temp;
    if ((1 << lgn) < n) {
        lgn = lgn + 1;
    }
    if ((1 << lgn) < n) {
        lgn = lgn + 1;
    }
    iprmpt = indxq + n + 1;
    iperm = iprmpt + n * lgn;
    iqptr = iperm + n * lgn;
    igivpt = iqptr + n + 2;
    igivcl = igivpt + n * lgn;

    igivnm = 0;
    iq = igivnm + 2 * n * lgn;
    iwrem = iq + n * n;

    /* Initialize pointers */
    for (i = 0; i <= subpbs; i++) {
        iwork[iprmpt + i] = 0;
        iwork[igivpt + i] = 0;
    }
    iwork[iqptr] = 0;

    /*
     * Solve each submatrix eigenproblem at the bottom of the divide and
     * conquer tree.
     */
    curr = 0;
    for (i = 0; i <= spm1; i++) {
        if (i == 0) {
            submat = 0;
            matsiz = iwork[0];
        } else {
            submat = iwork[i - 1];
            matsiz = iwork[i] - iwork[i - 1];
        }
        ll = iq + iwork[iqptr + curr];
        dsteqr("I", matsiz, &D[submat], &E[submat],
               &rwork[ll], matsiz, rwork, info);
        zlacrm(qsiz, matsiz, &Q[submat * ldq], ldq, &rwork[ll],
               matsiz, &qstore[submat * ldqs], ldqs,
               &rwork[iwrem]);
        iwork[iqptr + curr + 1] = iwork[iqptr + curr] + matsiz * matsiz;
        curr = curr + 1;
        if (*info > 0) {
            *info = (submat + 1) * (n + 1) + submat + matsiz;
            return;
        }
        k = 0;
        for (j = submat; j < iwork[i]; j++) {
            iwork[indxq + j] = k;
            k = k + 1;
        }
    }

    /*
     * Successively merge eigensystems of adjacent submatrices
     * into eigensystem for the corresponding larger matrix.
     *
     * while ( SUBPBS > 1 )
     */
    curlvl = 1;
    while (subpbs > 1) {
        spm2 = subpbs - 2;
        for (i = 0; i <= spm2; i += 2) {
            if (i == 0) {
                submat = 0;
                matsiz = iwork[1];
                msd2 = iwork[0];
                curprb = 0;
            } else {
                submat = iwork[i - 1];
                matsiz = iwork[i + 1] - iwork[i - 1];
                msd2 = matsiz / 2;
                curprb = curprb + 1;
            }

            /*
             * Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
             * into an eigensystem of size MATSIZ.  ZLAED7 handles the case
             * when the eigenvectors of a full or band Hermitian matrix (which
             * was reduced to tridiagonal form) are desired.
             *
             * I am free to use Q as a valuable working space until Loop 150.
             */
            zlaed7(matsiz, msd2, qsiz, tlvls, curlvl, curprb,
                   &D[submat], &qstore[submat * ldqs], ldqs,
                   E[submat + msd2 - 1], &iwork[indxq + submat],
                   &rwork[iq], &iwork[iqptr], &iwork[iprmpt],
                   &iwork[iperm], &iwork[igivpt],
                   &iwork[igivcl], &rwork[igivnm],
                   &Q[submat * ldq], &rwork[iwrem],
                   &iwork[subpbs], info);
            if (*info > 0) {
                *info = (submat + 1) * (n + 1) + submat + matsiz;
                return;
            }
            iwork[i / 2] = iwork[i + 1];
        }
        subpbs = subpbs / 2;
        curlvl = curlvl + 1;
    }
    /* end while */

    /*
     * Re-merge the eigenvalues/vectors which were deflated at the final
     * merge step.
     */
    for (i = 0; i < n; i++) {
        j = iwork[indxq + i];
        rwork[i] = D[j];
        cblas_zcopy(qsiz, &qstore[j * ldqs], 1, &Q[i * ldq], 1);
    }
    cblas_dcopy(n, rwork, 1, D, 1);

    return;
}
