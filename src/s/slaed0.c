/**
 * @file slaed0.c
 * @brief SLAED0 computes all eigenvalues and corresponding eigenvectors of an
 *        unreduced symmetric tridiagonal matrix using the divide and conquer method.
 */

#include <math.h>
#include <stdint.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAED0 computes all eigenvalues and corresponding eigenvectors of a
 * symmetric tridiagonal matrix using the divide and conquer method.
 *
 * @param[in]     icompq = 0: Compute eigenvalues only.
 *                         = 1: Compute eigenvectors of original dense symmetric
 *                              matrix also. On entry, Q contains the orthogonal
 *                              matrix used to reduce the original matrix to
 *                              tridiagonal form.
 *                         = 2: Compute eigenvalues and eigenvectors of
 *                              tridiagonal matrix.
 * @param[in]     qsiz   The dimension of the orthogonal matrix used to reduce
 *                        the full matrix to tridiagonal form. qsiz >= n if
 *                        icompq = 1.
 * @param[in]     n      The dimension of the symmetric tridiagonal matrix.
 *                        n >= 0.
 * @param[in,out] D      Double precision array, dimension (n).
 *                        On entry, the main diagonal of the tridiagonal matrix.
 *                        On exit, its eigenvalues.
 * @param[in]     E      Double precision array, dimension (n-1).
 *                        The off-diagonal elements of the tridiagonal matrix.
 *                        On exit, E has been destroyed.
 * @param[in,out] Q      Double precision array, dimension (ldq, n).
 *                        On entry, Q must contain an n-by-n orthogonal matrix.
 *                        If icompq = 0, Q is not referenced.
 *                        If icompq = 1, on entry Q is a subset of the columns
 *                        of the orthogonal matrix used to reduce the full matrix
 *                        to tridiagonal form.
 *                        If icompq = 2, on entry Q will be the identity matrix.
 *                        On exit, Q contains the eigenvectors of the tridiagonal
 *                        matrix.
 * @param[in]     ldq    The leading dimension of the array Q. If eigenvectors
 *                        are desired, then ldq >= max(1,n). In any case,
 *                        ldq >= 1.
 * @param[out]    qstore Double precision array, dimension (ldqs, n).
 *                        Referenced only when icompq = 1. Used to store parts
 *                        of the eigenvector matrix when the updating matrix
 *                        multiplies take place.
 * @param[in]     ldqs   The leading dimension of the array qstore. If
 *                        icompq = 1, then ldqs >= max(1,n). In any case,
 *                        ldqs >= 1.
 * @param[out]    work   Double precision workspace array.
 *                        If icompq = 0 or 1, dimension at least
 *                        1 + 3*n + 2*n*lg(n) + 3*n^2.
 *                        If icompq = 2, dimension at least 4*n + n^2.
 * @param[out]    iwork  Integer workspace array.
 *                        If icompq = 0 or 1, dimension at least
 *                        6 + 6*n + 5*n*lg(n).
 *                        If icompq = 2, dimension at least 3 + 5*n.
 * @param[out]    info   = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal
 *                             value.
 *                        > 0: The algorithm failed to compute an eigenvalue
 *                             while working on the submatrix lying in rows and
 *                             columns info/(n+1) through mod(info,n+1).
 */
void slaed0(const int icompq, const int qsiz, const int n,
            float* D, float* E, float* Q, const int ldq,
            float* qstore, const int ldqs,
            float* work, int* iwork, int* info)
{
    /* SMLSIZ from ILAENV(9, 'SLAED0', ...) - hardcoded per project plan */
    const int SMLSIZ = 25;

    int curlvl, curprb, curr, i, igivcl = 0, igivnm = 0,
        igivpt = 0, indxq, iperm = 0, iprmpt = 0, iq = 0, iqptr = 0, iwrem = 0,
        j, k, lgn, matsiz, msd2, smm1, spm1,
        spm2, submat, subpbs, tlvls;
    float temp;

    /* Test the input parameters. */
    *info = 0;

    if (icompq < 0 || icompq > 2) {
        *info = -1;
    } else if (icompq == 1 && qsiz < (n > 0 ? n : 0)) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldqs < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("SLAED0", -(*info));
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
    /* Note: IWORK uses 0-based indexing in C */
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
        submat = iwork[i];  /* 0-based: Fortran IWORK(I+1), points to start of next subproblem */
        smm1 = submat - 1;
        /* D and E are 0-based */
        D[smm1] = D[smm1] - fabsf(E[smm1]);
        D[submat] = D[submat] - fabsf(E[smm1]);
    }

    indxq = 4 * n + 3;   /* 0-based offset into IWORK */
    if (icompq != 2) {
        /*
         * Set up workspaces for eigenvalues only/accumulate new vectors
         * routine
         */
        temp = logf((float)n) / logf(2.0f);
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

        igivnm = 0;                      /* 0-based offset into WORK */
        iq = igivnm + 2 * n * lgn;
        iwrem = iq + n * n;

        /* Initialize pointers (0-based) */
        for (i = 0; i <= subpbs; i++) {
            iwork[iprmpt + i] = 0;
            iwork[igivpt + i] = 0;
        }
        iwork[iqptr] = 0;
    }

    /*
     * Solve each submatrix eigenproblem at the bottom of the divide and
     * conquer tree.
     */
    curr = 0;
    for (i = 0; i <= spm1; i++) {
        if (i == 0) {
            submat = 0;  /* 0-based start index */
            matsiz = iwork[0];
        } else {
            submat = iwork[i - 1];  /* 0-based start */
            matsiz = iwork[i] - iwork[i - 1];
        }
        if (icompq == 2) {
            ssteqr("I", matsiz, &D[submat], &E[submat],
                   &Q[submat + submat * ldq], ldq, work, info);
            if (*info != 0) {
                goto L130;
            }
        } else {
            ssteqr("I", matsiz, &D[submat], &E[submat],
                   &work[iq + iwork[iqptr + curr]], matsiz, work, info);
            if (*info != 0) {
                goto L130;
            }
            if (icompq == 1) {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            qsiz, matsiz, matsiz, 1.0f,
                            &Q[submat * ldq], ldq,
                            &work[iq + iwork[iqptr + curr]], matsiz,
                            0.0f, &qstore[submat * ldqs], ldqs);
            }
            iwork[iqptr + curr + 1] = iwork[iqptr + curr] + matsiz * matsiz;
            curr = curr + 1;
        }
        /* Initialize indxq for this subproblem with 0-based indices */
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
             * into an eigensystem of size MATSIZ.
             * SLAED1 is used only for the full eigensystem of a tridiagonal
             * matrix.
             * SLAED7 handles the cases in which eigenvalues only or eigenvalues
             * and eigenvectors of a full symmetric matrix (which was reduced to
             * tridiagonal form) are desired.
             */
            if (icompq == 2) {
                slaed1(matsiz, &D[submat], &Q[submat + submat * ldq],
                       ldq, &iwork[indxq + submat],
                       E[submat + msd2 - 1], msd2, work,
                       &iwork[subpbs], info);
            } else {
                slaed7(icompq, matsiz, qsiz, tlvls, curlvl,
                       curprb,
                       &D[submat], &qstore[submat * ldqs], ldqs,
                       &iwork[indxq + submat], E[submat + msd2 - 1],
                       msd2, &work[iq], &iwork[iqptr],
                       &iwork[iprmpt], &iwork[iperm],
                       &iwork[igivpt], &iwork[igivcl],
                       &work[igivnm], &work[iwrem],
                       &iwork[subpbs], info);
            }
            if (*info != 0) {
                goto L130;
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
    if (icompq == 1) {
        for (i = 0; i < n; i++) {
            j = iwork[indxq + i];  /* Already 0-based */
            work[i] = D[j];
            cblas_scopy(qsiz, &qstore[j * ldqs], 1, &Q[i * ldq], 1);
        }
        cblas_scopy(n, work, 1, D, 1);
    } else if (icompq == 2) {
        for (i = 0; i < n; i++) {
            j = iwork[indxq + i];  /* Already 0-based */
            work[i] = D[j];
            cblas_scopy(n, &Q[j * ldq], 1, &work[n + (int64_t)n * i], 1);
        }
        cblas_scopy(n, work, 1, D, 1);
        slacpy("A", n, n, &work[n], n, Q, ldq);
    } else {
        for (i = 0; i < n; i++) {
            j = iwork[indxq + i];  /* Already 0-based */
            work[i] = D[j];
        }
        cblas_scopy(n, work, 1, D, 1);
    }
    return;

L130:
    *info = (submat + 1) * (n + 1) + submat + matsiz;
    return;
}
