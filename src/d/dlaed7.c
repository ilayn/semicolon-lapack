/**
 * @file dlaed7.c
 * @brief DLAED7 computes the updated eigensystem of a diagonal matrix after
 *        modification by a rank-one symmetric matrix. Used when the original
 *        matrix is dense.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"


/**
 * DLAED7 computes the updated eigensystem of a diagonal
 * matrix after modification by a rank-one symmetric matrix. This
 * routine is used only for the eigenproblem which requires all
 * eigenvalues and optionally eigenvectors of a dense symmetric matrix
 * that has been reduced to tridiagonal form.  DLAED1 handles
 * the case in which all eigenvalues and eigenvectors of a symmetric
 * tridiagonal matrix are desired.
 *
 *   T = Q(in) ( D(in) + RHO * Z*Z**T ) Q**T(in) = Q(out) * D(out) * Q**T(out)
 *
 *    where Z = Q**Tu, u is a vector of length N with ones in the
 *    CUTPNT-1 and CUTPNT th elements and zeros elsewhere (0-based).
 *
 *    The eigenvectors of the original matrix are stored in Q, and the
 *    eigenvalues are in D.  The algorithm consists of three stages:
 *
 *       The first stage consists of deflating the size of the problem
 *       when there are multiple eigenvalues or if there is a zero in
 *       the Z vector.  For each such occurrence the dimension of the
 *       secular equation problem is reduced by one.  This stage is
 *       performed by the routine DLAED8.
 *
 *       The second stage consists of calculating the updated
 *       eigenvalues. This is done by finding the roots of the secular
 *       equation via the routine DLAED4 (as called by DLAED9).
 *       This routine also calculates the eigenvectors of the current
 *       problem.
 *
 *       The final stage consists of computing the updated eigenvectors
 *       directly using the updated eigenvalues.  The eigenvectors for
 *       the current problem are multiplied with the eigenvectors from
 *       the overall problem.
 *
 * @param[in]     icompq  = 0: Compute eigenvalues only.
 *                         = 1: Compute eigenvectors of original dense symmetric
 *                              matrix also. On entry, Q contains the orthogonal
 *                              matrix used to reduce the original matrix to
 *                              tridiagonal form.
 * @param[in]     n       The dimension of the symmetric tridiagonal matrix.
 *                        N >= 0.
 * @param[in]     qsiz    The dimension of the orthogonal matrix used to reduce
 *                        the full matrix to tridiagonal form. QSIZ >= N if
 *                        ICOMPQ = 1.
 * @param[in]     tlvls   The total number of merging levels in the overall
 *                        divide and conquer tree.
 * @param[in]     curlvl  The current level in the overall merge routine,
 *                        0 <= curlvl <= tlvls.
 * @param[in]     curpbm  The current problem in the current level in the overall
 *                        merge routine (counting from upper left to lower right).
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, the eigenvalues of the rank-1-perturbed matrix.
 *                        On exit, the eigenvalues of the repaired matrix.
 * @param[in,out] Q       Double precision array, dimension (LDQ, N).
 *                        On entry, the eigenvectors of the rank-1-perturbed matrix.
 *                        On exit, the eigenvectors of the repaired tridiagonal matrix.
 * @param[in]     ldq     The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[out]    indxq   Integer array, dimension (N).
 *                        The permutation which will reintegrate the subproblem just
 *                        solved back into sorted order, i.e., D( INDXQ( I ) )
 *                        will be in ascending order (0-based indices).
 * @param[in]     rho     The subdiagonal element used to create the rank-1
 *                        modification.
 * @param[in]     cutpnt  The location of the last eigenvalue in the leading
 *                        sub-matrix. min(1,N) <= CUTPNT <= N.
 * @param[in,out] qstore  Double precision array, dimension (N**2+1).
 *                        Stores eigenvectors of submatrices encountered during
 *                        divide and conquer, packed together. QPTR points to
 *                        beginning of the submatrices (0-based offsets).
 * @param[in,out] qptr    Integer array, dimension (N+2).
 *                        List of indices (0-based) pointing to beginning of
 *                        submatrices stored in QSTORE.
 * @param[in]     prmptr  Integer array, dimension (N lg N).
 *                        Contains a list of pointers (0-based) which indicate
 *                        where in PERM a level's permutation is stored.
 * @param[in]     perm    Integer array, dimension (N lg N).
 *                        Contains the permutations (from deflation and sorting)
 *                        to be applied to each eigenblock (0-based indices).
 * @param[in]     givptr  Integer array, dimension (N lg N).
 *                        Contains a list of pointers (0-based) which indicate
 *                        where in GIVCOL a level's Givens rotations are stored.
 * @param[in]     givcol  Integer array, dimension (2 * N lg N).
 *                        Each pair of numbers indicates a pair of columns to
 *                        take place in a Givens rotation (0-based indices).
 * @param[in]     givnum  Double precision array, dimension (2 * N lg N).
 *                        Each number indicates the S value to be used in the
 *                        corresponding Givens rotation.
 * @param[out]    work    Double precision array, dimension (3*N+2*QSIZ*N).
 * @param[out]    iwork   Integer array, dimension (4*N).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 *                        > 0: if info = 1, an eigenvalue did not converge.
 */
void dlaed7(const int icompq, const int n, const int qsiz,
            const int tlvls, const int curlvl, const int curpbm,
            double* D, double* Q, const int ldq, int* indxq,
            const double rho, const int cutpnt, double* qstore,
            int* qptr, int* prmptr, int* perm, int* givptr,
            int* givcol, double* givnum, double* work, int* iwork,
            int* info)
{
    int coltyp, curr, i, idlmda, indx, indxc, indxp,
        iq2, is, iw, iz, k, ldq2, n1, n2, ptr;
    double rho_local;

    /* Test the input parameters. */
    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (icompq == 1 && qsiz < n) {
        *info = -3;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -9;
    } else if ((1 < n ? 1 : n) > cutpnt || n < cutpnt) {
        *info = -12;
    }
    if (*info != 0) {
        xerbla("DLAED7", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /*
     * The following values are for bookkeeping purposes only. They are
     * integer pointers which indicate the portion of the workspace
     * used by a particular array in DLAED8 and DLAED9.
     */
    if (icompq == 1) {
        ldq2 = qsiz;
    } else {
        ldq2 = n;
    }

    iz = 0;
    idlmda = iz + n;
    iw = idlmda + n;
    iq2 = iw + n;
    is = iq2 + n * ldq2;

    indx = 0;
    indxc = indx + n;
    coltyp = indxc + n;
    indxp = coltyp + n;

    /*
     * Form the z-vector which consists of the last row of Q_1 and the
     * first row of Q_2.
     */

    /* PTR = 1 + 2**TLVLS in Fortran (1-based).
     * In 0-based: ptr = (1 << tlvls). The tree has 2^TLVLS leaf nodes at
     * level 0. The pointer for level curlvl, problem curpbm is computed
     * by walking the tree. */
    ptr = 1 << tlvls;
    for (i = 0; i < curlvl - 1; i++) {
        ptr = ptr + (1 << (tlvls - 1 - i));
    }
    curr = ptr + curpbm;

    dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr,
           givcol, givnum, qstore, qptr, &work[iz],
           &work[iz + n], info);

    /*
     * When solving the final problem, we no longer need the stored data,
     * so we will overwrite the data from this level onto the previously
     * used storage space.
     */
    if (curlvl == tlvls) {
        qptr[curr] = 0;
        prmptr[curr] = 0;
        givptr[curr] = 0;
    }

    /*
     * Sort and Deflate eigenvalues.
     */
    rho_local = rho;
    dlaed8(icompq, &k, n, qsiz, D, Q, ldq, indxq, &rho_local,
           cutpnt, &work[iz], &work[idlmda], &work[iq2], ldq2,
           &work[iw], &perm[prmptr[curr]], &givptr[curr + 1],
           &givcol[2 * givptr[curr]],
           &givnum[2 * givptr[curr]], &iwork[indxp],
           &iwork[indx], info);
    prmptr[curr + 1] = prmptr[curr] + n;
    givptr[curr + 1] = givptr[curr + 1] + givptr[curr];

    /*
     * Solve Secular Equation.
     */
    if (k != 0) {
        dlaed9(k, 0, k, n, D, &work[is], k, rho_local,
               &work[idlmda], &work[iw],
               &qstore[qptr[curr]], k, info);
        if (*info != 0)
            return;

        if (icompq == 1) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        qsiz, k, k, 1.0, &work[iq2], ldq2,
                        &qstore[qptr[curr]], k, 0.0, Q, ldq);
        }
        qptr[curr + 1] = qptr[curr] + k * k;

        /* Prepare the INDXQ sorting permutation. */
        n1 = k;
        n2 = n - k;
        dlamrg(n1, n2, D, 1, -1, indxq);
    } else {
        qptr[curr + 1] = qptr[curr];
        for (i = 0; i < n; i++) {
            indxq[i] = i;
        }
    }

    return;
}
