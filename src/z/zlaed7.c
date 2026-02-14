/**
 * @file zlaed7.c
 * @brief ZLAED7 computes the updated eigensystem of a diagonal matrix after
 *        modification by a rank-one symmetric matrix. Used when the original
 *        matrix is dense.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"


/**
 * ZLAED7 computes the updated eigensystem of a diagonal
 * matrix after modification by a rank-one symmetric matrix. This
 * routine is used only for the eigenproblem which requires all
 * eigenvalues and optionally eigenvectors of a dense or banded
 * Hermitian matrix that has been reduced to tridiagonal form.
 *
 *   T = Q(in) ( D(in) + RHO * Z*Z**H ) Q**H(in) = Q(out) * D(out) * Q**H(out)
 *
 *    where Z = Q**Hu, u is a vector of length N with ones in the
 *    CUTPNT-1 and CUTPNT th elements and zeros elsewhere (0-based).
 *
 *    The eigenvectors of the original matrix are stored in Q, and the
 *    eigenvalues are in D.  The algorithm consists of three stages:
 *
 *       The first stage consists of deflating the size of the problem
 *       when there are multiple eigenvalues or if there is a zero in
 *       the Z vector.  For each such occurrence the dimension of the
 *       secular equation problem is reduced by one.  This stage is
 *       performed by the routine DLAED2.
 *
 *       The second stage consists of calculating the updated
 *       eigenvalues. This is done by finding the roots of the secular
 *       equation via the routine DLAED4 (as called by SLAED3).
 *       This routine also calculates the eigenvectors of the current
 *       problem.
 *
 *       The final stage consists of computing the updated eigenvectors
 *       directly using the updated eigenvalues.  The eigenvectors for
 *       the current problem are multiplied with the eigenvectors from
 *       the overall problem.
 *
 * @param[in]     n       The dimension of the symmetric tridiagonal matrix.
 *                        N >= 0.
 * @param[in]     cutpnt  Contains the location of the last eigenvalue in the
 *                        leading sub-matrix. min(1,N) <= CUTPNT <= N.
 * @param[in]     qsiz    The dimension of the unitary matrix used to reduce
 *                        the full matrix to tridiagonal form. QSIZ >= N.
 * @param[in]     tlvls   The total number of merging levels in the overall
 *                        divide and conquer tree.
 * @param[in]     curlvl  The current level in the overall merge routine,
 *                        0 <= curlvl <= tlvls.
 * @param[in]     curpbm  The current problem in the current level in the overall
 *                        merge routine (counting from upper left to lower right).
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, the eigenvalues of the rank-1-perturbed matrix.
 *                        On exit, the eigenvalues of the repaired matrix.
 * @param[in,out] Q       Complex array, dimension (LDQ, N).
 *                        On entry, the eigenvectors of the rank-1-perturbed matrix.
 *                        On exit, the eigenvectors of the repaired tridiagonal matrix.
 * @param[in]     ldq     The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[in]     rho     Contains the subdiagonal element used to create the rank-1
 *                        modification.
 * @param[out]    indxq   Integer array, dimension (N).
 *                        This contains the permutation which will reintegrate the
 *                        subproblem just solved back into sorted order,
 *                        i.e., D( INDXQ( I ) ) will be in ascending order (0-based).
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
 * @param[out]    work    Complex array, dimension (QSIZ*N).
 * @param[out]    rwork   Double precision array, dimension (3*N+2*QSIZ*N).
 * @param[out]    iwork   Integer array, dimension (4*N).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = 1, an eigenvalue did not converge.
 */
void zlaed7(const int n, const int cutpnt, const int qsiz,
            const int tlvls, const int curlvl, const int curpbm,
            f64* D, c128* Q, const int ldq,
            const f64 rho, int* indxq, f64* qstore,
            int* qptr, int* prmptr, int* perm, int* givptr,
            int* givcol, f64* givnum, c128* work,
            f64* rwork, int* iwork, int* info)
{
    int coltyp, curr, i, idlmda, indx, indxc, indxp,
        iq, iw, iz, k, n1, n2, ptr;
    f64 rho_local;

    /* Test the input parameters. */
    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if ((1 < n ? 1 : n) > cutpnt || n < cutpnt) {
        *info = -2;
    } else if (qsiz < n) {
        *info = -3;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("ZLAED7", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /*
     * The following values are for bookkeeping purposes only.  They are
     * integer pointers which indicate the portion of the workspace
     * used by a particular array in DLAED2 and SLAED3.
     */
    iz = 0;
    idlmda = iz + n;
    iw = idlmda + n;
    iq = iw + n;

    indx = 0;
    indxc = indx + n;
    coltyp = indxc + n;
    indxp = coltyp + n;

    /*
     * Form the z-vector which consists of the last row of Q_1 and the
     * first row of Q_2.
     */
    ptr = 1 << tlvls;
    for (i = 0; i < curlvl - 1; i++) {
        ptr = ptr + (1 << (tlvls - 1 - i));
    }
    curr = ptr + curpbm;

    dlaeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr,
           givcol, givnum, qstore, qptr, &rwork[iz],
           &rwork[iz + n], info);

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
    zlaed8(&k, n, qsiz, Q, ldq, D, &rho_local, cutpnt, &rwork[iz],
           &rwork[idlmda], work, qsiz, &rwork[iw],
           &iwork[indxp], &iwork[indx], indxq,
           &perm[prmptr[curr]], &givptr[curr + 1],
           &givcol[2 * givptr[curr]],
           &givnum[2 * givptr[curr]], info);
    prmptr[curr + 1] = prmptr[curr] + n;
    givptr[curr + 1] = givptr[curr + 1] + givptr[curr];

    /*
     * Solve Secular Equation.
     */
    if (k != 0) {
        dlaed9(k, 0, k, n, D, &rwork[iq], k, rho_local,
               &rwork[idlmda], &rwork[iw],
               &qstore[qptr[curr]], k, info);
        zlacrm(qsiz, k, work, qsiz, &qstore[qptr[curr]], k,
               Q, ldq, &rwork[iq]);
        qptr[curr + 1] = qptr[curr] + k * k;
        if (*info != 0) {
            return;
        }

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
