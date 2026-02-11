/**
 * @file slaed1.c
 * @brief SLAED1 computes the updated eigensystem of a diagonal matrix after
 *        modification by a rank-one symmetric matrix. Used when the original
 *        matrix is tridiagonal.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"


/**
 * SLAED1 computes the updated eigensystem of a diagonal
 * matrix after modification by a rank-one symmetric matrix.  This
 * routine is used only for the eigenproblem which requires all
 * eigenvalues and eigenvectors of a tridiagonal matrix.  SLAED7 handles
 * the case in which eigenvalues only or eigenvalues and eigenvectors
 * of a full symmetric matrix (which was reduced to tridiagonal form)
 * are desired.
 *
 *   T = Q(in) ( D(in) + RHO * Z*Z**T ) Q**T(in) = Q(out) * D(out) * Q**T(out)
 *
 *    where Z = Q**T*u, u is a vector of length N with ones in the
 *    CUTPNT-1 and CUTPNT th elements and zeros elsewhere (0-based).
 *
 *    The eigenvectors of the original matrix are stored in Q, and the
 *    eigenvalues are in D.  The algorithm consists of three stages:
 *
 *       The first stage consists of deflating the size of the problem
 *       when there are multiple eigenvalues or if there is a zero in
 *       the Z vector.  For each such occurrence the dimension of the
 *       secular equation problem is reduced by one.  This stage is
 *       performed by the routine SLAED2.
 *
 *       The second stage consists of calculating the updated
 *       eigenvalues. This is done by finding the roots of the secular
 *       equation via the routine SLAED4 (as called by SLAED3).
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
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, the eigenvalues of the rank-1-perturbed matrix.
 *                        On exit, the eigenvalues of the repaired matrix.
 * @param[in,out] Q       Double precision array, dimension (LDQ, N).
 *                        On entry, the eigenvectors of the rank-1-perturbed matrix.
 *                        On exit, the eigenvectors of the repaired tridiagonal matrix.
 * @param[in]     ldq     The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[in,out] indxq   Integer array, dimension (N).
 *                        On entry, the permutation which separately sorts the two
 *                        subproblems in D into ascending order.
 *                        On exit, the permutation which will reintegrate the
 *                        subproblems back into sorted order,
 *                        i.e. D( INDXQ( I ) ) will be in ascending order (0-based).
 * @param[in]     rho     The subdiagonal entry used to create the rank-1 modification.
 * @param[in]     cutpnt  The location of the last eigenvalue in the leading
 *                        sub-matrix. min(1,N) <= CUTPNT <= N/2.
 * @param[out]    work    Double precision array, dimension (4*N + N**2).
 * @param[out]    iwork   Integer array, dimension (4*N).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = 1, an eigenvalue did not converge.
 */
void slaed1(const int n, float* D, float* Q, const int ldq,
            int* indxq, const float rho, const int cutpnt,
            float* work, int* iwork, int* info)
{
    int coltyp, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, n1, n2, zpp1;
    float rho_local;

    /* Test the input parameters. */
    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -4;
    } else if ((1 < n / 2 ? 1 : n / 2) > cutpnt || (n / 2) < cutpnt) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("SLAED1", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /*
     * The following values are integer pointers which indicate
     * the portion of the workspace
     * used by a particular array in SLAED2 and SLAED3.
     */
    iz = 0;
    idlmda = iz + n;
    iw = idlmda + n;
    iq2 = iw + n;

    indx = 0;
    indxc = indx + n;
    coltyp = indxc + n;
    indxp = coltyp + n;

    /*
     * Form the z-vector which consists of the last row of Q_1 and the
     * first row of Q_2.
     */

    /* In Fortran: DCOPY(CUTPNT, Q(CUTPNT, 1), LDQ, WORK(IZ), 1)
     * Q(CUTPNT, 1) in 1-based Fortran is row CUTPNT, col 1.
     * In 0-based C: row (cutpnt - 1), col 0 => Q[(cutpnt - 1) + 0 * ldq]
     * Stride is ldq (stepping across columns in column-major).
     */
    cblas_scopy(cutpnt, &Q[(cutpnt - 1)], ldq, &work[iz], 1);

    /* In Fortran: ZPP1 = CUTPNT + 1
     * DCOPY(N-CUTPNT, Q(ZPP1, ZPP1), LDQ, WORK(IZ+CUTPNT), 1)
     * Q(ZPP1, ZPP1) in 1-based is row CUTPNT+1, col CUTPNT+1.
     * In 0-based C: row cutpnt, col cutpnt => Q[cutpnt + cutpnt * ldq]
     */
    zpp1 = cutpnt;  /* 0-based row index for what was CUTPNT+1 in Fortran */
    cblas_scopy(n - cutpnt, &Q[zpp1 + zpp1 * ldq], ldq, &work[iz + cutpnt], 1);

    /* Deflate eigenvalues. */
    n1 = cutpnt;
    rho_local = rho;
    slaed2(&k, n, n1, D, Q, ldq, indxq, &rho_local, &work[iz],
           &work[idlmda], &work[iw], &work[iq2],
           &iwork[indx], &iwork[indxc], &iwork[indxp],
           &iwork[coltyp], info);

    if (*info != 0)
        return;

    /* Solve Secular Equation. */
    if (k != 0) {
        is = (iwork[coltyp] + iwork[coltyp + 1]) * cutpnt +
             (iwork[coltyp + 1] + iwork[coltyp + 2]) * (n - cutpnt) + iq2;
        slaed3(k, n, n1, D, Q, ldq, rho_local, &work[idlmda],
               &work[iq2], &iwork[indxc], &iwork[coltyp],
               &work[iw], &work[is], info);
        if (*info != 0)
            return;

        /* Prepare the INDXQ sorting permutation. */
        n1 = k;
        n2 = n - k;
        slamrg(n1, n2, D, 1, -1, indxq);
    } else {
        for (i = 0; i < n; i++) {
            indxq[i] = i;
        }
    }
}
