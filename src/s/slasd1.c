/**
 * @file slasd1.c
 * @brief SLASD1 computes the SVD of an upper bidiagonal matrix B of the
 *        specified size. Used by sbdsdc.
 */

#include "semicolon_lapack_single.h"
#include <math.h>

/**
 * SLASD1 computes the SVD of an upper bidiagonal n-by-m matrix B,
 * where n = nl + nr + 1 and m = n + sqre. SLASD1 is called from SLASD0.
 *
 * A related subroutine SLASD7 handles the case in which the singular
 * values (and the singular vectors in factored form) are desired.
 *
 * SLASD1 computes the SVD as follows:
 * @rst
 * .. code-block:: text
 *
 *                   ( D1(in)    0    0       0 )
 *       B = U(in) * (   Z1**T   a   Z2**T    b ) * VT(in)
 *                   (   0       0   D2(in)   0 )
 *
 *         = U(out) * ( D(out) 0) * VT(out)
 * @endrst
 *
 * where Z**T = (Z1**T a Z2**T b) = u**T VT**T, and u is a vector of dimension m
 * with alpha and beta in u[nl] and u[nl+1] and zeros elsewhere; and the
 * entry b is empty if sqre = 0.
 *
 * The left singular vectors of the original matrix are stored in U, and
 * the transpose of the right singular vectors are stored in VT, and the
 * singular values are in D.  The algorithm consists of three stages:
 *
 * The first stage consists of deflating the size of the problem
 * when there are multiple singular values or when there are zeros in
 * the Z vector.  For each such occurrence the dimension of the
 * secular equation problem is reduced by one.  This stage is
 * performed by the routine SLASD2.
 *
 * The second stage consists of calculating the updated
 * singular values. This is done by finding the square roots of the
 * roots of the secular equation via the routine SLASD4 (as called
 * by SLASD3). This routine also calculates the singular vectors of
 * the current problem.
 *
 * The final stage consists of computing the updated singular vectors
 * directly using the updated singular values.  The singular vectors
 * for the current problem are multiplied with the singular vectors
 * from the overall problem.
 *
 * @param[in]     nl      The row dimension of the upper block.  nl >= 1.
 * @param[in]     nr      The row dimension of the lower block.  nr >= 1.
 * @param[in]     sqre    = 0: the lower block is an nr-by-nr square matrix.
 *                        = 1: the lower block is an nr-by-(nr+1) rectangular matrix.
 *                        The bidiagonal matrix has row dimension n = nl + nr + 1,
 *                        and column dimension m = n + sqre.
 * @param[in,out] D       Double precision array,
 *                        dimension (n = nl+nr+1).
 *                        On entry D[0:nl-1, 0:nl-1] contains the singular values
 *                        of the upper block; and D[nl+1:n-1] contains the singular
 *                        values of the lower block. On exit D[0:n-1] contains the
 *                        singular values of the modified matrix.
 * @param[in,out] alpha   Contains the diagonal element associated with the added row.
 * @param[in,out] beta    Contains the off-diagonal element associated with the added
 *                        row.
 * @param[in,out] U       Double precision array, dimension (ldu, n).
 *                        On entry U[0:nl-1, 0:nl-1] contains the left singular
 *                        vectors of the upper block; U[nl+1:n-1, nl+1:n-1] contains
 *                        the left singular vectors of the lower block. On exit U
 *                        contains the left singular vectors of the bidiagonal
 *                        matrix.
 * @param[in]     ldu     The leading dimension of the array U.  ldu >= max( 1, n ).
 * @param[in,out] VT      Double precision array, dimension (ldvt, m)
 *                        where m = n + sqre.
 *                        On entry VT[0:nl, 0:nl]**T contains the right singular
 *                        vectors of the upper block; VT[nl+1:m-1, nl+1:m-1]**T
 *                        contains the right singular vectors of the lower block.
 *                        On exit VT**T contains the right singular vectors of the
 *                        bidiagonal matrix.
 * @param[in]     ldvt    The leading dimension of the array VT.  ldvt >= max( 1, m ).
 * @param[in,out] IDXQ    Integer array, dimension (n).
 *                        This contains the permutation which will reintegrate the
 *                        subproblem just solved back into sorted order, i.e.
 *                        D[ IDXQ[ i = 0, n-1 ] ] will be in ascending order.
 * @param[out]    IWORK   Integer array, dimension (4*n).
 * @param[out]    work    Double precision array, dimension (3*m**2 + 2*m).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = 1, a singular value did not converge
 *
 * @par Contributors:
 * @rst
 * Ming Gu and Huan Ren, Computer Science Division, University of
 * California at Berkeley, USA
 * @endrst
 */
void slasd1(const INT nl, const INT nr, const INT sqre,
            f32* restrict D, f32* alpha, f32* beta,
            f32* restrict U, const INT ldu,
            f32* restrict VT, const INT ldvt,
            INT* restrict IDXQ, INT* restrict IWORK,
            f32* restrict work, INT* info)
{
    INT coltyp, i, idx, idxc, idxp, iq, isigma, iu2, ivt2, iz;
    INT k, ldq, ldu2, ldvt2, m, n, n1, n2;
    f32 orgnrm;

    /* Test the input parameters. */
    *info = 0;

    if (nl < 1) {
        *info = -1;
    } else if (nr < 1) {
        *info = -2;
    } else if (sqre < 0 || sqre > 1) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("SLASD1", -(*info));
        return;
    }

    n = nl + nr + 1;
    m = n + sqre;

    /* The following values are for bookkeeping purposes only.  They are
     * integer pointers which indicate the portion of the workspace
     * used by a particular array in SLASD2 and SLASD3. */
    ldu2 = n;
    ldvt2 = m;

    iz = 0;
    isigma = iz + m;
    iu2 = isigma + n;
    ivt2 = iu2 + ldu2 * n;
    iq = ivt2 + ldvt2 * m;

    idx = 0;
    idxc = idx + n;
    coltyp = idxc + n;
    idxp = coltyp + n;

    /* Scale. */
    orgnrm = fabsf(*alpha) > fabsf(*beta) ? fabsf(*alpha) : fabsf(*beta);
    D[nl] = 0.0f;
    for (i = 0; i < n; i++) {
        if (fabsf(D[i]) > orgnrm) {
            orgnrm = fabsf(D[i]);
        }
    }
    slascl("G", 0, 0, orgnrm, 1.0f, n, 1, D, n, info);
    *alpha = *alpha / orgnrm;
    *beta = *beta / orgnrm;

    /* Deflate singular values. */
    slasd2(nl, nr, sqre, &k, D, &work[iz], *alpha, *beta, U, ldu,
           VT, ldvt, &work[isigma], &work[iu2], ldu2,
           &work[ivt2], ldvt2, &IWORK[idxp], &IWORK[idx],
           &IWORK[idxc], IDXQ, &IWORK[coltyp], info);

    /* Solve Secular Equation and update singular vectors. */
    ldq = k;
    slasd3(nl, nr, sqre, k, D, &work[iq], ldq, &work[isigma],
           U, ldu, &work[iu2], ldu2, VT, ldvt, &work[ivt2],
           ldvt2, &IWORK[idxc], &IWORK[coltyp], &work[iz], info);

    /* Report the convergence failure. */
    if (*info != 0) {
        return;
    }

    /* Unscale. */
    slascl("G", 0, 0, 1.0f, orgnrm, n, 1, D, n, info);

    /* Prepare the IDXQ sorting permutation. */
    n1 = k;
    n2 = n - k;
    slamrg(n1, n2, D, 1, -1, IDXQ);
}
