/**
 * @file cgbsvx.c
 * @brief CGBSVX computes the solution to a complex system of linear equations
 *        A * X = B for banded matrices, using the LU factorization with
 *        equilibration and iterative refinement.
 */

#include <math.h>
#include <complex.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGBSVX uses the LU factorization to compute the solution to a complex
 * system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B,  A**T * X = B,  or  A**H * X = B
 * @endrst
 * where A is a band matrix of order `n` with `kl` subdiagonals and `ku`
 * superdiagonals, and X and `B` are `n`-by-`nrhs` matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @rst
 * The following steps are performed by this subroutine:
 *
 * 1. If ``fact='E'``, real scaling factors are computed to equilibrate
 *    the system::
 *
 *        trans = 'N':  diag(R)*A*diag(C)     *inv(diag(C))*X = diag(R)*B
 *        trans = 'T': (diag(R)*A*diag(C))**T *inv(diag(R))*X = diag(C)*B
 *        trans = 'C': (diag(R)*A*diag(C))**H *inv(diag(R))*X = diag(C)*B
 *
 *    Whether or not the system will be equilibrated depends on the
 *    scaling of the matrix A, but if equilibration is used, A is
 *    overwritten by diag(R)*A*diag(C) and B by diag(R)*B (if ``trans='N'``)
 *    or diag(C)*B (if ``trans='T'`` or ``'C'``).
 *
 * 2. If ``fact='N'`` or ``'E'``, the LU decomposition is used to factor
 *    the matrix A (after equilibration if ``fact='E'``) as::
 *
 *        A = L * U
 *
 *    where L is a product of permutation and unit lower triangular
 *    matrices with kl subdiagonals, and U is upper triangular with
 *    kl+ku superdiagonals.
 *
 * 3. If some ``U(i,i)=0``, so that U is exactly singular, then the routine
 *    returns with ``info=i``. Otherwise, the factored form of A is used
 *    to estimate the condition number of the matrix A. If the reciprocal
 *    of the condition number is less than machine precision, ``info=n+1``
 *    is returned as a warning, but the routine still goes on to solve for
 *    X and compute error bounds as described below.
 *
 * 4. The system of equations is solved for X using the factored form of A.
 *
 * 5. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates for it.
 *
 * 6. If equilibration was used, the matrix X is premultiplied by
 *    ``diag(C)`` (if ``trans='N'``) or ``diag(R)`` (if ``trans='T'`` or
 *    ``'C'``) so that it solves the original system before equilibration.
 * @endrst
 *
 * @param[in]     fact
 *                        - `'F'`: `AFB` and `ipiv` contain the factored form of A.
 *                        - `'N'`: The matrix A will be copied to `AFB` and factored.
 *                        - `'E'`: The matrix A will be equilibrated if necessary,
 *                          then copied to `AFB` and factored.
 * @param[in]     trans
 *                        - `'N'`: A * X = B (No transpose)
 *                        - `'T'`: A**T * X = B (Transpose)
 *                        - `'C'`: A**H * X = B (Conjugate transpose)
 * @param[in]     n       The number of linear equations (order of A). `n>=0`.
 * @param[in]     kl      The number of subdiagonals within the band of A. `kl>=0`.
 * @param[in]     ku      The number of superdiagonals within the band of A. `ku>=0`.
 * @param[in]     nrhs    The number of right hand sides. `nrhs>=0`.
 * @param[in,out] AB      Array of dimension (`ldab`, `n`).
 *                        On entry, the matrix A in band storage, in rows 0 to
 *                        `kl+ku`. The j-th column of A is stored in the j-th column
 *                        of the array `AB` as follows:
 *                        `AB[ku+i-j + j*ldab] = A(i,j)` for `max(0,j-ku)<=i<=min(n-1,j+kl)`.
 *                        On exit, if `equed != 'N'`, A is scaled.
 * @param[in]     ldab    The leading dimension of `AB`. `ldab>=kl+ku+1`.
 * @param[in,out] AFB     Array of dimension (`ldafb`, `n`).
 *                        On entry (if `fact='F'`), contains the LU factors.
 *                        On exit, contains the factors L and U.
 * @param[in]     ldafb   The leading dimension of `AFB`. `ldafb>=2*kl+ku+1`.
 * @param[in,out] ipiv    Array of dimension (`n`).
 *                        Pivot indices from factorization.
 * @param[in,out] equed   On entry (if `fact='F'`), specifies equilibration done.
 *                        On exit, specifies the form of equilibration:
 *                        - `'N'`: No equilibration
 *                        - `'R'`: Row equilibration (A := diag(R) * A)
 *                        - `'C'`: Column equilibration (A := A * diag(C))
 *                        - `'B'`: Both (A := diag(R) * A * diag(C))
 * @param[in,out] R       Array of dimension (`n`).
 *                        Row scale factors.
 * @param[in,out] C       Array of dimension (`n`).
 *                        Column scale factors.
 * @param[in,out] B       Array of dimension (`ldb`, `nrhs`).
 *                        On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                        On exit, if equilibration was done, `B` is scaled.
 * @param[in]     ldb     The leading dimension of `B`. `ldb>=max(1,n)`.
 * @param[out]    X       Array of dimension (`ldx`, `nrhs`).
 *                        The `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldx     The leading dimension of `X`. `ldx>=max(1,n)`.
 * @param[out]    rcond   Reciprocal condition number estimate.
 * @param[out]    ferr    Array of dimension (`nrhs`).
 *                        Forward error bound for each solution vector.
 * @param[out]    berr    Array of dimension (`nrhs`).
 *                        Backward error for each solution vector.
 * @param[out]    work    Complex workspace array of dimension (`2*n`).
 * @param[out]    rwork   Real workspace array of dimension (`max(1,n)`).
 *                        On exit, `rwork[0]` contains the reciprocal pivot growth
 *                        factor.
 * @param[out]    info
 *                          - `info=0`: successful exit
 *                          - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                            value
 *                          - `info>0`: if `info=i`, U(i,i) is exactly zero (1-based).
 *                            If `info=n+1`, U is nonsingular but `rcond` < machine
 *                            precision.
 */
void cgbsvx(
    const char* fact,
    const char* trans,
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    c64* restrict AB,
    const INT ldab,
    c64* restrict AFB,
    const INT ldafb,
    INT* restrict ipiv,
    char* equed,
    f32* restrict R,
    f32* restrict C,
    c64* restrict B,
    const INT ldb,
    c64* restrict X,
    const INT ldx,
    f32* rcond,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j, j1, j2, infequ;
    f32 amax, anorm, bignum, colcnd, rcmax, rcmin, rowcnd, rpvgrw, smlnum;
    char norm;
    INT nofact, equil, notran, rowequ, colequ;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (nofact || equil) {
        *equed = 'N';
        rowequ = 0;
        colequ = 0;
    } else {
        rowequ = (*equed == 'R' || *equed == 'r' || *equed == 'B' || *equed == 'b');
        colequ = (*equed == 'C' || *equed == 'c' || *equed == 'B' || *equed == 'b');
        smlnum = FLT_MIN;
        bignum = ONE / smlnum;
    }

    /* Test the input parameters */
    if (!nofact && !equil && fact[0] != 'F' && fact[0] != 'f') {
        *info = -1;
    } else if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (kl < 0) {
        *info = -4;
    } else if (ku < 0) {
        *info = -5;
    } else if (nrhs < 0) {
        *info = -6;
    } else if (ldab < kl + ku + 1) {
        *info = -8;
    } else if (ldafb < 2 * kl + ku + 1) {
        *info = -10;
    } else if (fact[0] == 'F' || fact[0] == 'f') {
        if (!rowequ && !colequ && *equed != 'N' && *equed != 'n') {
            *info = -12;
        }
    }

    if (*info == 0) {
        if (rowequ) {
            rcmin = bignum;
            rcmax = ZERO;
            for (j = 0; j < n; j++) {
                if (R[j] < rcmin) rcmin = R[j];
                if (R[j] > rcmax) rcmax = R[j];
            }
            if (rcmin <= ZERO) {
                *info = -13;
            } else if (n > 0) {
                f32 rmin = (rcmin > smlnum) ? rcmin : smlnum;
                f32 rmax = (rcmax < bignum) ? rcmax : bignum;
                rowcnd = rmin / rmax;
            } else {
                rowcnd = ONE;
            }
        }
        if (colequ && *info == 0) {
            rcmin = bignum;
            rcmax = ZERO;
            for (j = 0; j < n; j++) {
                if (C[j] < rcmin) rcmin = C[j];
                if (C[j] > rcmax) rcmax = C[j];
            }
            if (rcmin <= ZERO) {
                *info = -14;
            } else if (n > 0) {
                f32 cmin = (rcmin > smlnum) ? rcmin : smlnum;
                f32 cmax = (rcmax < bignum) ? rcmax : bignum;
                colcnd = cmin / cmax;
            } else {
                colcnd = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (n > 1 ? n : 1)) {
                *info = -16;
            } else if (ldx < (n > 1 ? n : 1)) {
                *info = -18;
            }
        }
    }

    if (*info != 0) {
        xerbla("CGBSVX", -(*info));
        return;
    }

    if (equil) {
        /* Compute row and column scalings to equilibrate the matrix A */
        cgbequ(n, n, kl, ku, AB, ldab, R, C, &rowcnd, &colcnd, &amax, &infequ);
        if (infequ == 0) {
            /* Equilibrate the matrix */
            claqgb(n, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax, equed);
            rowequ = (*equed == 'R' || *equed == 'r' || *equed == 'B' || *equed == 'b');
            colequ = (*equed == 'C' || *equed == 'c' || *equed == 'B' || *equed == 'b');
        }
    }

    /* Scale the right hand side */
    if (notran) {
        if (rowequ) {
            for (j = 0; j < nrhs; j++) {
                for (i = 0; i < n; i++) {
                    B[i + j * ldb] = R[i] * B[i + j * ldb];
                }
            }
        }
    } else if (colequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = C[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        /* Compute the LU factorization of the band matrix A.
         * Copy AB to AFB, converting from kl+ku+1 rows to 2*kl+ku+1 rows storage */
        for (j = 0; j < n; j++) {
            j1 = (j - ku > 0) ? j - ku : 0;
            j2 = (j + kl < n - 1) ? j + kl : n - 1;
            INT len = j2 - j1 + 1;
            /* Source: AB at row ku + j1 - j, column j
             * Dest: AFB at row kl + ku + j1 - j, column j */
            cblas_ccopy(len, &AB[ku + j1 - j + j * ldab], 1,
                        &AFB[kl + ku + j1 - j + j * ldafb], 1);
        }

        cgbtrf(n, n, kl, ku, AFB, ldafb, ipiv, info);

        /* Return if INFO is non-zero */
        if (*info > 0) {
            /* Compute the reciprocal pivot growth factor of the
             * leading rank-deficient INFO columns of A */
            anorm = ZERO;
            for (j = 0; j < *info; j++) {
                INT i_start = (ku - j > 0) ? ku - j : 0;
                INT i_end = (n + ku - j - 1 < kl + ku) ? n + ku - j - 1 : kl + ku;
                for (i = i_start; i <= i_end; i++) {
                    f32 temp = cabsf(AB[i + j * ldab]);
                    if (anorm < temp) anorm = temp;
                }
            }
            INT k_val = (*info - 1 < kl + ku) ? *info - 1 : kl + ku;
            INT start_offset = (kl + ku + 1 - *info > 0) ? kl + ku + 1 - *info : 0;
            rpvgrw = clantb("M", "U", "N", *info, k_val, &AFB[start_offset], ldafb, rwork);
            if (rpvgrw == ZERO) {
                rpvgrw = ONE;
            } else {
                rpvgrw = anorm / rpvgrw;
            }
            rwork[0] = rpvgrw;
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A and the reciprocal pivot growth factor RPVGRW */
    if (notran) {
        norm = '1';
    } else {
        norm = 'I';
    }
    anorm = clangb(&norm, n, kl, ku, AB, ldab, rwork);
    rpvgrw = clantb("M", "U", "N", n, kl + ku, AFB, ldafb, rwork);
    if (rpvgrw == ZERO) {
        rpvgrw = ONE;
    } else {
        rpvgrw = clangb("M", n, kl, ku, AB, ldab, rwork) / rpvgrw;
    }

    /* Compute the reciprocal of the condition number of A */
    cgbcon(&norm, n, kl, ku, AFB, ldafb, ipiv, anorm, rcond, work, rwork, info);

    /* Compute the solution matrix X */
    clacpy("F", n, nrhs, B, ldb, X, ldx);
    cgbtrs(trans, n, kl, ku, nrhs, AFB, ldafb, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solution and
     * compute error bounds and backward error estimates for it */
    cgbrfs(trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv,
           B, ldb, X, ldx, ferr, berr, work, rwork, info);

    /* Transform the solution matrix X to a solution of the original system */
    if (notran) {
        if (colequ) {
            for (j = 0; j < nrhs; j++) {
                for (i = 0; i < n; i++) {
                    X[i + j * ldx] = C[i] * X[i + j * ldx];
                }
            }
            for (j = 0; j < nrhs; j++) {
                ferr[j] = ferr[j] / colcnd;
            }
        }
    } else if (rowequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                X[i + j * ldx] = R[i] * X[i + j * ldx];
            }
        }
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / rowcnd;
        }
    }

    /* Set INFO = N+1 if the matrix is singular to working precision */
    if (*rcond < FLT_EPSILON) {
        *info = n + 1;
    }

    rwork[0] = rpvgrw;
}
