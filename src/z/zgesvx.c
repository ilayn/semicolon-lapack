/**
 * @file zgesvx.c
 * @brief ZGESVX computes the solution to a complex system of linear equations
 *        A * X = B, using the LU factorization with equilibration and
 *        iterative refinement.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGESVX uses the LU factorization to compute the solution to a complex
 * system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where `A` is an `n`-by-`n` matrix and X and `B` are `n`-by-`nrhs` matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='E'``, real scaling factors are computed to equilibrate
 *    the system::
 *
 *        trans = 'N':  diag(R)*A*diag(C)     *inv(diag(C))*X = diag(R)*B
 *        trans = 'T': (diag(R)*A*diag(C))**T *inv(diag(R))*X = diag(C)*B
 *        trans = 'C': (diag(R)*A*diag(C))**H *inv(diag(R))*X = diag(C)*B
 *
 * 2. If ``fact='N'`` or ``'E'``, the LU decomposition is used to factor the
 *    matrix ``A`` (after equilibration if ``fact='E'``) as::
 *
 *        A = P * L * U
 *
 * 3. If some ``U(i,i)=0``, so that U is exactly singular, then the routine
 *    returns with ``info=i``. Otherwise, the factored form of ``A`` is used
 *    to estimate the condition number of the matrix ``A``.
 *
 * 4. The system of equations is solved for X using the factored form of ``A``.
 *
 * 5. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates for it.
 *
 * 6. If equilibration was used, the matrix X is premultiplied by
 *    ``diag(C)`` (if ``trans='N'``) or ``diag(R)`` (if ``trans='T'`` or ``'C'``).
 * @endrst
 *
 * @param[in]     fact    - `'F'`: `AF` and `ipiv` contain the factored form of `A`.
 *                        - `'N'`: The matrix `A` will be copied to `AF` and factored.
 *                        - `'E'`: The matrix `A` will be equilibrated if necessary,
 *                          then copied to `AF` and factored.
 * @param[in]     trans   - `'N'`: A * X = B (No transpose)
 *                        - `'T'`: A**T * X = B (Transpose)
 *                        - `'C'`: A**H * X = B (Conjugate transpose)
 * @param[in]     n       The number of linear equations (order of `A`). `n>=0`.
 * @param[in]     nrhs    The number of right hand sides. `nrhs>=0`.
 * @param[in,out] A       Complex array of dimension (`lda`, `n`).
 *                        On entry, the `n`-by-`n` matrix `A`.
 *                        On exit, if equilibration was done, `A` is scaled.
 * @param[in]     lda     The leading dimension of `A`. `lda>=max(1,n)`.
 * @param[in,out] AF      Complex array of dimension (`ldaf`, `n`).
 *                        On entry (if `fact='F'`), contains the LU factors.
 *                        On exit, contains the factors L and U.
 * @param[in]     ldaf    The leading dimension of `AF`. `ldaf>=max(1,n)`.
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
 * @param[in,out] B       Complex array of dimension (`ldb`, `nrhs`).
 *                        On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                        On exit, if equilibration was done, `B` is scaled.
 * @param[in]     ldb     The leading dimension of `B`. `ldb>=max(1,n)`.
 * @param[out]    X       Complex array of dimension (`ldx`, `nrhs`).
 *                        The `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldx     The leading dimension of `X`. `ldx>=max(1,n)`.
 * @param[out]    rcond   Reciprocal condition number estimate.
 * @param[out]    ferr    Array of dimension (`nrhs`).
 *                        Forward error bound for each solution vector.
 * @param[out]    berr    Array of dimension (`nrhs`).
 *                        Backward error for each solution vector.
 * @param[out]    work    Complex workspace array of dimension (`2*n`).
 * @param[out]    rwork   Real workspace array of dimension (`max(1,2*n)`).
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
void zgesvx(
    const char* fact,
    const char* trans,
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    c128* restrict AF,
    const INT ldaf,
    INT* restrict ipiv,
    char* equed,
    f64* restrict R,
    f64* restrict C,
    c128* restrict B,
    const INT ldb,
    c128* restrict X,
    const INT ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, infequ;
    f64 amax, anorm, bignum, colcnd, rcmax, rcmin, rowcnd, rpvgrw, smlnum;
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
        smlnum = DBL_MIN;
        bignum = ONE / smlnum;
    }

    // Test the input parameters
    if (!nofact && !equil && fact[0] != 'F' && fact[0] != 'f') {
        *info = -1;
    } else if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (fact[0] == 'F' || fact[0] == 'f') {
        if (!rowequ && !colequ && *equed != 'N' && *equed != 'n') {
            *info = -10;
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
                *info = -11;
            } else if (n > 0) {
                f64 rmin = (rcmin > smlnum) ? rcmin : smlnum;
                f64 rmax = (rcmax < bignum) ? rcmax : bignum;
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
                *info = -12;
            } else if (n > 0) {
                f64 cmin = (rcmin > smlnum) ? rcmin : smlnum;
                f64 cmax = (rcmax < bignum) ? rcmax : bignum;
                colcnd = cmin / cmax;
            } else {
                colcnd = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (n > 1 ? n : 1)) {
                *info = -14;
            } else if (ldx < (n > 1 ? n : 1)) {
                *info = -16;
            }
        }
    }

    if (*info != 0) {
        xerbla("ZGESVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A
        zgeequ(n, n, A, lda, R, C, &rowcnd, &colcnd, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix
            zlaqge(n, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
            rowequ = (*equed == 'R' || *equed == 'r' || *equed == 'B' || *equed == 'b');
            colequ = (*equed == 'C' || *equed == 'c' || *equed == 'B' || *equed == 'b');
        }
    }

    // Scale the right hand side
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
        // Compute the LU factorization of A
        zlacpy("F", n, n, A, lda, AF, ldaf);
        zgetrf(n, n, AF, ldaf, ipiv, info);

        // Return if INFO is non-zero
        if (*info > 0) {
            // Compute the reciprocal pivot growth factor of the
            // leading rank-deficient INFO columns of A
            rpvgrw = zlantr("M", "U", "N", *info, *info, AF, ldaf, rwork);
            if (rpvgrw == ZERO) {
                rpvgrw = ONE;
            } else {
                rpvgrw = zlange("M", n, *info, A, lda, rwork) / rpvgrw;
            }
            rwork[0] = rpvgrw;
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A and the reciprocal pivot growth factor RPVGRW
    if (notran) {
        norm = '1';
    } else {
        norm = 'I';
    }
    anorm = zlange(&norm, n, n, A, lda, rwork);
    rpvgrw = zlantr("M", "U", "N", n, n, AF, ldaf, rwork);
    if (rpvgrw == ZERO) {
        rpvgrw = ONE;
    } else {
        rpvgrw = zlange("M", n, n, A, lda, rwork) / rpvgrw;
    }

    // Compute the reciprocal of the condition number of A
    zgecon(&norm, n, AF, ldaf, anorm, rcond, work, rwork, info);

    // Compute the solution matrix X
    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zgetrs(trans, n, nrhs, AF, ldaf, ipiv, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it
    zgerfs(trans, n, nrhs, A, lda, AF, ldaf, ipiv, B, ldb, X, ldx,
           ferr, berr, work, rwork, info);

    // Transform the solution matrix X to a solution of the original system
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

    // Set INFO = N+1 if the matrix is singular to working precision
    if (*rcond < DBL_EPSILON) {
        *info = n + 1;
    }

    rwork[0] = rpvgrw;
}
