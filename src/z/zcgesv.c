/**
 * @file zcgesv.c
 * @brief Mixed precision iterative refinement solver for complex general linear systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"
#include "semicolon_lapack_complex_single.h"

/**
 * ZCGESV computes the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where `A` is an `n`-by-`n` matrix and `X` and `B` are `n`-by-`nrhs` matrices.
 *
 * ZCGESV first attempts to factorize the matrix in COMPLEX and use this
 * factorization within an iterative refinement procedure to produce a
 * solution with COMPLEX*16 normwise backward error quality (see below).
 * If the approach fails the method switches to a COMPLEX*16
 * factorization and solve.
 *
 * @rst
 * The iterative refinement process is stopped if::
 *
 *     ITER > ITERMAX
 *
 * or for all the RHS we have::
 *
 *     RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
 *
 * where
 *
 * - ITER is the number of the current iteration in the iterative
 *   refinement process
 * - RNRM is the infinity-norm of the residual
 * - XNRM is the infinity-norm of the solution
 * - ANRM is the infinity-operator-norm of the matrix ``A``
 * - EPS is the machine epsilon returned by DLAMCH('Epsilon')
 *
 * The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00
 * respectively.
 * @endrst
 *
 * @param[in]     n     The number of linear equations, i.e., the order of `A`.
 *                      `n>=0`.
 * @param[in]     nrhs  The number of right hand sides. `nrhs>=0`.
 * @param[in,out] A     Array of dimension (`lda`, `n`).
 *                      On entry, the `n`-by-`n` coefficient matrix `A`.
 *                      On exit, if iterative refinement succeeded (`iter>=0`), `A`
 *                      is unchanged. If COMPLEX*16 factorization was used
 *                      (`iter<0`), `A` contains the factors L and U from A = P*L*U.
 * @param[in]     lda   The leading dimension of `A`. `lda>=max(1,n)`.
 * @param[out]    ipiv  Array of dimension `n`.
 *                      The pivot indices.
 * @param[in]     B     Array of dimension (`ldb`, `nrhs`).
 *                      The `n`-by-`nrhs` right hand side matrix `B`.
 * @param[in]     ldb   The leading dimension of `B`. `ldb>=max(1,n)`.
 * @param[out]    X     Array of dimension (`ldx`, `nrhs`).
 *                      If `info=0`, the `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldx   The leading dimension of `X`. `ldx>=max(1,n)`.
 * @param[out]    work  Complex*16 workspace for residual vectors.
 *                      Array of dimension (`n`, `nrhs`).
 * @param[out]    swork Complex (single precision) workspace for matrix and solutions.
 *                      Array of dimension `n*(n+nrhs)`.
 * @param[out]    rwork Double precision array, dimension (`n`).
 * @param[out]    iter  Iteration count:
 *                      - `iter<0`: iterative refinement has failed, COMPLEX*16
 *                        factorization has been performed
 *                        - -1 : the routine fell back to full precision for
 *                          implementation- or machine-specific reasons
 *                        - -2 : narrowing the precision induced an overflow,
 *                          the routine fell back to full precision
 *                        - -3 : failure of CGETRF
 *                        - -31: stop the iterative refinement after the 30th
 *                          iterations
 *                      - `iter>0`: iterative refinement has been successfully used.
 *                        Returns the number of iterations
 * @param[out]    info
 *                          - `info=0`: successful exit
 *                          - `info<0`: if `info=-i`, argument i had an illegal value
 *                          - `info>0`: if `info=i`, U(i,i) is exactly zero
 */
void zcgesv(
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    INT* restrict ipiv,
    const c128* restrict B,
    const INT ldb,
    c128* restrict X,
    const INT ldx,
    c128* restrict work,
    c64* restrict swork,
    f64* restrict rwork,
    INT* iter,
    INT* info)
{
    const INT ITERMAX = 30;
    const f64 BWDMAX = 1.0;
    const c128 NEGONE = CMPLX(-1.0, + 0.0);
    const c128 ONE = CMPLX(1.0, + 0.0);

    INT i, iiter, iinfo;
    f64 anrm, cte, eps, rnrm, xnrm;
    INT converged;

    c64* SA;
    c64* SX;

    *info = 0;
    *iter = 0;

    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("ZCGESV", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    SA = swork;
    SX = &swork[n * n];

    anrm = zlange("I", n, n, A, lda, rwork);
    eps = dlamch("E");
    cte = anrm * eps * sqrt((f64)n) * BWDMAX;

    zlag2c(n, nrhs, B, ldb, SX, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    zlag2c(n, n, A, lda, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    cgetrf(n, n, SA, n, ipiv, &iinfo);
    if (iinfo != 0) {
        *iter = -3;
        goto fallback;
    }

    cgetrs("N", n, nrhs, SA, n, ipiv, SX, n, &iinfo);

    clag2z(n, nrhs, SX, n, X, ldx, &iinfo);

    zlacpy("A", n, nrhs, B, ldb, work, n);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, nrhs, n, &NEGONE, A, lda, X, ldx, &ONE, work, n);

    converged = 1;
    for (i = 0; i < nrhs; i++) {
        INT imax_x = izmax1(n, &X[i * ldx], 1);
        INT imax_r = izmax1(n, &work[i * n], 1);
        xnrm = cabs1(X[imax_x + i * ldx]);
        rnrm = cabs1(work[imax_r + i * n]);
        if (rnrm > xnrm * cte) {
            converged = 0;
            break;
        }
    }

    if (converged) {
        *iter = 0;
        return;
    }

    for (iiter = 1; iiter <= ITERMAX; iiter++) {
        zlag2c(n, nrhs, work, n, SX, n, &iinfo);
        if (iinfo != 0) {
            *iter = -2;
            goto fallback;
        }

        cgetrs("N", n, nrhs, SA, n, ipiv, SX, n, &iinfo);

        clag2z(n, nrhs, SX, n, work, n, &iinfo);

        for (i = 0; i < nrhs; i++) {
            cblas_zaxpy(n, &ONE, &work[i * n], 1, &X[i * ldx], 1);
        }

        zlacpy("A", n, nrhs, B, ldb, work, n);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, nrhs, n, &NEGONE, A, lda, X, ldx, &ONE, work, n);

        converged = 1;
        for (i = 0; i < nrhs; i++) {
            INT imax_x = izmax1(n, &X[i * ldx], 1);
            INT imax_r = izmax1(n, &work[i * n], 1);
            xnrm = cabs1(X[imax_x + i * ldx]);
            rnrm = cabs1(work[imax_r + i * n]);
            if (rnrm > xnrm * cte) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *iter = iiter;
            return;
        }
    }

    *iter = -ITERMAX - 1;

fallback:
    zgetrf(n, n, A, lda, ipiv, info);
    if (*info != 0) {
        return;
    }

    zlacpy("A", n, nrhs, B, ldb, X, ldx);
    zgetrs("N", n, nrhs, A, lda, ipiv, X, ldx, info);
}
