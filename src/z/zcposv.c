/**
 * @file zcposv.c
 * @brief ZCPOSV computes the solution to a complex system of linear equations
 *        A * X = B for Hermitian positive definite matrices using mixed
 *        precision iterative refinement.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"
#include "semicolon_lapack_complex_single.h"

/**
 * ZCPOSV computes the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian positive definite matrix and X and B
 * are N-by-NRHS matrices.
 *
 * ZCPOSV first attempts to factorize the matrix in COMPLEX and use this
 * factorization within an iterative refinement procedure to produce a
 * solution with COMPLEX*16 normwise backward error quality (see below).
 * If the approach fails the method switches to a COMPLEX*16
 * factorization and solve.
 *
 * The iterative refinement process is stopped if
 *     ITER > ITERMAX
 * or for all the RHS we have:
 *     RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
 * where
 *     o ITER is the number of the current iteration in the iterative
 *       refinement process
 *     o RNRM is the infinity-norm of the residual
 *     o XNRM is the infinity-norm of the solution
 *     o ANRM is the infinity-operator-norm of the matrix A
 *     o EPS is the machine epsilon returned by DLAMCH('Epsilon')
 * The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00
 * respectively.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the Hermitian matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The number of linear equations, i.e., the order of
 *                      the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. nrhs >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A. If UPLO = 'U', the
 *                      leading N-by-N upper triangular part of A contains the
 *                      upper triangular part of the matrix A. If UPLO = 'L',
 *                      the leading N-by-N lower triangular part of A contains
 *                      the lower triangular part of the matrix A.
 *                      Note that the imaginary parts of the diagonal
 *                      elements need not be set and are assumed to be zero.
 *                      On exit, if iterative refinement has been successfully
 *                      used (info = 0 and iter >= 0) then A is unchanged.
 *                      If COMPLEX*16 factorization has been used
 *                      (info = 0 and iter < 0) then the array A contains
 *                      the factor U or L from the Cholesky factorization
 *                      A = U**H*U or A = L*L**H.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     B     Complex*16 array, dimension (ldb, nrhs).
 *                      The N-by-NRHS right hand side matrix B.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    X     Complex*16 array, dimension (ldx, nrhs).
 *                      If info = 0, the N-by-NRHS solution matrix X.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1, n).
 * @param[out]    work  Complex*16 array, dimension (n, nrhs).
 *                      This array is used to hold the residual vectors.
 * @param[out]    swork Complex (single precision) array, dimension (n*(n+nrhs)).
 *                      This array is used to use the single precision matrix
 *                      and the right-hand sides or solutions in single precision.
 * @param[out]    rwork Double precision array, dimension (n).
 * @param[out]    iter  Iteration count:
 *                      - < 0: iterative refinement has failed, COMPLEX*16
 *                        factorization has been performed
 *                        - -1 : the routine fell back to full precision for
 *                          implementation- or machine-specific reasons
 *                        - -2 : narrowing the precision induced an overflow,
 *                          the routine fell back to full precision
 *                        - -3 : failure of CPOTRF
 *                        - -31: stop the iterative refinement after the 30th
 *                          iterations
 *                      - > 0: iterative refinement has been successfully used.
 *                        Returns the number of iterations
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, the leading principal minor of order i
 *                           of (COMPLEX*16) A is not positive, so the
 *                           factorization could not be completed, and the
 *                           solution has not been computed.
 */
void zcposv(
    const char* uplo,
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
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

    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("ZCPOSV", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    anrm = zlanhe("I", uplo, n, A, lda, rwork);
    eps = dlamch("E");
    cte = anrm * eps * sqrt((f64)n) * BWDMAX;

    SA = swork;
    SX = &swork[n * n];

    zlag2c(n, nrhs, B, ldb, SX, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    zlat2c(uplo, n, A, lda, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    cpotrf(uplo, n, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -3;
        goto fallback;
    }

    cpotrs(uplo, n, nrhs, SA, n, SX, n, &iinfo);

    clag2z(n, nrhs, SX, n, X, ldx, &iinfo);

    zlacpy("A", n, nrhs, B, ldb, work, n);
    cblas_zhemm(CblasColMajor,
                CblasLeft,
                upper ? CblasUpper : CblasLower,
                n, nrhs, &NEGONE, A, lda, X, ldx, &ONE, work, n);

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

        cpotrs(uplo, n, nrhs, SA, n, SX, n, &iinfo);

        clag2z(n, nrhs, SX, n, work, n, &iinfo);

        for (i = 0; i < nrhs; i++) {
            cblas_zaxpy(n, &ONE, &work[i * n], 1, &X[i * ldx], 1);
        }

        zlacpy("A", n, nrhs, B, ldb, work, n);
        cblas_zhemm(CblasColMajor,
                    CblasLeft,
                    upper ? CblasUpper : CblasLower,
                    n, nrhs, &NEGONE, A, lda, X, ldx, &ONE, work, n);

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
    zpotrf(uplo, n, A, lda, info);
    if (*info != 0) {
        return;
    }

    zlacpy("A", n, nrhs, B, ldb, X, ldx);
    zpotrs(uplo, n, nrhs, A, lda, X, ldx, info);
}
