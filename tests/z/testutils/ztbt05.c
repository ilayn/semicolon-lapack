/**
 * @file ztbt05.c
 * @brief ZTBT05 tests the error bounds from iterative refinement for a
 *        triangular band system.
 *
 * Port of LAPACK TESTING/LIN/ztbt05.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZTBT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * triangular band matrix.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / (norm(X) * FERR)
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / (NZ*EPS + (*)), where
 *             (*) = NZ*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i)
 *             and NZ = max. number of nonzeros in any row of A, plus 1
 *
 * @param[in]     uplo    Specifies whether the matrix A is upper or lower triangular.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     trans   Specifies the form of the system of equations.
 *                        = 'N': A * X = B  (No transpose)
 *                        = 'T': A'* X = B  (Transpose)
 *                        = 'C': A'* X = B  (Conjugate transpose = Transpose)
 * @param[in]     diag    Specifies whether or not the matrix A is unit triangular.
 *                        = 'N': Non-unit triangular
 *                        = 'U': Unit triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      The number of super-diagonals (UPLO='U') or sub-diagonals
 *                        (UPLO='L') of A. kd >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     AB      Array (ldab, n). The triangular band matrix A.
 * @param[in]     ldab    The leading dimension of AB. ldab >= kd+1.
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[in]     XACT    Array (ldxact, nrhs). The exact solution vectors.
 * @param[in]     ldxact  The leading dimension of XACT. ldxact >= max(1, n).
 * @param[in]     ferr    Array (nrhs). The estimated forward error bounds.
 * @param[in]     berr    Array (nrhs). The componentwise relative backward errors.
 * @param[out]    reslts  Array (2). The test results.
 */
void ztbt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const c128* AB, const INT ldab,
            const c128* B, const INT ldb,
            const c128* X, const INT ldx,
            const c128* XACT, const INT ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    INT notran, unit, upper;
    INT i, ifu, imax, j, k, nz;
    f64 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    /* Quick exit if N = 0 or NRHS = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    ovfl = ONE / unfl;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    unit = (diag[0] == 'U' || diag[0] == 'u');
    nz = (kd < n - 1 ? kd : n - 1) + 1;

    /* Test 1: Compute the maximum of
     * norm(X - XACT) / (norm(X) * FERR)
     * over all the vectors X and XACT using the infinity-norm. */
    errbnd = ZERO;
    for (j = 0; j < nrhs; j++) {
        imax = cblas_izamax(n, &X[j * ldx], 1);
        xnorm = fmax(cabs1(X[imax + j * ldx]), unfl);
        diff = ZERO;
        for (i = 0; i < n; i++) {
            f64 d = cabs1(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > ONE) {
            /* Continue to ratio computation */
        } else if (diff <= ovfl * xnorm) {
            /* Continue to ratio computation */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= ferr[j]) {
            f64 r = (diff / xnorm) / ferr[j];
            if (r > errbnd) errbnd = r;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / (NZ*EPS + (*)), where
     * (*) = NZ*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i) */
    ifu = unit ? 1 : 0;
    reslts[1] = ZERO;

    for (k = 0; k < nrhs; k++) {
        axbi = ZERO;
        for (i = 0; i < n; i++) {
            tmp = cabs1(B[i + k * ldb]);
            if (upper) {
                if (!notran) {
                    /* UPPER, TRANS/CONJTRANS: DO 40 J = MAX(I-KD,1), I-IFU
                     * Fortran: AB(KD+1-I+J, I) = A(J, I) (0-based: AB[kd+j-i + i*ldab]) */
                    INT jstart = (i - kd > 0) ? i - kd : 0;
                    for (j = jstart; j < i + 1 - ifu; j++) {
                        tmp += cabs1(AB[kd + j - i + i * ldab]) * cabs1(X[j + k * ldx]);
                    }
                    if (unit) {
                        tmp += cabs1(X[i + k * ldx]);
                    }
                } else {
                    /* UPPER, NOTRAN: DO 50 J = I+IFU, MIN(I+KD,N)
                     * Fortran: AB(KD+1+I-J, J) = A(I, J) (0-based: AB[kd+i-j + j*ldab]) */
                    if (unit) {
                        tmp += cabs1(X[i + k * ldx]);
                    }
                    INT jend = (i + kd < n - 1) ? i + kd : n - 1;
                    for (j = i + ifu; j <= jend; j++) {
                        tmp += cabs1(AB[kd + i - j + j * ldab]) * cabs1(X[j + k * ldx]);
                    }
                }
            } else {
                if (notran) {
                    /* LOWER, NOTRAN: DO 60 J = MAX(I-KD,1), I-IFU
                     * Fortran: AB(1+I-J, J) = A(I, J) (0-based: AB[i-j + j*ldab]) */
                    INT jstart = (i - kd > 0) ? i - kd : 0;
                    for (j = jstart; j < i + 1 - ifu; j++) {
                        tmp += cabs1(AB[i - j + j * ldab]) * cabs1(X[j + k * ldx]);
                    }
                    if (unit) {
                        tmp += cabs1(X[i + k * ldx]);
                    }
                } else {
                    /* LOWER, TRANS/CONJTRANS: DO 70 J = I+IFU, MIN(I+KD,N)
                     * Fortran: AB(1+J-I, I) = A(J, I) (0-based: AB[j-i + i*ldab]) */
                    if (unit) {
                        tmp += cabs1(X[i + k * ldx]);
                    }
                    INT jend = (i + kd < n - 1) ? i + kd : n - 1;
                    for (j = i + ifu; j <= jend; j++) {
                        tmp += cabs1(AB[j - i + i * ldab]) * cabs1(X[j + k * ldx]);
                    }
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        tmp = berr[k] / (nz * eps + nz * unfl / fmax(axbi, nz * unfl));
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) reslts[1] = tmp;
        }
    }
}
