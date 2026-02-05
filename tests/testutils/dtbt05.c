/**
 * @file dtbt05.c
 * @brief DTBT05 tests the error bounds from iterative refinement for a
 *        triangular band system.
 *
 * Port of LAPACK TESTING/LIN/dtbt05.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern double dlamch(const char* cmach);

/**
 * DTBT05 tests the error bounds from iterative refinement for the
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
void dtbt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const double* AB, const int ldab,
            const double* B, const int ldb,
            const double* X, const int ldx,
            const double* XACT, const int ldxact,
            const double* ferr, const double* berr,
            double* reslts)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    int notran, unit, upper;
    int i, ifu, imax, j, k, nz;
    double axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

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
        imax = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fmax(fabs(X[imax + j * ldx]), unfl);
        diff = ZERO;
        for (i = 0; i < n; i++) {
            double d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
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
            double r = (diff / xnorm) / ferr[j];
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
            tmp = fabs(B[i + k * ldb]);
            if (upper) {
                if (!notran) {
                    /* UPPER, TRANS: DO 40 J = MAX(I-KD,1), I-IFU
                     * Fortran: AB(KD+1-I+J, I) = A(J, I) (0-based: AB[kd+j-i + i*ldab]) */
                    int jstart = (i - kd > 0) ? i - kd : 0;
                    for (j = jstart; j < i + 1 - ifu; j++) {
                        tmp += fabs(AB[kd + j - i + i * ldab]) * fabs(X[j + k * ldx]);
                    }
                    if (unit) {
                        tmp += fabs(X[i + k * ldx]);
                    }
                } else {
                    /* UPPER, NOTRAN: DO 50 J = I+IFU, MIN(I+KD,N)
                     * Fortran: AB(KD+1+I-J, J) = A(I, J) (0-based: AB[kd+i-j + j*ldab]) */
                    if (unit) {
                        tmp += fabs(X[i + k * ldx]);
                    }
                    int jend = (i + kd < n - 1) ? i + kd : n - 1;
                    for (j = i + ifu; j <= jend; j++) {
                        tmp += fabs(AB[kd + i - j + j * ldab]) * fabs(X[j + k * ldx]);
                    }
                }
            } else {
                if (notran) {
                    /* LOWER, NOTRAN: DO 60 J = MAX(I-KD,1), I-IFU
                     * Fortran: AB(1+I-J, J) = A(I, J) (0-based: AB[i-j + j*ldab]) */
                    int jstart = (i - kd > 0) ? i - kd : 0;
                    for (j = jstart; j < i + 1 - ifu; j++) {
                        tmp += fabs(AB[i - j + j * ldab]) * fabs(X[j + k * ldx]);
                    }
                    if (unit) {
                        tmp += fabs(X[i + k * ldx]);
                    }
                } else {
                    /* LOWER, TRANS: DO 70 J = I+IFU, MIN(I+KD,N)
                     * Fortran: AB(1+J-I, I) = A(J, I) (0-based: AB[j-i + i*ldab]) */
                    if (unit) {
                        tmp += fabs(X[i + k * ldx]);
                    }
                    int jend = (i + kd < n - 1) ? i + kd : n - 1;
                    for (j = i + ifu; j <= jend; j++) {
                        tmp += fabs(AB[j - i + i * ldab]) * fabs(X[j + k * ldx]);
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
