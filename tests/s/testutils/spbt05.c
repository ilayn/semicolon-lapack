/**
 * @file spbt05.c
 * @brief SPBT05 tests the error bounds from iterative refinement for the
 *        computed solution to a system of equations A*X = B, where A is a
 *        symmetric band matrix.
 *
 * Port of LAPACK TESTING/LIN/spbt05.f
 */

#include <math.h>
#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SPBT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * symmetric band matrix.
 *
 * RESLTS(1) = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS(2) = residual from the iterative refinement routine
 *           = the maximum of BERR / ( NZ*EPS + (*) ), where
 *             (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
 *             and NZ = max. number of nonzeros in any row of A, plus 1
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows of the matrices X, B, and XACT,
 *                        and the order of the matrix A. n >= 0.
 * @param[in]     kd      The number of super-diagonals of the matrix A if
 *                        uplo = 'U', or the number of sub-diagonals if
 *                        uplo = 'L'. kd >= 0.
 * @param[in]     nrhs    The number of columns of the matrices X, B, and XACT.
 *                        nrhs >= 0.
 * @param[in]     AB      The upper or lower triangle of the symmetric band
 *                        matrix A, stored in the first kd+1 rows.
 *                        Dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of AB. ldab >= kd+1.
 * @param[in]     B       The right hand side matrix. Dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[in]     X       The computed solution vectors. Dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1,n).
 * @param[in]     XACT    The exact solution vectors. Dimension (ldxact, nrhs).
 * @param[in]     ldxact  The leading dimension of XACT. ldxact >= max(1,n).
 * @param[in]     ferr    The estimated forward error bounds. Dimension (nrhs).
 * @param[in]     berr    The componentwise relative backward errors.
 *                        Dimension (nrhs).
 * @param[out]    reslts  The maximum over the nrhs solution vectors of the
 *                        ratios:
 *                        RESLTS(1) = norm(X - XACT) / ( norm(X) * FERR )
 *                        RESLTS(2) = BERR / ( NZ*EPS + (*) )
 *                        Dimension (2).
 */
void spbt05(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32* B, const INT ldb,
            const f32* X, const INT ldx,
            const f32* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f32 eps = slamch("Epsilon");
    f32 unfl = slamch("Safe minimum");
    f32 ovfl = ONE / unfl;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    INT nz = 2 * ((kd > n - 1) ? kd : n - 1) + 1;

    f32 errbnd = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        INT imax = cblas_isamax(n, &X[j * ldx], 1);
        f32 xnorm = fabsf(X[imax + j * ldx]);
        if (xnorm < unfl) {
            xnorm = unfl;
        }

        f32 diff = ZERO;
        for (INT i = 0; i < n; i++) {
            f32 tmp = fabsf(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (tmp > diff) {
                diff = tmp;
            }
        }

        if (xnorm > ONE) {
            goto label20;
        } else if (diff <= ovfl * xnorm) {
            goto label20;
        } else {
            errbnd = ONE / eps;
            continue;
        }

label20:
        if (diff / xnorm <= ferr[j]) {
            f32 tmp = (diff / xnorm) / ferr[j];
            if (tmp > errbnd) {
                errbnd = tmp;
            }
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    for (INT k = 0; k < nrhs; k++) {
        f32 axbi = ZERO;
        for (INT i = 0; i < n; i++) {
            f32 tmp = fabsf(B[i + k * ldb]);

            if (upper) {
                INT j_start = (i - kd > 0) ? i - kd : 0;
                for (INT jj = j_start; jj <= i; jj++) {
                    tmp += fabsf(AB[kd - i + jj + i * ldab]) * fabsf(X[jj + k * ldx]);
                }
                INT j_end = (i + kd < n - 1) ? i + kd : n - 1;
                for (INT jj = i + 1; jj <= j_end; jj++) {
                    tmp += fabsf(AB[kd + i - jj + jj * ldab]) * fabsf(X[jj + k * ldx]);
                }
            } else {
                INT j_start = (i - kd > 0) ? i - kd : 0;
                for (INT jj = j_start; jj < i; jj++) {
                    tmp += fabsf(AB[i - jj + jj * ldab]) * fabsf(X[jj + k * ldx]);
                }
                INT j_end = (i + kd < n - 1) ? i + kd : n - 1;
                for (INT jj = i; jj <= j_end; jj++) {
                    tmp += fabsf(AB[jj - i + i * ldab]) * fabsf(X[jj + k * ldx]);
                }
            }

            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) {
                    axbi = tmp;
                }
            }
        }
        f32 denom = nz * eps + nz * unfl / ((axbi > nz * unfl) ? axbi : nz * unfl);
        f32 ratio = berr[k] / denom;
        if (k == 0) {
            reslts[1] = ratio;
        } else {
            if (ratio > reslts[1]) {
                reslts[1] = ratio;
            }
        }
    }
}
