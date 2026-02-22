/**
 * @file sgbsvx.c
 * @brief SGBSVX computes the solution to a real system of linear equations
 *        A * X = B for banded matrices, using the LU factorization with
 *        equilibration and iterative refinement.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGBSVX uses the LU factorization to compute the solution to a real
 * system of linear equations A * X = B, A**T * X = B, or A**H * X = B,
 * where A is a band matrix of order N with KL subdiagonals and KU
 * superdiagonals, and X and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @param[in]     fact    'F': AFB and IPIV contain the factored form of A.
 *                        'N': The matrix A will be copied to AFB and factored.
 *                        'E': The matrix A will be equilibrated if necessary,
 *                             then copied to AFB and factored.
 * @param[in]     trans   'N': A * X = B (No transpose)
 *                        'T': A**T * X = B (Transpose)
 *                        'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n       The number of linear equations (order of A). n >= 0.
 * @param[in]     kl      The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku      The number of superdiagonals within the band of A. ku >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in,out] AB      On entry, the matrix A in band storage, in rows 0 to kl+ku.
 *                        On exit, if equilibration was done, A is scaled.
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of AB. ldab >= kl+ku+1.
 * @param[in,out] AFB     On entry (if fact='F'), contains the LU factors.
 *                        On exit, contains the factors L and U.
 *                        Array of dimension (ldafb, n).
 * @param[in]     ldafb   The leading dimension of AFB. ldafb >= 2*kl+ku+1.
 * @param[in,out] ipiv    Pivot indices from factorization. Array of dimension (n).
 * @param[in,out] equed   On entry (if fact='F'), specifies equilibration done.
 *                        On exit, specifies the form of equilibration:
 *                        'N': No equilibration
 *                        'R': Row equilibration (A := diag(R) * A)
 *                        'C': Column equilibration (A := A * diag(C))
 *                        'B': Both (A := diag(R) * A * diag(C))
 * @param[in,out] R       Row scale factors. Array of dimension (n).
 * @param[in,out] C       Column scale factors. Array of dimension (n).
 * @param[in,out] B       On entry, the N-by-NRHS right hand side matrix B.
 *                        On exit, if equilibration was done, B is scaled.
 *                        Array of dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    X       The N-by-NRHS solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[out]    rcond   Reciprocal condition number estimate.
 * @param[out]    ferr    Forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr    Backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work    Workspace array of dimension (3*n).
 *                        On exit, work[0] contains the reciprocal pivot growth factor.
 * @param[out]    iwork   Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, U(i,i) is exactly zero (1-based).
 *                           if info = n+1, U is nonsingular but RCOND < machine precision.
 */
void sgbsvx(
    const char* fact,
    const char* trans,
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    f32* restrict AB,
    const INT ldab,
    f32* restrict AFB,
    const INT ldafb,
    INT* restrict ipiv,
    char* equed,
    f32* restrict R,
    f32* restrict C,
    f32* restrict B,
    const INT ldb,
    f32* restrict X,
    const INT ldx,
    f32* rcond,
    f32* restrict ferr,
    f32* restrict berr,
    f32* restrict work,
    INT* restrict iwork,
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
        xerbla("SGBSVX", -(*info));
        return;
    }

    if (equil) {
        /* Compute row and column scalings to equilibrate the matrix A */
        sgbequ(n, n, kl, ku, AB, ldab, R, C, &rowcnd, &colcnd, &amax, &infequ);
        if (infequ == 0) {
            /* Equilibrate the matrix */
            slaqgb(n, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax, equed);
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
            cblas_scopy(len, &AB[ku + j1 - j + j * ldab], 1,
                        &AFB[kl + ku + j1 - j + j * ldafb], 1);
        }

        sgbtrf(n, n, kl, ku, AFB, ldafb, ipiv, info);

        /* Return if INFO is non-zero */
        if (*info > 0) {
            /* Compute the reciprocal pivot growth factor of the
             * leading rank-deficient INFO columns of A */
            anorm = ZERO;
            for (j = 0; j < *info; j++) {
                INT i_start = (ku - j > 0) ? ku - j : 0;
                INT i_end = (n + ku - j - 1 < kl + ku) ? n + ku - j - 1 : kl + ku;
                for (i = i_start; i <= i_end; i++) {
                    f32 temp = fabsf(AB[i + j * ldab]);
                    if (anorm < temp) anorm = temp;
                }
            }
            INT k_val = (*info - 1 < kl + ku) ? *info - 1 : kl + ku;
            INT start_offset = (kl + ku + 1 - *info > 0) ? kl + ku + 1 - *info : 0;
            rpvgrw = slantb("M", "U", "N", *info, k_val, &AFB[start_offset], ldafb, work);
            if (rpvgrw == ZERO) {
                rpvgrw = ONE;
            } else {
                rpvgrw = anorm / rpvgrw;
            }
            work[0] = rpvgrw;
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
    anorm = slangb(&norm, n, kl, ku, AB, ldab, work);
    rpvgrw = slantb("M", "U", "N", n, kl + ku, AFB, ldafb, work);
    if (rpvgrw == ZERO) {
        rpvgrw = ONE;
    } else {
        rpvgrw = slangb("M", n, kl, ku, AB, ldab, work) / rpvgrw;
    }

    /* Compute the reciprocal of the condition number of A */
    sgbcon(&norm, n, kl, ku, AFB, ldafb, ipiv, anorm, rcond, work, iwork, info);

    /* Compute the solution matrix X */
    slacpy("F", n, nrhs, B, ldb, X, ldx);
    sgbtrs(trans, n, kl, ku, nrhs, AFB, ldafb, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solution and
     * compute error bounds and backward error estimates for it */
    sgbrfs(trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv,
           B, ldb, X, ldx, ferr, berr, work, iwork, info);

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

    work[0] = rpvgrw;
}
