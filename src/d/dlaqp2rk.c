/**
 * @file dlaqp2rk.c
 * @brief DLAQP2RK computes truncated QR factorization with column pivoting of a real matrix block using Level 2 BLAS.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLAQP2RK computes a truncated (rank K) or full rank Householder QR
 * factorization with column pivoting of a real matrix
 * block A(IOFFSET+1:M,1:N) as
 *
 *   A * P(K) = Q(K) * R(K).
 *
 * The routine uses Level 2 BLAS.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] ioffset
 *          The number of rows that must be pivoted but not factorized.
 *
 * @param[in] kmax
 *          The maximum number of columns to factorize. kmax >= 0.
 *
 * @param[in] abstol
 *          The absolute tolerance for maximum column 2-norm.
 *
 * @param[in] reltol
 *          The relative tolerance for maximum column 2-norm.
 *
 * @param[in] kp1
 *          The index of the column with the maximum 2-norm (0-based).
 *
 * @param[in] maxc2nrm
 *          The maximum column 2-norm of the original matrix.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n+nrhs).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] K
 *          The factorization rank.
 *
 * @param[out] maxc2nrmk
 *          The maximum column 2-norm of the residual matrix.
 *
 * @param[out] relmaxc2nrmk
 *          The ratio maxc2nrmk / maxc2nrm.
 *
 * @param[out] jpiv
 *          Integer array, dimension (n). Column pivot indices.
 *
 * @param[out] tau
 *          Double precision array, dimension (min(m-ioffset, n)).
 *
 * @param[in,out] vn1
 *          Double precision array, dimension (n). Partial column norms.
 *
 * @param[in,out] vn2
 *          Double precision array, dimension (n). Exact column norms.
 *
 * @param[out] work
 *          Double precision array, dimension (n-1).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - = j (1 <= j <= n): NaN detected in column j
 *                         - = j (n+1 <= j <= 2*n): Inf detected in column j-n
 */
void dlaqp2rk(
    const INT m,
    const INT n,
    const INT nrhs,
    const INT ioffset,
    INT kmax,
    const f64 abstol,
    const f64 reltol,
    const INT kp1,
    const f64 maxc2nrm,
    f64* restrict A,
    const INT lda,
    INT* K,
    f64* maxc2nrmk,
    f64* relmaxc2nrmk,
    INT* restrict jpiv,
    f64* restrict tau,
    f64* restrict vn1,
    f64* restrict vn2,
    f64* restrict work,
    INT* info)
{
    INT i, itemp, j, jmaxc2nrm, kk, kp, minmnfact, minmnupdt;
    f64 hugeval, temp, temp2, tol3z;

    *info = 0;

    minmnfact = (m - ioffset < n) ? (m - ioffset) : n;
    minmnupdt = (m - ioffset < n + nrhs) ? (m - ioffset) : (n + nrhs);
    kmax = (kmax < minmnfact) ? kmax : minmnfact;
    tol3z = sqrt(dlamch("E"));
    hugeval = dlamch("O");

    for (kk = 0; kk < kmax; kk++) {

        i = ioffset + kk;

        if (i == 0) {

            kp = kp1;

        } else {

            kp = kk + cblas_idamax(n - kk, &vn1[kk], 1);

            *maxc2nrmk = vn1[kp];

            if (disnan(*maxc2nrmk)) {

                *K = kk;
                *info = *K + kp + 1;

                *relmaxc2nrmk = *maxc2nrmk;

                return;
            }

            if (*maxc2nrmk == 0.0) {

                *K = kk;
                *relmaxc2nrmk = 0.0;

                for (j = kk; j < minmnfact; j++) {
                    tau[j] = 0.0;
                }

                return;

            }

            if (*info == 0 && *maxc2nrmk > hugeval) {
                *info = n + kk + kp + 1;
            }

            *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;

            if (*maxc2nrmk <= abstol || *relmaxc2nrmk <= reltol) {

                *K = kk;

                for (j = kk; j < minmnfact; j++) {
                    tau[j] = 0.0;
                }

                return;

            }

        }

        if (kp != kk) {
            cblas_dswap(m, &A[0 + kp * lda], 1, &A[0 + kk * lda], 1);
            vn1[kp] = vn1[kk];
            vn2[kp] = vn2[kk];
            itemp = jpiv[kp];
            jpiv[kp] = jpiv[kk];
            jpiv[kk] = itemp;
        }

        if (i < m - 1) {
            dlarfg(m - i, &A[i + kk * lda], &A[(i + 1) + kk * lda], 1, &tau[kk]);
        } else {
            tau[kk] = 0.0;
        }

        if (disnan(tau[kk])) {
            *K = kk;
            *info = kk + 1;

            *maxc2nrmk = tau[kk];
            *relmaxc2nrmk = tau[kk];

            return;
        }

        if (kk < minmnupdt - 1) {
            dlarf1f("L", m - i, n + nrhs - kk - 1, &A[i + kk * lda], 1,
                    tau[kk], &A[i + (kk + 1) * lda], lda, &work[0]);
        }

        if (kk < minmnfact - 1) {

            for (j = kk + 1; j < n; j++) {
                if (vn1[j] != 0.0) {

                    temp = 1.0 - pow(fabs(A[i + j * lda]) / vn1[j], 2.0);
                    temp = (temp > 0.0) ? temp : 0.0;
                    temp2 = temp * pow(vn1[j] / vn2[j], 2.0);
                    if (temp2 <= tol3z) {

                        vn1[j] = cblas_dnrm2(m - i - 1, &A[(i + 1) + j * lda], 1);
                        vn2[j] = vn1[j];

                    } else {

                        vn1[j] = vn1[j] * sqrt(temp);

                    }
                }
            }

        }

    }

    *K = kmax;

    if (*K < minmnfact) {

        jmaxc2nrm = *K + cblas_idamax(n - *K, &vn1[*K], 1);
        *maxc2nrmk = vn1[jmaxc2nrm];

        if (*K == 0) {
            *relmaxc2nrmk = 1.0;
        } else {
            *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;
        }

    } else {
        *maxc2nrmk = 0.0;
        *relmaxc2nrmk = 0.0;
    }

    for (j = *K; j < minmnfact; j++) {
        tau[j] = 0.0;
    }
}
