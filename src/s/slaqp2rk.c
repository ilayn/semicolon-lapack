/**
 * @file slaqp2rk.c
 * @brief SLAQP2RK computes truncated QR factorization with column pivoting of a real matrix block using Level 2 BLAS.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAQP2RK computes a truncated (rank K) or full rank Householder QR
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
void slaqp2rk(
    const int m,
    const int n,
    const int nrhs,
    const int ioffset,
    int kmax,
    const f32 abstol,
    const f32 reltol,
    const int kp1,
    const f32 maxc2nrm,
    f32* restrict A,
    const int lda,
    int* K,
    f32* maxc2nrmk,
    f32* relmaxc2nrmk,
    int* restrict jpiv,
    f32* restrict tau,
    f32* restrict vn1,
    f32* restrict vn2,
    f32* restrict work,
    int* info)
{
    int i, itemp, j, jmaxc2nrm, kk, kp, minmnfact, minmnupdt;
    f32 hugeval, temp, temp2, tol3z;

    *info = 0;

    minmnfact = (m - ioffset < n) ? (m - ioffset) : n;
    minmnupdt = (m - ioffset < n + nrhs) ? (m - ioffset) : (n + nrhs);
    kmax = (kmax < minmnfact) ? kmax : minmnfact;
    tol3z = sqrtf(slamch("E"));
    hugeval = slamch("O");

    for (kk = 0; kk < kmax; kk++) {

        i = ioffset + kk;

        if (i == 0) {

            kp = kp1;

        } else {

            kp = kk + cblas_isamax(n - kk, &vn1[kk], 1);

            *maxc2nrmk = vn1[kp];

            if (sisnan(*maxc2nrmk)) {

                *K = kk;
                *info = *K + kp + 1;

                *relmaxc2nrmk = *maxc2nrmk;

                return;
            }

            if (*maxc2nrmk == 0.0f) {

                *K = kk;
                *relmaxc2nrmk = 0.0f;

                for (j = kk; j < minmnfact; j++) {
                    tau[j] = 0.0f;
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
                    tau[j] = 0.0f;
                }

                return;

            }

        }

        if (kp != kk) {
            cblas_sswap(m, &A[0 + kp * lda], 1, &A[0 + kk * lda], 1);
            vn1[kp] = vn1[kk];
            vn2[kp] = vn2[kk];
            itemp = jpiv[kp];
            jpiv[kp] = jpiv[kk];
            jpiv[kk] = itemp;
        }

        if (i < m - 1) {
            slarfg(m - i, &A[i + kk * lda], &A[(i + 1) + kk * lda], 1, &tau[kk]);
        } else {
            tau[kk] = 0.0f;
        }

        if (sisnan(tau[kk])) {
            *K = kk;
            *info = kk + 1;

            *maxc2nrmk = tau[kk];
            *relmaxc2nrmk = tau[kk];

            return;
        }

        if (kk < minmnupdt - 1) {
            slarf1f("L", m - i, n + nrhs - kk - 1, &A[i + kk * lda], 1,
                    tau[kk], &A[i + (kk + 1) * lda], lda, &work[0]);
        }

        if (kk < minmnfact - 1) {

            for (j = kk + 1; j < n; j++) {
                if (vn1[j] != 0.0f) {

                    temp = 1.0f - powf(fabsf(A[i + j * lda]) / vn1[j], 2.0f);
                    temp = (temp > 0.0f) ? temp : 0.0f;
                    temp2 = temp * powf(vn1[j] / vn2[j], 2.0f);
                    if (temp2 <= tol3z) {

                        vn1[j] = cblas_snrm2(m - i - 1, &A[(i + 1) + j * lda], 1);
                        vn2[j] = vn1[j];

                    } else {

                        vn1[j] = vn1[j] * sqrtf(temp);

                    }
                }
            }

        }

    }

    *K = kmax;

    if (*K < minmnfact) {

        jmaxc2nrm = *K + cblas_isamax(n - *K, &vn1[*K], 1);
        *maxc2nrmk = vn1[jmaxc2nrm];

        if (*K == 0) {
            *relmaxc2nrmk = 1.0f;
        } else {
            *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;
        }

    } else {
        *maxc2nrmk = 0.0f;
        *relmaxc2nrmk = 0.0f;
    }

    for (j = *K; j < minmnfact; j++) {
        tau[j] = 0.0f;
    }
}
