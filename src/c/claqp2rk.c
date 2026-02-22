/**
 * @file claqp2rk.c
 * @brief CLAQP2RK computes truncated QR factorization with column pivoting of a complex matrix block using Level 2 BLAS.
 */

#include <complex.h>
#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CLAQP2RK computes a truncated (rank K) or full rank Householder QR
 * factorization with column pivoting of a complex matrix
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
 *          Complex*16 array, dimension (lda, n+nrhs).
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
 *          Complex*16 array, dimension (min(m-ioffset, n)).
 *
 * @param[in,out] vn1
 *          Single precision array, dimension (n). Partial column norms.
 *
 * @param[in,out] vn2
 *          Single precision array, dimension (n). Exact column norms.
 *
 * @param[out] work
 *          Complex*16 array, dimension (n-1).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - = j (1 <= j <= n): NaN detected in column j
 *                         - = j (n+1 <= j <= 2*n): Inf detected in column j-n
 */
void claqp2rk(
    const INT m,
    const INT n,
    const INT nrhs,
    const INT ioffset,
    INT kmax,
    const f32 abstol,
    const f32 reltol,
    const INT kp1,
    const f32 maxc2nrm,
    c64* restrict A,
    const INT lda,
    INT* K,
    f32* maxc2nrmk,
    f32* relmaxc2nrmk,
    INT* restrict jpiv,
    c64* restrict tau,
    f32* restrict vn1,
    f32* restrict vn2,
    c64* restrict work,
    INT* info)
{
    INT i, itemp, j, jmaxc2nrm, kk, kp, minmnfact, minmnupdt;
    f32 hugeval, taunan, temp, temp2, tol3z;

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
                    tau[j] = CMPLXF(0.0f, 0.0f);
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
                    tau[j] = CMPLXF(0.0f, 0.0f);
                }

                return;

            }

        }

        if (kp != kk) {
            cblas_cswap(m, &A[0 + kp * lda], 1, &A[0 + kk * lda], 1);
            vn1[kp] = vn1[kk];
            vn2[kp] = vn2[kk];
            itemp = jpiv[kp];
            jpiv[kp] = jpiv[kk];
            jpiv[kk] = itemp;
        }

        if (i < m - 1) {
            clarfg(m - i, &A[i + kk * lda], &A[(i + 1) + kk * lda], 1, &tau[kk]);
        } else {
            tau[kk] = CMPLXF(0.0f, 0.0f);
        }

        if (sisnan(crealf(tau[kk]))) {
            taunan = crealf(tau[kk]);
        } else if (sisnan(cimagf(tau[kk]))) {
            taunan = cimagf(tau[kk]);
        } else {
            taunan = 0.0f;
        }

        if (sisnan(taunan)) {
            *K = kk;
            *info = kk + 1;

            *maxc2nrmk = taunan;
            *relmaxc2nrmk = taunan;

            return;
        }

        if (kk < minmnupdt - 1) {
            clarf1f("L", m - i, n + nrhs - kk - 1, &A[i + kk * lda], 1,
                    conjf(tau[kk]), &A[i + (kk + 1) * lda], lda, &work[0]);
        }

        if (kk < minmnfact - 1) {

            for (j = kk + 1; j < n; j++) {
                if (vn1[j] != 0.0f) {

                    temp = 1.0f - powf(cabsf(A[i + j * lda]) / vn1[j], 2.0f);
                    temp = (temp > 0.0f) ? temp : 0.0f;
                    temp2 = temp * powf(vn1[j] / vn2[j], 2.0f);
                    if (temp2 <= tol3z) {

                        vn1[j] = cblas_scnrm2(m - i - 1, &A[(i + 1) + j * lda], 1);
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
        tau[j] = CMPLXF(0.0f, 0.0f);
    }
}
