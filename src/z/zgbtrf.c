/**
 * @file zgbtrf.c
 * @brief ZGBTRF computes the LU factorization of a general band matrix
 *        using the blocked version of the algorithm.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/* Maximum block size for work arrays */
#define NBMAX 64
#define LDWORK (NBMAX + 1)

/**
 * ZGBTRF computes an LU factorization of a complex m-by-n band matrix A
 * using partial pivoting with row interchanges.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * The factorization has the form A = P * L * U where P is a permutation
 * matrix, L is lower triangular with unit diagonal elements (lower
 * trapezoidal if m > n), and U is upper triangular (upper trapezoidal
 * if m < n).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     kl    The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku    The number of superdiagonals within the band of A. ku >= 0.
 * @param[in,out] AB    Double complex array, dimension (ldab, n).
 *                      On entry, the matrix A in band storage, in rows kl to
 *                      2*kl+ku; rows 0 to kl-1 of the array need not be set.
 *                      The j-th column of A is stored in the j-th column of
 *                      the array AB as follows:
 *                      AB[kl+ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl).
 *
 *                      On exit, details of the factorization: U is stored as an
 *                      upper triangular band matrix with kl+ku superdiagonals in
 *                      rows 0 to kl+ku, and the multipliers used during the
 *                      factorization are stored in rows kl+ku+1 to 2*kl+ku.
 * @param[in]     ldab  The leading dimension of the array AB. ldab >= 2*kl+ku+1.
 * @param[out]    ipiv  Integer array, dimension (min(m,n)).
 *                      The pivot indices; for 0 <= i < min(m,n), row i of the
 *                      matrix was interchanged with row ipiv[i]. 0-based indexing.
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but the factor U is exactly
 *                           singular, and division by zero will occur if it is used
 *                           to solve a system of equations.
 */
void zgbtrf(
    const INT m,
    const INT n,
    const INT kl,
    const INT ku,
    c128* restrict AB,
    const INT ldab,
    INT* restrict ipiv,
    INT* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    /* Work arrays for A13 and A31 blocks that fall outside the band */
    c128 work13[LDWORK * NBMAX];
    c128 work31[LDWORK * NBMAX];

    INT i, i2, i3, ii, ip, j, j2, j3, jb, jj, jm, jp, ju, k2, km, kv, nb, nw;
    c128 temp;
    INT minmn;

    /* kv is the number of superdiagonals in the factor U, allowing for fill-in */
    kv = ku + kl;

    /* Test the input parameters */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (ldab < kl + kv + 1) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("ZGBTRF", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Determine the block size for this environment */
    nb = lapack_get_nb("GBTRF");

    /* The block size must not exceed the limit set by the size of the
       local arrays work13 and work31 */
    if (nb > NBMAX) nb = NBMAX;

    if (nb <= 1 || nb > kl) {
        /* Use unblocked code */
        zgbtf2(m, n, kl, ku, AB, ldab, ipiv, info);
    } else {
        /* Use blocked code */

        /* Zero the superdiagonal elements of the work array work13 */
        for (j = 0; j < nb; j++) {
            for (i = 0; i < j; i++) {
                work13[i + j * LDWORK] = ZERO;
            }
        }

        /* Zero the subdiagonal elements of the work array work31 */
        for (j = 0; j < nb; j++) {
            for (i = j + 1; i < nb; i++) {
                work31[i + j * LDWORK] = ZERO;
            }
        }

        /* Gaussian elimination with partial pivoting */

        /* Set fill-in elements in columns ku+1 to kv-1 to zero
           0-based: columns ku+1 to kv-1 */
        for (j = ku + 1; j < kv && j < n; j++) {
            for (i = kv - j; i < kl; i++) {
                AB[i + j * ldab] = ZERO;
            }
        }

        /* ju is the index of the last column affected by the current
           stage of the factorization. 0-based. */
        ju = 0;

        minmn = (m < n) ? m : n;

        for (j = 0; j < minmn; j += nb) {
            jb = (minmn - j < nb) ? minmn - j : nb;

            /* The active part of the matrix is partitioned
             *
             *   A11   A12   A13
             *   A21   A22   A23
             *   A31   A32   A33
             *
             * Here A11, A21 and A31 denote the current block of jb columns
             * which is about to be factorized. The number of rows in the
             * partitioning are jb, i2, i3 respectively, and the numbers
             * of columns are jb, j2, j3. The superdiagonal elements of A13
             * and the subdiagonal elements of A31 lie outside the band.
             */
            i2 = (kl - jb < m - j - jb) ? kl - jb : m - j - jb;
            if (i2 < 0) i2 = 0;
            i3 = (jb < m - j - kl) ? jb : m - j - kl;
            if (i3 < 0) i3 = 0;

            /* j2 and j3 are computed after ju has been updated. */

            /* Factorize the current block of jb columns */
            for (jj = j; jj < j + jb; jj++) {
                /* Set fill-in elements in column jj+kv to zero */
                if (jj + kv < n) {
                    for (i = 0; i < kl; i++) {
                        AB[i + (jj + kv) * ldab] = ZERO;
                    }
                }

                /* Find pivot and test for singularity. km is the number of
                   subdiagonal elements in the current column. */
                km = (kl < m - 1 - jj) ? kl : m - 1 - jj;
                if (km < 0) km = 0;

                jp = cblas_izamax(km + 1, &AB[kv + jj * ldab], 1);
                ipiv[jj] = jp + jj - j;  /* Store relative to block start initially */

                if (AB[kv + jp + jj * ldab] != ZERO) {
                    INT new_ju = jj + ku + jp;
                    if (new_ju > n - 1) new_ju = n - 1;
                    if (new_ju > ju) ju = new_ju;

                    if (jp != 0) {
                        /* Apply interchange to columns j to j+jb-1 */
                        if (jp + jj < j + kl) {
                            /* The interchange does not affect A31 */
                            cblas_zswap(jb, &AB[kv + jj - j + j * ldab], ldab - 1,
                                        &AB[kv + jp + jj - j + j * ldab], ldab - 1);
                        } else {
                            /* The interchange affects columns j to jj-1 of A31
                               which are stored in the work array work31 */
                            cblas_zswap(jj - j, &AB[kv + jj - j + j * ldab], ldab - 1,
                                        &work31[jp + jj - j - kl + 0 * LDWORK], LDWORK);
                            cblas_zswap(j + jb - jj, &AB[kv + jj * ldab], ldab - 1,
                                        &AB[kv + jp + jj * ldab], ldab - 1);
                        }
                    }

                    /* Compute multipliers */
                    {
                        const c128 scale = ONE / AB[kv + jj * ldab];
                        cblas_zscal(km, &scale, &AB[kv + 1 + jj * ldab], 1);
                    }

                    /* Update trailing submatrix within the band and within
                       the current block. jm is the index of the last column
                       which needs to be updated. */
                    jm = (ju < j + jb - 1) ? ju : j + jb - 1;
                    if (jm > jj) {
                        cblas_zgeru(CblasColMajor, km, jm - jj, &NEG_ONE,
                                   &AB[kv + 1 + jj * ldab], 1,
                                   &AB[kv - 1 + (jj + 1) * ldab], ldab - 1,
                                   &AB[kv + (jj + 1) * ldab], ldab - 1);
                    }
                } else {
                    /* If pivot is zero, set INFO to the index of the pivot
                       unless a zero pivot has already been found. */
                    if (*info == 0) {
                        *info = jj + 1;  /* 1-based for reporting */
                    }
                }

                /* Copy current column of A31 into the work array work31 */
                nw = (jj - j + 1 < i3) ? jj - j + 1 : i3;
                if (nw > 0) {
                    cblas_zcopy(nw, &AB[kv + kl - jj + j + jj * ldab], 1,
                                &work31[0 + (jj - j) * LDWORK], 1);
                }
            }

            if (j + jb < n) {
                /* Apply the row interchanges to the other blocks */
                j2 = (ju - j + 1 < kv) ? ju - j + 1 : kv;
                j2 = j2 - jb;
                if (j2 < 0) j2 = 0;
                j3 = ju - j - kv + 1;
                if (j3 < 0) j3 = 0;

                /* Use zlaswp to apply the row interchanges to A12, A22, and A32 */
                zlaswp(j2, &AB[kv - jb + (j + jb) * ldab], ldab - 1, 0, jb - 1, &ipiv[j], 1);

                /* Adjust the pivot indices */
                for (i = j; i < j + jb; i++) {
                    ipiv[i] = ipiv[i] + j;
                }

                /* Apply the row interchanges to A13, A23, and A33 columnwise */
                k2 = j + jb + j2;
                for (i = 0; i < j3; i++) {
                    jj = k2 + i;
                    for (ii = j + i; ii < j + jb; ii++) {
                        ip = ipiv[ii];
                        if (ip != ii) {
                            temp = AB[kv + ii - jj + jj * ldab];
                            AB[kv + ii - jj + jj * ldab] = AB[kv + ip - jj + jj * ldab];
                            AB[kv + ip - jj + jj * ldab] = temp;
                        }
                    }
                }

                /* Update the relevant part of the trailing submatrix */
                if (j2 > 0) {
                    /* Update A12 */
                    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                                jb, j2, &ONE, &AB[kv + j * ldab], ldab - 1,
                                &AB[kv - jb + (j + jb) * ldab], ldab - 1);

                    if (i2 > 0) {
                        /* Update A22 */
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i2, j2, jb, &NEG_ONE,
                                    &AB[kv + jb + j * ldab], ldab - 1,
                                    &AB[kv - jb + (j + jb) * ldab], ldab - 1,
                                    &ONE, &AB[kv + (j + jb) * ldab], ldab - 1);
                    }

                    if (i3 > 0) {
                        /* Update A32 */
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i3, j2, jb, &NEG_ONE,
                                    work31, LDWORK,
                                    &AB[kv - jb + (j + jb) * ldab], ldab - 1,
                                    &ONE, &AB[kv + kl - jb + (j + jb) * ldab], ldab - 1);
                    }
                }

                if (j3 > 0) {
                    /* Copy the lower triangle of A13 into the work array work13 */
                    for (jj = 0; jj < j3; jj++) {
                        for (ii = jj; ii < jb; ii++) {
                            work13[ii + jj * LDWORK] = AB[ii - jj + (jj + j + kv) * ldab];
                        }
                    }

                    /* Update A13 in the work array */
                    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                                jb, j3, &ONE, &AB[kv + j * ldab], ldab - 1,
                                work13, LDWORK);

                    if (i2 > 0) {
                        /* Update A23 */
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i2, j3, jb, &NEG_ONE,
                                    &AB[kv + jb + j * ldab], ldab - 1,
                                    work13, LDWORK,
                                    &ONE, &AB[jb + (j + kv) * ldab], ldab - 1);
                    }

                    if (i3 > 0) {
                        /* Update A33 */
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i3, j3, jb, &NEG_ONE,
                                    work31, LDWORK,
                                    work13, LDWORK,
                                    &ONE, &AB[kl + (j + kv) * ldab], ldab - 1);
                    }

                    /* Copy the lower triangle of A13 back into place */
                    for (jj = 0; jj < j3; jj++) {
                        for (ii = jj; ii < jb; ii++) {
                            AB[ii - jj + (jj + j + kv) * ldab] = work13[ii + jj * LDWORK];
                        }
                    }
                }
            } else {
                /* Adjust the pivot indices */
                for (i = j; i < j + jb; i++) {
                    ipiv[i] = ipiv[i] + j;
                }
            }

            /* Partially undo the interchanges in the current block to
               restore the upper triangular form of A31 and copy the upper
               triangle of A31 back into place */
            for (jj = j + jb - 1; jj >= j; jj--) {
                jp = ipiv[jj] - jj;
                if (jp != 0) {
                    /* Apply interchange to columns j to jj-1 */
                    if (jp + jj < j + kl) {
                        /* The interchange does not affect A31 */
                        cblas_zswap(jj - j, &AB[kv + jj - j + j * ldab], ldab - 1,
                                    &AB[kv + jp + jj - j + j * ldab], ldab - 1);
                    } else {
                        /* The interchange does affect A31 */
                        cblas_zswap(jj - j, &AB[kv + jj - j + j * ldab], ldab - 1,
                                    &work31[jp + jj - j - kl + 0 * LDWORK], LDWORK);
                    }
                }

                /* Copy the current column of A31 back into place */
                nw = (i3 < jj - j + 1) ? i3 : jj - j + 1;
                if (nw > 0) {
                    cblas_zcopy(nw, &work31[0 + (jj - j) * LDWORK], 1,
                                &AB[kv + kl - jj + j + jj * ldab], 1);
                }
            }
        }
    }
}
