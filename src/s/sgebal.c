/**
 * @file sgebal.c
 * @brief SGEBAL balances a general real matrix A.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>

/**
 * SGEBAL balances a general real matrix A. This involves, first,
 * permuting A by a similarity transformation to isolate eigenvalues
 * in the first 0 to ILO-1 and last IHI+1 to N-1 elements on the
 * diagonal; and second, applying a diagonal similarity transformation
 * to rows and columns ILO to IHI to make the rows and columns as
 * close in norm as possible. Both steps are optional.
 *
 * Balancing may reduce the 1-norm of the matrix, and improve the
 * accuracy of the computed eigenvalues and/or eigenvectors.
 *
 * @param[in]     job    Specifies the operations to be performed on A:
 *                       = 'N': none: simply set ILO = 0, IHI = N-1, SCALE(I) = 1.0
 *                              for i = 0,...,N-1;
 *                       = 'P': permute only;
 *                       = 'S': scale only;
 *                       = 'B': both permute and scale.
 * @param[in]     n      The order of the matrix A. N >= 0.
 * @param[in,out] A      Double precision array, dimension (LDA,N).
 *                       On entry, the input matrix A.
 *                       On exit, A is overwritten by the balanced matrix.
 *                       If JOB = 'N', A is not referenced.
 * @param[in]     lda    The leading dimension of the array A. LDA >= max(1,N).
 * @param[out]    ilo    ILO and IHI are set to integers such that on exit
 *                       A(i,j) = 0 if i > j and j = 0,...,ILO-1 or I = IHI+1,...,N-1.
 *                       If JOB = 'N' or 'S', ILO = 0 and IHI = N-1.
 *                       Uses 0-based indexing.
 * @param[out]    ihi    See ILO. Uses 0-based indexing.
 * @param[out]    scale  Double precision array, dimension (N).
 *                       Details of the permutations and scaling factors applied to A.
 *                       If P(j) is the index of the row and column interchanged
 *                       with row and column j and D(j) is the scaling factor
 *                       applied to row and column j, then
 *                       SCALE(j) = P(j)    for j = 0,...,ILO-1
 *                                = D(j)    for j = ILO,...,IHI
 *                                = P(j)    for j = IHI+1,...,N-1.
 *                       The order in which the interchanges are made is N-1 to IHI+1,
 *                       then 0 to ILO-1. Permutation indices are 0-based.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if INFO = -i, the i-th argument had an illegal value.
 */
void sgebal(const char* job, const INT n, f32* A, const INT lda,
            INT* ilo, INT* ihi, f32* scale, INT* info)
{
    /* Constants */
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 SCLFAC = 2.0f;
    const f32 FACTOR = 0.95f;

    /* Local variables */
    INT noconv, canswap;
    INT i, ica, ira, j, k, l;
    f32 c, ca, f, g, r, ra, s, sfmax1, sfmax2, sfmin1, sfmin2;

    /* Test the input parameters */
    *info = 0;
    if (!(job[0] == 'N' || job[0] == 'n') &&
        !(job[0] == 'P' || job[0] == 'p') &&
        !(job[0] == 'S' || job[0] == 's') &&
        !(job[0] == 'B' || job[0] == 'b')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SGEBAL", -(*info));
        return;
    }

    /* Quick returns */
    if (n == 0) {
        *ilo = 0;
        *ihi = -1;  /* Empty range */
        return;
    }

    if (job[0] == 'N' || job[0] == 'n') {
        for (i = 0; i < n; i++) {
            scale[i] = ONE;
        }
        *ilo = 0;
        *ihi = n - 1;
        return;
    }

    /* Permutation to isolate eigenvalues if possible */
    k = 0;
    l = n - 1;

    if (!(job[0] == 'S' || job[0] == 's')) {
        /* Row and column exchange */

        /* Search for rows isolating an eigenvalue and push them down */
        noconv = 1;
        while (noconv) {
            noconv = 0;
            for (i = l; i >= 0; i--) {
                canswap = 1;
                for (j = 0; j <= l; j++) {
                    if (i != j && A[i + j * lda] != ZERO) {
                        canswap = 0;
                        break;
                    }
                }

                if (canswap) {
                    scale[l] = (f32)i;  /* Store 0-based index */
                    if (i != l) {
                        /* Swap columns i and l */
                        cblas_sswap(l + 1, &A[0 + i * lda], 1, &A[0 + l * lda], 1);
                        /* Swap rows i and l from column k onwards */
                        cblas_sswap(n - k, &A[i + k * lda], lda, &A[l + k * lda], lda);
                    }
                    noconv = 1;

                    if (l == 0) {
                        *ilo = 0;
                        *ihi = 0;
                        return;
                    }

                    l--;
                }
            }
        }

        /* Search for columns isolating an eigenvalue and push them left */
        noconv = 1;
        while (noconv) {
            noconv = 0;
            for (j = k; j <= l; j++) {
                canswap = 1;
                for (i = k; i <= l; i++) {
                    if (i != j && A[i + j * lda] != ZERO) {
                        canswap = 0;
                        break;
                    }
                }

                if (canswap) {
                    scale[k] = (f32)j;  /* Store 0-based index */
                    if (j != k) {
                        /* Swap columns j and k */
                        cblas_sswap(l + 1, &A[0 + j * lda], 1, &A[0 + k * lda], 1);
                        /* Swap rows j and k from column k onwards */
                        cblas_sswap(n - k, &A[j + k * lda], lda, &A[k + k * lda], lda);
                    }
                    noconv = 1;

                    k++;
                }
            }
        }
    }

    /* Initialize SCALE for non-permuted submatrix */
    for (i = k; i <= l; i++) {
        scale[i] = ONE;
    }

    /* If we only had to permute, we are done */
    if (job[0] == 'P' || job[0] == 'p') {
        *ilo = k;
        *ihi = l;
        return;
    }

    /* Balance the submatrix in rows k to l */

    /* Iterative loop for norm reduction */
    sfmin1 = slamch("S") / slamch("P");
    sfmax1 = ONE / sfmin1;
    sfmin2 = sfmin1 * SCLFAC;
    sfmax2 = ONE / sfmin2;

    noconv = 1;
    while (noconv) {
        noconv = 0;

        for (i = k; i <= l; i++) {
            /* Compute column norm (rows k to l) */
            c = cblas_snrm2(l - k + 1, &A[k + i * lda], 1);
            /* Compute row norm (columns k to l) */
            r = cblas_snrm2(l - k + 1, &A[i + k * lda], lda);
            /* Find max absolute value in column (rows 0 to l) */
            ica = cblas_isamax(l + 1, &A[0 + i * lda], 1);
            ca = fabsf(A[ica + i * lda]);
            /* Find max absolute value in row (columns k to n-1) */
            ira = cblas_isamax(n - k, &A[i + k * lda], lda);
            ra = fabsf(A[i + (ira + k) * lda]);

            /* Guard against zero C or R due to underflow */
            if (c == ZERO || r == ZERO) continue;

            /* Exit if NaN to avoid infinite loop */
            if (sisnan(c + ca + r + ra)) {
                *info = -3;
                xerbla("SGEBAL", -(*info));
                return;
            }

            g = r / SCLFAC;
            f = ONE;
            s = c + r;

            while (c < g && (f > sfmax2 ? sfmax2 : f) < sfmax2 &&
                   (c > sfmax2 ? sfmax2 : c) < sfmax2 &&
                   (ca > sfmax2 ? sfmax2 : ca) < sfmax2 &&
                   (r < sfmin2 ? sfmin2 : r) > sfmin2 &&
                   (g < sfmin2 ? sfmin2 : g) > sfmin2 &&
                   (ra < sfmin2 ? sfmin2 : ra) > sfmin2) {
                /* Re-check the actual condition from Fortran */
                f32 maxfcca = f;
                if (c > maxfcca) maxfcca = c;
                if (ca > maxfcca) maxfcca = ca;
                f32 minrgra = r;
                if (g < minrgra) minrgra = g;
                if (ra < minrgra) minrgra = ra;
                if (!(c < g && maxfcca < sfmax2 && minrgra > sfmin2)) break;

                f = f * SCLFAC;
                c = c * SCLFAC;
                ca = ca * SCLFAC;
                r = r / SCLFAC;
                g = g / SCLFAC;
                ra = ra / SCLFAC;
            }

            g = c / SCLFAC;

            while (g >= r) {
                f32 maxrra = r;
                if (ra > maxrra) maxrra = ra;
                f32 minfcgca = f;
                if (c < minfcgca) minfcgca = c;
                if (g < minfcgca) minfcgca = g;
                if (ca < minfcgca) minfcgca = ca;
                if (!(maxrra < sfmax2 && minfcgca > sfmin2)) break;

                f = f / SCLFAC;
                c = c / SCLFAC;
                g = g / SCLFAC;
                ca = ca / SCLFAC;
                r = r * SCLFAC;
                ra = ra * SCLFAC;
            }

            /* Now balance */
            if ((c + r) >= FACTOR * s) continue;
            if (f < ONE && scale[i] < ONE) {
                if (f * scale[i] <= sfmin1) continue;
            }
            if (f > ONE && scale[i] > ONE) {
                if (scale[i] >= sfmax1 / f) continue;
            }
            g = ONE / f;
            scale[i] = scale[i] * f;
            noconv = 1;

            /* Scale row i */
            cblas_sscal(n - k, g, &A[i + k * lda], lda);
            /* Scale column i */
            cblas_sscal(l + 1, f, &A[0 + i * lda], 1);
        }
    }

    *ilo = k;
    *ihi = l;
}
