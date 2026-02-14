/**
 * @file zggbal.c
 * @brief ZGGBAL balances a pair of general complex matrices (A,B).
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGGBAL balances a pair of general complex matrices (A,B). This
 * involves, first, permuting A and B by similarity transformations to
 * isolate eigenvalues in the first 1 to ILO-1 and last IHI+1 to N
 * elements on the diagonal; and second, applying a diagonal similarity
 * transformation to rows and columns ILO to IHI to make the rows
 * and columns as close in norm as possible. Both steps are optional.
 *
 * Balancing may reduce the 1-norm of the matrices, and improve the
 * accuracy of the computed eigenvalues and/or eigenvectors in the
 * generalized eigenvalue problem A*x = lambda*B*x.
 *
 * @param[in]     job     Specifies the operations to be performed on A and B:
 *                        = 'N': none: simply set ILO = 1, IHI = N, LSCALE(I) = 1.0
 *                               and RSCALE(I) = 1.0 for i = 1,...,N.
 *                        = 'P': permute only;
 *                        = 'S': scale only;
 *                        = 'B': both permute and scale.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Complex array of dimension (lda, n). On entry, the input matrix A.
 *                        On exit, A is overwritten by the balanced matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Complex array of dimension (ldb, n). On entry, the input matrix B.
 *                        On exit, B is overwritten by the balanced matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    ilo     See ihi.
 * @param[out]    ihi     ILO and IHI are set to integers such that on exit
 *                        A(i,j) = 0 and B(i,j) = 0 if i > j and
 *                        j = 1,...,ILO-1 or i = IHI+1,...,N.
 * @param[out]    lscale  Array of dimension (n). Details of permutations and
 *                        scaling factors applied to the left side of A and B.
 * @param[out]    rscale  Array of dimension (n). Details of permutations and
 *                        scaling factors applied to the right side of A and B.
 * @param[out]    work    Real workspace array of dimension (6*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zggbal(
    const char* job,
    const int n,
    double complex* const restrict A,
    const int lda,
    double complex* const restrict B,
    const int ldb,
    int* ilo,
    int* ihi,
    double* const restrict lscale,
    double* const restrict rscale,
    double* const restrict work,
    int* info)
{
    const double ZERO = 0.0;
    const double HALF = 0.5;
    const double ONE = 1.0;
    const double THREE = 3.0;
    const double SCLFAC = 10.0;
    const double complex CZERO = CMPLX(0.0, 0.0);

    int i, icab, ir, irab, it, j, jc;
    int k, kount, l, lcab, lrab, lsfmax, lsfmin;
    int m, nr, nrp2;
    double alpha, basl, beta, cab, cmax, coef, coef2;
    double coef5, cor, ew, ewc, gamma, pgamma, rab, sfmax;
    double sfmin, sum, t, ta, tb, tc;

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
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("ZGGBAL", -(*info));
        return;
    }

    if (n == 0) {
        *ilo = 1;
        *ihi = n;
        return;
    }

    if (n == 1) {
        *ilo = 1;
        *ihi = n;
        lscale[0] = ONE;
        rscale[0] = ONE;
        return;
    }

    if (job[0] == 'N' || job[0] == 'n') {
        *ilo = 1;
        *ihi = n;
        for (i = 0; i < n; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        return;
    }

    k = 1;
    l = n;

    if (!(job[0] == 'S' || job[0] == 's')) {

        /*
         * Permute the matrices A and B to isolate the eigenvalues.
         *
         * Row isolation: find rows with one nonzero in columns 0..l-1,
         * deflating from the bottom (decreasing l).
         */
        for (;;) {
            if (l == 1) {
                rscale[0] = ONE;
                lscale[0] = ONE;
                goto scaling;
            }
            int found = 0;
            for (i = l - 1; i >= 0; i--) {
                int nz_col = -1;
                int isolated = 1;
                for (j = 0; j < l; j++) {
                    if (A[i + j * lda] != CZERO || B[i + j * ldb] != CZERO) {
                        if (nz_col >= 0) {
                            isolated = 0;
                            break;
                        }
                        nz_col = j;
                    }
                }
                if (!isolated)
                    continue;
                j = (nz_col >= 0) ? nz_col : l - 1;
                m = l;
                lscale[m - 1] = (double)(i + 1);
                if (i != m - 1) {
                    cblas_zswap(n - k + 1, &A[i + (k - 1) * lda], lda,
                                &A[m - 1 + (k - 1) * lda], lda);
                    cblas_zswap(n - k + 1, &B[i + (k - 1) * ldb], ldb,
                                &B[m - 1 + (k - 1) * ldb], ldb);
                }
                rscale[m - 1] = (double)(j + 1);
                if (j != m - 1) {
                    cblas_zswap(l, &A[j * lda], 1, &A[(m - 1) * lda], 1);
                    cblas_zswap(l, &B[j * ldb], 1, &B[(m - 1) * ldb], 1);
                }
                l--;
                found = 1;
                break;
            }
            if (!found)
                break;
        }

        /*
         * Column isolation: find columns with one nonzero in rows k-1..l-1,
         * deflating from the top (increasing k).
         */
        for (;;) {
            int found = 0;
            for (j = k - 1; j < l; j++) {
                int nz_row = -1;
                int isolated = 1;
                for (i = k - 1; i < l; i++) {
                    if (A[i + j * lda] != CZERO || B[i + j * ldb] != CZERO) {
                        if (nz_row >= 0) {
                            isolated = 0;
                            break;
                        }
                        nz_row = i;
                    }
                }
                if (!isolated)
                    continue;
                i = (nz_row >= 0) ? nz_row : l - 1;
                m = k;
                lscale[m - 1] = (double)(i + 1);
                if (i != m - 1) {
                    cblas_zswap(n - k + 1, &A[i + (k - 1) * lda], lda,
                                &A[m - 1 + (k - 1) * lda], lda);
                    cblas_zswap(n - k + 1, &B[i + (k - 1) * ldb], ldb,
                                &B[m - 1 + (k - 1) * ldb], ldb);
                }
                rscale[m - 1] = (double)(j + 1);
                if (j != m - 1) {
                    cblas_zswap(l, &A[j * lda], 1, &A[(m - 1) * lda], 1);
                    cblas_zswap(l, &B[j * ldb], 1, &B[(m - 1) * ldb], 1);
                }
                k++;
                found = 1;
                break;
            }
            if (!found)
                break;
        }
    }

scaling:
    *ilo = k;
    *ihi = l;

    if (job[0] == 'P' || job[0] == 'p') {
        for (i = k - 1; i < l; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        return;
    }

    if (k == l)
        return;

    /*
     * Balance the submatrix in rows ILO to IHI.
     */
    nr = l - k + 1;
    for (i = k - 1; i < l; i++) {
        rscale[i] = ZERO;
        lscale[i] = ZERO;

        work[i] = ZERO;
        work[i + n] = ZERO;
        work[i + 2 * n] = ZERO;
        work[i + 3 * n] = ZERO;
        work[i + 4 * n] = ZERO;
        work[i + 5 * n] = ZERO;
    }

    /*
     * Compute right side vector in resulting linear equations
     */
    basl = log10(SCLFAC);
    for (i = k - 1; i < l; i++) {
        for (j = k - 1; j < l; j++) {
            if (A[i + j * lda] == CZERO)
                ta = ZERO;
            else
                ta = log10(cabs1(A[i + j * lda])) / basl;

            if (B[i + j * ldb] == CZERO)
                tb = ZERO;
            else
                tb = log10(cabs1(B[i + j * ldb])) / basl;

            work[i + 4 * n] = work[i + 4 * n] - ta - tb;
            work[j + 5 * n] = work[j + 5 * n] - ta - tb;
        }
    }

    coef = ONE / (double)(2 * nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    /*
     * Generalized conjugate gradient iteration
     */
    for (it = 1; it <= nrp2; it++) {

        gamma = cblas_ddot(nr, &work[k - 1 + 4 * n], 1, &work[k - 1 + 4 * n], 1) +
                cblas_ddot(nr, &work[k - 1 + 5 * n], 1, &work[k - 1 + 5 * n], 1);

        ew = ZERO;
        ewc = ZERO;
        for (i = k - 1; i < l; i++) {
            ew = ew + work[i + 4 * n];
            ewc = ewc + work[i + 5 * n];
        }

        gamma = coef * gamma - coef2 * (ew * ew + ewc * ewc) -
                coef5 * (ew - ewc) * (ew - ewc);
        if (gamma == ZERO)
            break;
        if (it != 1)
            beta = gamma / pgamma;
        t = coef5 * (ewc - THREE * ew);
        tc = coef5 * (ew - THREE * ewc);

        cblas_dscal(nr, beta, &work[k - 1], 1);
        cblas_dscal(nr, beta, &work[k - 1 + n], 1);

        cblas_daxpy(nr, coef, &work[k - 1 + 4 * n], 1, &work[k - 1 + n], 1);
        cblas_daxpy(nr, coef, &work[k - 1 + 5 * n], 1, &work[k - 1], 1);

        for (i = k - 1; i < l; i++) {
            work[i] = work[i] + tc;
            work[i + n] = work[i + n] + t;
        }

        /* Apply matrix to vector */

        for (i = k - 1; i < l; i++) {
            kount = 0;
            sum = ZERO;
            for (j = k - 1; j < l; j++) {
                if (A[i + j * lda] != CZERO) {
                    kount++;
                    sum += work[j];
                }
                if (B[i + j * ldb] != CZERO) {
                    kount++;
                    sum += work[j];
                }
            }
            work[i + 2 * n] = (double)kount * work[i + n] + sum;
        }

        for (j = k - 1; j < l; j++) {
            kount = 0;
            sum = ZERO;
            for (i = k - 1; i < l; i++) {
                if (A[i + j * lda] != CZERO) {
                    kount++;
                    sum += work[i + n];
                }
                if (B[i + j * ldb] != CZERO) {
                    kount++;
                    sum += work[i + n];
                }
            }
            work[j + 3 * n] = (double)kount * work[j] + sum;
        }

        sum = cblas_ddot(nr, &work[k - 1 + n], 1, &work[k - 1 + 2 * n], 1) +
              cblas_ddot(nr, &work[k - 1], 1, &work[k - 1 + 3 * n], 1);
        alpha = gamma / sum;

        /* Determine correction to current iteration */

        cmax = ZERO;
        for (i = k - 1; i < l; i++) {
            cor = alpha * work[i + n];
            if (fabs(cor) > cmax)
                cmax = fabs(cor);
            lscale[i] = lscale[i] + cor;
            cor = alpha * work[i];
            if (fabs(cor) > cmax)
                cmax = fabs(cor);
            rscale[i] = rscale[i] + cor;
        }
        if (cmax < HALF)
            break;

        cblas_daxpy(nr, -alpha, &work[k - 1 + 2 * n], 1, &work[k - 1 + 4 * n], 1);
        cblas_daxpy(nr, -alpha, &work[k - 1 + 3 * n], 1, &work[k - 1 + 5 * n], 1);

        pgamma = gamma;
    }

    /* End generalized conjugate gradient iteration */

    sfmin = dlamch("S");
    sfmax = ONE / sfmin;
    lsfmin = (int)(log10(sfmin) / basl + ONE);
    lsfmax = (int)(log10(sfmax) / basl);
    for (i = k - 1; i < l; i++) {
        irab = cblas_izamax(n - k + 1, &A[i + (k - 1) * lda], lda);
        rab = cabs(A[i + (irab + k - 1) * lda]);
        irab = cblas_izamax(n - k + 1, &B[i + (k - 1) * ldb], ldb);
        rab = (rab > cabs(B[i + (irab + k - 1) * ldb])) ?
              rab : cabs(B[i + (irab + k - 1) * ldb]);
        lrab = (int)(log10(rab + sfmin) / basl + ONE);
        ir = (int)(lscale[i] + (lscale[i] >= 0 ? HALF : -HALF));
        ir = (ir > lsfmin) ? ir : lsfmin;
        ir = (ir < lsfmax) ? ir : lsfmax;
        ir = (ir < lsfmax - lrab) ? ir : lsfmax - lrab;
        lscale[i] = pow(SCLFAC, (double)ir);
        icab = cblas_izamax(l, &A[i * lda], 1);
        cab = cabs(A[icab + i * lda]);
        icab = cblas_izamax(l, &B[i * ldb], 1);
        cab = (cab > cabs(B[icab + i * ldb])) ?
              cab : cabs(B[icab + i * ldb]);
        lcab = (int)(log10(cab + sfmin) / basl + ONE);
        jc = (int)(rscale[i] + (rscale[i] >= 0 ? HALF : -HALF));
        jc = (jc > lsfmin) ? jc : lsfmin;
        jc = (jc < lsfmax) ? jc : lsfmax;
        jc = (jc < lsfmax - lcab) ? jc : lsfmax - lcab;
        rscale[i] = pow(SCLFAC, (double)jc);
    }

    /* Row scaling of matrices A and B */

    for (i = k - 1; i < l; i++) {
        cblas_zdscal(n - k + 1, lscale[i], &A[i + (k - 1) * lda], lda);
        cblas_zdscal(n - k + 1, lscale[i], &B[i + (k - 1) * ldb], ldb);
    }

    /* Column scaling of matrices A and B */

    for (j = k - 1; j < l; j++) {
        cblas_zdscal(l, rscale[j], &A[j * lda], 1);
        cblas_zdscal(l, rscale[j], &B[j * ldb], 1);
    }
}
