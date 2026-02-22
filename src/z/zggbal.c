/**
 * @file zggbal.c
 * @brief ZGGBAL balances a pair of general complex matrices (A,B).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGGBAL balances a pair of general complex matrices (A,B). This
 * involves, first, permuting A and B by similarity transformations to
 * isolate eigenvalues in the first 0 to ILO-1 and last IHI+1 to N-1
 * elements on the diagonal; and second, applying a diagonal similarity
 * transformation to rows and columns ILO to IHI to make the rows
 * and columns as close in norm as possible. Both steps are optional.
 *
 * Balancing may reduce the 1-norm of the matrices, and improve the
 * accuracy of the computed eigenvalues and/or eigenvectors in the
 * generalized eigenvalue problem A*x = lambda*B*x.
 *
 * @param[in]     job     Specifies the operations to be performed on A and B:
 *                        = 'N': none: simply set ILO = 0, IHI = N-1, LSCALE(I) = 1.0
 *                               and RSCALE(I) = 1.0 for i = 0,...,N-1.
 *                        = 'P': permute only;
 *                        = 'S': scale only;
 *                        = 'B': both permute and scale.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Complex array of dimension (lda, n). On entry, the input matrix A.
 *                        On exit, A is overwritten by the balanced matrix.
 *                        If job = 'N', A is not referenced.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Complex array of dimension (ldb, n). On entry, the input matrix B.
 *                        On exit, B is overwritten by the balanced matrix.
 *                        If job = 'N', B is not referenced.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    ilo     See ihi.
 * @param[out]    ihi     ILO and IHI are set to integers such that on exit
 *                        A(i,j) = 0 and B(i,j) = 0 if i > j and
 *                        j = 0,...,ILO-1 or i = IHI+1,...,N-1.
 *                        If job = 'N' or 'S', ILO = 0 and IHI = N-1.
 * @param[out]    lscale  Array of dimension (n). Details of the permutations and
 *                        scaling factors applied to the left side of A and B.
 *                        If P(j) is the index of the row interchanged with row j,
 *                        and D(j) is the scaling factor applied to row j, then
 *                          lscale[j] = P(j)    for j = 0,...,ILO-1
 *                                    = D(j)    for j = ILO,...,IHI
 *                                    = P(j)    for j = IHI+1,...,N-1.
 *                        The order in which the interchanges are made is N-1 to IHI+1,
 *                        then 0 to ILO-1.
 * @param[out]    rscale  Array of dimension (n). Details of the permutations and
 *                        scaling factors applied to the right side of A and B.
 *                        If P(j) is the index of the column interchanged with column j,
 *                        and D(j) is the scaling factor applied to column j, then
 *                          rscale[j] = P(j)    for j = 0,...,ILO-1
 *                                    = D(j)    for j = ILO,...,IHI
 *                                    = P(j)    for j = IHI+1,...,N-1.
 *                        The order in which the interchanges are made is N-1 to IHI+1,
 *                        then 0 to ILO-1.
 * @param[out]    work    Real workspace array of dimension (lwork).
 *                        lwork must be at least max(1, 6*n) when job = 'S' or 'B', and
 *                        at least 1 when job = 'N' or 'P'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zggbal(
    const char* job,
    const INT n,
    c128* restrict A,
    const INT lda,
    c128* restrict B,
    const INT ldb,
    INT* ilo,
    INT* ihi,
    f64* restrict lscale,
    f64* restrict rscale,
    f64* restrict work,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 THREE = 3.0;
    const f64 SCLFAC = 10.0;
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT i, icab, ir, irab, it, j, jc;
    INT k, kount, l, lcab, lrab, lsfmax, lsfmin;
    INT m, nr, nrp2;
    f64 alpha, basl, beta, cab, cmax, coef, coef2;
    f64 coef5, cor, ew, ewc, gamma, pgamma, rab, sfmax;
    f64 sfmin, sum, t, ta, tb, tc;

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
        *ilo = 0;
        *ihi = -1;
        return;
    }

    if (n == 1) {
        *ilo = 0;
        *ihi = 0;
        lscale[0] = ONE;
        rscale[0] = ONE;
        return;
    }

    if (job[0] == 'N' || job[0] == 'n') {
        *ilo = 0;
        *ihi = n - 1;
        for (i = 0; i < n; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        return;
    }

    k = 0;
    l = n - 1;

    if (!(job[0] == 'S' || job[0] == 's')) {

        /* Permute the matrices A and B to isolate the eigenvalues. */

        /* Find row with one nonzero in columns 1 through L */

        INT done_permuting = 0;
        for (;;) {
            if (l == 0) {
                rscale[0] = ONE;
                lscale[0] = ONE;
                done_permuting = 1;
                break;
            }
            INT found = 0;
            for (i = l; i >= 0; i--) {
                INT nz_col = -1;
                INT isolated = 1;
                for (j = 0; j <= l; j++) {
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
                j = (nz_col >= 0) ? nz_col : l;

                /* Permute rows M and I */

                m = l;
                lscale[m] = (f64)i;
                if (i != m) {
                    cblas_zswap(n - k, &A[i + k * lda], lda, &A[m + k * lda], lda);
                    cblas_zswap(n - k, &B[i + k * ldb], ldb, &B[m + k * ldb], ldb);
                }

                /* Permute columns M and J */

                rscale[m] = (f64)j;
                if (j != m) {
                    cblas_zswap(l + 1, &A[j * lda], 1, &A[m * lda], 1);
                    cblas_zswap(l + 1, &B[j * ldb], 1, &B[m * ldb], 1);
                }
                l--;
                found = 1;
                break;
            }
            if (!found)
                break;
        }

        /* Find column with one nonzero in rows K through N */

        if (!done_permuting) {
            for (;;) {
                INT found = 0;
                for (j = k; j <= l; j++) {
                    INT nz_row = -1;
                    INT isolated = 1;
                    for (i = k; i <= l; i++) {
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
                    i = (nz_row >= 0) ? nz_row : l;

                    /* Permute rows M and I */

                    m = k;
                    lscale[m] = (f64)i;
                    if (i != m) {
                        cblas_zswap(n - k, &A[i + k * lda], lda, &A[m + k * lda], lda);
                        cblas_zswap(n - k, &B[i + k * ldb], ldb, &B[m + k * ldb], ldb);
                    }

                    /* Permute columns M and J */

                    rscale[m] = (f64)j;
                    if (j != m) {
                        cblas_zswap(l + 1, &A[j * lda], 1, &A[m * lda], 1);
                        cblas_zswap(l + 1, &B[j * ldb], 1, &B[m * ldb], 1);
                    }
                    k++;
                    found = 1;
                    break;
                }
                if (!found)
                    break;
            }
        }
    }

    *ilo = k;
    *ihi = l;

    if (job[0] == 'P' || job[0] == 'p') {
        for (i = k; i <= l; i++) {
            lscale[i] = ONE;
            rscale[i] = ONE;
        }
        return;
    }

    if (k == l)
        return;

    /* Balance the submatrix in rows ILO to IHI. */

    nr = l - k + 1;
    for (i = k; i <= l; i++) {
        rscale[i] = ZERO;
        lscale[i] = ZERO;

        work[i] = ZERO;
        work[i + n] = ZERO;
        work[i + 2 * n] = ZERO;
        work[i + 3 * n] = ZERO;
        work[i + 4 * n] = ZERO;
        work[i + 5 * n] = ZERO;
    }

    /* Compute right side vector in resulting linear equations */

    basl = log10(SCLFAC);
    for (i = k; i <= l; i++) {
        for (j = k; j <= l; j++) {
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

    coef = ONE / (f64)(2 * nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    /* Start generalized conjugate gradient iteration */

    for (it = 1; it <= nrp2; it++) {

        gamma = cblas_ddot(nr, &work[k + 4 * n], 1, &work[k + 4 * n], 1) +
                cblas_ddot(nr, &work[k + 5 * n], 1, &work[k + 5 * n], 1);

        ew = ZERO;
        ewc = ZERO;
        for (i = k; i <= l; i++) {
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

        cblas_dscal(nr, beta, &work[k], 1);
        cblas_dscal(nr, beta, &work[k + n], 1);

        cblas_daxpy(nr, coef, &work[k + 4 * n], 1, &work[k + n], 1);
        cblas_daxpy(nr, coef, &work[k + 5 * n], 1, &work[k], 1);

        for (i = k; i <= l; i++) {
            work[i] = work[i] + tc;
            work[i + n] = work[i + n] + t;
        }

        /* Apply matrix to vector */

        for (i = k; i <= l; i++) {
            kount = 0;
            sum = ZERO;
            for (j = k; j <= l; j++) {
                if (A[i + j * lda] != CZERO) {
                    kount = kount + 1;
                    sum = sum + work[j];
                }
                if (B[i + j * ldb] != CZERO) {
                    kount = kount + 1;
                    sum = sum + work[j];
                }
            }
            work[i + 2 * n] = (f64)kount * work[i + n] + sum;
        }

        for (j = k; j <= l; j++) {
            kount = 0;
            sum = ZERO;
            for (i = k; i <= l; i++) {
                if (A[i + j * lda] != CZERO) {
                    kount = kount + 1;
                    sum = sum + work[i + n];
                }
                if (B[i + j * ldb] != CZERO) {
                    kount = kount + 1;
                    sum = sum + work[i + n];
                }
            }
            work[j + 3 * n] = (f64)kount * work[j] + sum;
        }

        sum = cblas_ddot(nr, &work[k + n], 1, &work[k + 2 * n], 1) +
              cblas_ddot(nr, &work[k], 1, &work[k + 3 * n], 1);
        alpha = gamma / sum;

        /* Determine correction to current iteration */

        cmax = ZERO;
        for (i = k; i <= l; i++) {
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

        cblas_daxpy(nr, -alpha, &work[k + 2 * n], 1, &work[k + 4 * n], 1);
        cblas_daxpy(nr, -alpha, &work[k + 3 * n], 1, &work[k + 5 * n], 1);

        pgamma = gamma;
    }

    /* End generalized conjugate gradient iteration */

    sfmin = dlamch("S");
    sfmax = ONE / sfmin;
    lsfmin = (INT)(log10(sfmin) / basl + ONE);
    lsfmax = (INT)(log10(sfmax) / basl);
    for (i = k; i <= l; i++) {
        irab = cblas_izamax(n - k, &A[i + k * lda], lda);
        rab = cabs(A[i + (irab + k) * lda]);
        irab = cblas_izamax(n - k, &B[i + k * ldb], ldb);
        rab = (rab > cabs(B[i + (irab + k) * ldb])) ? rab : cabs(B[i + (irab + k) * ldb]);
        lrab = (INT)(log10(rab + sfmin) / basl + ONE);
        ir = (INT)(lscale[i] + (lscale[i] >= 0 ? HALF : -HALF));
        ir = (ir > lsfmin) ? ir : lsfmin;
        ir = (ir < lsfmax) ? ir : lsfmax;
        ir = (ir < lsfmax - lrab) ? ir : lsfmax - lrab;
        lscale[i] = pow(SCLFAC, (f64)ir);
        icab = cblas_izamax(l + 1, &A[i * lda], 1);
        cab = cabs(A[icab + i * lda]);
        icab = cblas_izamax(l + 1, &B[i * ldb], 1);
        cab = (cab > cabs(B[icab + i * ldb])) ? cab : cabs(B[icab + i * ldb]);
        lcab = (INT)(log10(cab + sfmin) / basl + ONE);
        jc = (INT)(rscale[i] + (rscale[i] >= 0 ? HALF : -HALF));
        jc = (jc > lsfmin) ? jc : lsfmin;
        jc = (jc < lsfmax) ? jc : lsfmax;
        jc = (jc < lsfmax - lcab) ? jc : lsfmax - lcab;
        rscale[i] = pow(SCLFAC, (f64)jc);
    }

    /* Row scaling of matrices A and B */

    for (i = k; i <= l; i++) {
        cblas_zdscal(n - k, lscale[i], &A[i + k * lda], lda);
        cblas_zdscal(n - k, lscale[i], &B[i + k * ldb], ldb);
    }

    /* Column scaling of matrices A and B */

    for (j = k; j <= l; j++) {
        cblas_zdscal(l + 1, rscale[j], &A[j * lda], 1);
        cblas_zdscal(l + 1, rscale[j], &B[j * ldb], 1);
    }
}
