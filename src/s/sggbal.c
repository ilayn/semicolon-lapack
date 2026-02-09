/**
 * @file sggbal.c
 * @brief SGGBAL balances a pair of general real matrices (A,B).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGGBAL balances a pair of general real matrices (A,B). This
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
 * @param[in,out] A       Array of dimension (lda, n). On entry, the input matrix A.
 *                        On exit, A is overwritten by the balanced matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Array of dimension (ldb, n). On entry, the input matrix B.
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
 * @param[out]    work    Workspace array of dimension (6*n).
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void sggbal(
    const char* job,
    const int n,
    float* const restrict A,
    const int lda,
    float* const restrict B,
    const int ldb,
    int* ilo,
    int* ihi,
    float* const restrict lscale,
    float* const restrict rscale,
    float* const restrict work,
    int* info)
{
    const float ZERO = 0.0f;
    const float HALF = 0.5f;
    const float ONE = 1.0f;
    const float THREE = 3.0f;
    const float SCLFAC = 10.0f;

    int i, icab, iflow, ip1, ir, irab, it, j, jc, jp1;
    int k, kount, l, lcab, lm1, lrab, lsfmax, lsfmin;
    int m, nr, nrp2;
    float alpha, basl, beta, cab, cmax, coef, coef2;
    float coef5, cor, ew, ewc, gamma, pgamma, rab, sfmax;
    float sfmin, sum, t, ta, tb, tc;

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
        xerbla("SGGBAL", -(*info));
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
    if (job[0] == 'S' || job[0] == 's')
        goto L190;

    goto L30;

L20:
    l = lm1;
    if (l != 1)
        goto L30;

    rscale[0] = ONE;
    lscale[0] = ONE;
    goto L190;

L30:
    lm1 = l - 1;
    for (i = l - 1; i >= 0; i--) {
        for (j = 0; j < lm1; j++) {
            jp1 = j + 1;
            if (A[i + j * lda] != ZERO || B[i + j * ldb] != ZERO)
                goto L50;
        }
        j = l - 1;
        goto L70;

    L50:
        for (j = jp1; j < l; j++) {
            if (A[i + j * lda] != ZERO || B[i + j * ldb] != ZERO)
                goto L80;
        }
        j = jp1 - 1;

    L70:
        m = l;
        iflow = 1;
        goto L160;
    L80:;
    }
    goto L100;

L90:
    k = k + 1;

L100:
    for (j = k - 1; j < l; j++) {
        for (i = k - 1; i < lm1; i++) {
            ip1 = i + 1;
            if (A[i + j * lda] != ZERO || B[i + j * ldb] != ZERO)
                goto L120;
        }
        i = l - 1;
        goto L140;
    L120:
        for (i = ip1; i < l; i++) {
            if (A[i + j * lda] != ZERO || B[i + j * ldb] != ZERO)
                goto L150;
        }
        i = ip1 - 1;
    L140:
        m = k;
        iflow = 2;
        goto L160;
    L150:;
    }
    goto L190;

L160:
    lscale[m - 1] = (float)(i + 1);
    if (i != m - 1) {
        cblas_sswap(n - k + 1, &A[i + (k - 1) * lda], lda, &A[m - 1 + (k - 1) * lda], lda);
        cblas_sswap(n - k + 1, &B[i + (k - 1) * ldb], ldb, &B[m - 1 + (k - 1) * ldb], ldb);
    }

    rscale[m - 1] = (float)(j + 1);
    if (j != m - 1) {
        cblas_sswap(l, &A[j * lda], 1, &A[(m - 1) * lda], 1);
        cblas_sswap(l, &B[j * ldb], 1, &B[(m - 1) * ldb], 1);
    }

    if (iflow == 1)
        goto L20;
    else
        goto L90;

L190:
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

    basl = log10f(SCLFAC);
    for (i = k - 1; i < l; i++) {
        for (j = k - 1; j < l; j++) {
            tb = B[i + j * ldb];
            ta = A[i + j * lda];
            if (ta == ZERO)
                goto L210;
            ta = log10f(fabsf(ta)) / basl;
        L210:
            if (tb == ZERO)
                goto L220;
            tb = log10f(fabsf(tb)) / basl;
        L220:
            work[i + 4 * n] = work[i + 4 * n] - ta - tb;
            work[j + 5 * n] = work[j + 5 * n] - ta - tb;
        }
    }

    coef = ONE / (float)(2 * nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;
    it = 1;

L250:
    gamma = cblas_sdot(nr, &work[k - 1 + 4 * n], 1, &work[k - 1 + 4 * n], 1) +
            cblas_sdot(nr, &work[k - 1 + 5 * n], 1, &work[k - 1 + 5 * n], 1);

    ew = ZERO;
    ewc = ZERO;
    for (i = k - 1; i < l; i++) {
        ew = ew + work[i + 4 * n];
        ewc = ewc + work[i + 5 * n];
    }

    gamma = coef * gamma - coef2 * (ew * ew + ewc * ewc) - coef5 * (ew - ewc) * (ew - ewc);
    if (gamma == ZERO)
        goto L350;
    if (it != 1)
        beta = gamma / pgamma;
    t = coef5 * (ewc - THREE * ew);
    tc = coef5 * (ew - THREE * ewc);

    cblas_sscal(nr, beta, &work[k - 1], 1);
    cblas_sscal(nr, beta, &work[k - 1 + n], 1);

    cblas_saxpy(nr, coef, &work[k - 1 + 4 * n], 1, &work[k - 1 + n], 1);
    cblas_saxpy(nr, coef, &work[k - 1 + 5 * n], 1, &work[k - 1], 1);

    for (i = k - 1; i < l; i++) {
        work[i] = work[i] + tc;
        work[i + n] = work[i + n] + t;
    }

    for (i = k - 1; i < l; i++) {
        kount = 0;
        sum = ZERO;
        for (j = k - 1; j < l; j++) {
            if (A[i + j * lda] == ZERO)
                goto L280;
            kount = kount + 1;
            sum = sum + work[j];
        L280:
            if (B[i + j * ldb] == ZERO)
                continue;
            kount = kount + 1;
            sum = sum + work[j];
        }
        work[i + 2 * n] = (float)kount * work[i + n] + sum;
    }

    for (j = k - 1; j < l; j++) {
        kount = 0;
        sum = ZERO;
        for (i = k - 1; i < l; i++) {
            if (A[i + j * lda] == ZERO)
                goto L310;
            kount = kount + 1;
            sum = sum + work[i + n];
        L310:
            if (B[i + j * ldb] == ZERO)
                continue;
            kount = kount + 1;
            sum = sum + work[i + n];
        }
        work[j + 3 * n] = (float)kount * work[j] + sum;
    }

    sum = cblas_sdot(nr, &work[k - 1 + n], 1, &work[k - 1 + 2 * n], 1) +
          cblas_sdot(nr, &work[k - 1], 1, &work[k - 1 + 3 * n], 1);
    alpha = gamma / sum;

    cmax = ZERO;
    for (i = k - 1; i < l; i++) {
        cor = alpha * work[i + n];
        if (fabsf(cor) > cmax)
            cmax = fabsf(cor);
        lscale[i] = lscale[i] + cor;
        cor = alpha * work[i];
        if (fabsf(cor) > cmax)
            cmax = fabsf(cor);
        rscale[i] = rscale[i] + cor;
    }
    if (cmax < HALF)
        goto L350;

    cblas_saxpy(nr, -alpha, &work[k - 1 + 2 * n], 1, &work[k - 1 + 4 * n], 1);
    cblas_saxpy(nr, -alpha, &work[k - 1 + 3 * n], 1, &work[k - 1 + 5 * n], 1);

    pgamma = gamma;
    it = it + 1;
    if (it <= nrp2)
        goto L250;

L350:
    sfmin = slamch("S");
    sfmax = ONE / sfmin;
    lsfmin = (int)(log10f(sfmin) / basl + ONE);
    lsfmax = (int)(log10f(sfmax) / basl);
    for (i = k - 1; i < l; i++) {
        irab = cblas_isamax(n - k + 1, &A[i + (k - 1) * lda], lda);
        rab = fabsf(A[i + (irab + k - 1) * lda]);
        irab = cblas_isamax(n - k + 1, &B[i + (k - 1) * ldb], ldb);
        rab = (rab > fabsf(B[i + (irab + k - 1) * ldb])) ? rab : fabsf(B[i + (irab + k - 1) * ldb]);
        lrab = (int)(log10f(rab + sfmin) / basl + ONE);
        ir = (int)(lscale[i] + (lscale[i] >= 0 ? HALF : -HALF));
        ir = (ir > lsfmin) ? ir : lsfmin;
        ir = (ir < lsfmax) ? ir : lsfmax;
        ir = (ir < lsfmax - lrab) ? ir : lsfmax - lrab;
        lscale[i] = powf(SCLFAC, (float)ir);
        icab = cblas_isamax(l, &A[i * lda], 1);
        cab = fabsf(A[icab + i * lda]);
        icab = cblas_isamax(l, &B[i * ldb], 1);
        cab = (cab > fabsf(B[icab + i * ldb])) ? cab : fabsf(B[icab + i * ldb]);
        lcab = (int)(log10f(cab + sfmin) / basl + ONE);
        jc = (int)(rscale[i] + (rscale[i] >= 0 ? HALF : -HALF));
        jc = (jc > lsfmin) ? jc : lsfmin;
        jc = (jc < lsfmax) ? jc : lsfmax;
        jc = (jc < lsfmax - lcab) ? jc : lsfmax - lcab;
        rscale[i] = powf(SCLFAC, (float)jc);
    }

    for (i = k - 1; i < l; i++) {
        cblas_sscal(n - k + 1, lscale[i], &A[i + (k - 1) * lda], lda);
        cblas_sscal(n - k + 1, lscale[i], &B[i + (k - 1) * ldb], ldb);
    }

    for (j = k - 1; j < l; j++) {
        cblas_sscal(l, rscale[j], &A[j * lda], 1);
        cblas_sscal(l, rscale[j], &B[j * ldb], 1);
    }
}
