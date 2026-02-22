/**
 * @file ztgsen.c
 * @brief ZTGSEN reorders the generalized Schur decomposition of a complex matrix pair.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZTGSEN reorders the generalized Schur decomposition of a complex
 * matrix pair (A, B) (in terms of an unitary equivalence trans-
 * formation Q**H * (A, B) * Z), so that a selected cluster of eigenvalues
 * appears in the leading diagonal blocks of the pair (A,B). The leading
 * columns of Q and Z form unitary bases of the corresponding left and
 * right eigenspaces (deflating subspaces). (A, B) must be in
 * generalized Schur canonical form, that is, A and B are both upper
 * triangular.
 *
 * ZTGSEN also computes the generalized eigenvalues
 *
 *          w(j)= ALPHA(j) / BETA(j)
 *
 * of the reordered matrix pair (A, B).
 *
 * Optionally, the routine computes estimates of reciprocal condition
 * numbers for eigenvalues and eigenspaces.
 *
 * @param[in]     ijob    Specifies whether condition numbers are required.
 *                        = 0: Only reorder w.r.t. SELECT. No extras.
 *                        = 1: Reciprocal of norms of "projections" (PL and PR).
 *                        = 2: Upper bounds on Difu and Difl (F-norm-based).
 *                        = 3: Estimate of Difu and Difl (1-norm-based).
 *                        = 4: Compute PL, PR and DIF (i.e. 0, 1 and 2 above).
 *                        = 5: Compute PL, PR and DIF (i.e. 0, 1 and 3 above).
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     select  Integer array of dimension (n). Specifies the eigenvalues
 *                        in the selected cluster.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Complex array of dimension (lda, n). Upper triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Complex array of dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    alpha   Complex array of dimension (n). Diagonal elements of A.
 * @param[out]    beta    Complex array of dimension (n). Diagonal elements of B.
 * @param[in,out] Q       Complex array of dimension (ldq, n). The unitary matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Complex array of dimension (ldz, n). The unitary matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[out]    m       The dimension of the specified pair of deflating subspaces.
 * @param[out]    pl      Lower bound on reciprocal of projection onto left eigenspace.
 * @param[out]    pr      Lower bound on reciprocal of projection onto right eigenspace.
 * @param[out]    dif     Array of dimension (2). Estimates of Difu and Difl.
 * @param[out]    work    Complex array of dimension (max(1, lwork)).
 * @param[in]     lwork   The dimension of work.
 * @param[out]    iwork   Integer array of dimension (max(1, liwork)).
 * @param[in]     liwork  The dimension of iwork.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: reordering failed
 */
void ztgsen(
    const INT ijob,
    const INT wantq,
    const INT wantz,
    const INT* restrict select,
    const INT n,
    c128* A,
    const INT lda,
    c128* B,
    const INT ldb,
    c128* alpha,
    c128* beta,
    c128* Q,
    const INT ldq,
    c128* Z,
    const INT ldz,
    INT* m,
    f64* pl,
    f64* pr,
    f64* dif,
    c128* work,
    const INT lwork,
    INT* iwork,
    const INT liwork,
    INT* info)
{
    const INT IDIFJB = 3;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT lquery, swap, wantd, wantd1, wantd2, wantp;
    INT i, ierr, ijb, k, kase, ks, liwmin, lwmin, mn2, n1, n2;
    f64 dscale, dsum, rdscal, safmin;
    c128 temp1, temp2;
    INT isave[3];

    *info = 0;
    lquery = (lwork == -1 || liwork == -1);

    if (ijob < 0 || ijob > 5) {
        *info = -1;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -13;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -15;
    }

    if (*info != 0) {
        xerbla("ZTGSEN", -(*info));
        return;
    }

    ierr = 0;

    wantp = (ijob == 1 || ijob >= 4);
    wantd1 = (ijob == 2 || ijob == 4);
    wantd2 = (ijob == 3 || ijob == 5);
    wantd = (wantd1 || wantd2);

    /* Set M to the dimension of the specified pair of deflating
       subspaces. */
    *m = 0;
    if (!lquery || ijob != 0) {
        for (k = 0; k < n; k++) {
            alpha[k] = A[k + k * lda];
            beta[k] = B[k + k * ldb];
            if (k < n - 1) {
                if (select[k])
                    *m = *m + 1;
            } else {
                if (select[n - 1])
                    *m = *m + 1;
            }
        }
    }

    if (ijob == 1 || ijob == 2 || ijob == 4) {
        lwmin = 1 > 2 * (*m) * (n - *m) ? 1 : 2 * (*m) * (n - *m);
        liwmin = 1 > n + 2 ? 1 : n + 2;
    } else if (ijob == 3 || ijob == 5) {
        lwmin = 1 > 4 * (*m) * (n - *m) ? 1 : 4 * (*m) * (n - *m);
        INT tmp = 1 > 2 * (*m) * (n - *m) ? 1 : 2 * (*m) * (n - *m);
        liwmin = tmp > n + 2 ? tmp : n + 2;
    } else {
        lwmin = 1;
        liwmin = 1;
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
    iwork[0] = liwmin;

    if (lwork < lwmin && !lquery) {
        *info = -21;
    } else if (liwork < liwmin && !lquery) {
        *info = -23;
    }

    if (*info != 0) {
        xerbla("ZTGSEN", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible. */
    if (*m == n || *m == 0) {
        if (wantp) {
            *pl = ONE;
            *pr = ONE;
        }
        if (wantd) {
            dscale = ZERO;
            dsum = ONE;
            for (i = 0; i < n; i++) {
                zlassq(n, &A[0 + i * lda], 1, &dscale, &dsum);
                zlassq(n, &B[0 + i * ldb], 1, &dscale, &dsum);
            }
            dif[0] = dscale * sqrt(dsum);
            dif[1] = dif[0];
        }
        goto L70;
    }

    /* Get machine constant */
    safmin = dlamch("S");

    /* Collect the selected blocks at the top-left corner of (A, B). */
    ks = 0;
    for (k = 0; k < n; k++) {
        swap = select[k];
        if (swap) {
            ks = ks + 1;

            /* Swap the K-th block to position KS. Compute unitary Q
               and Z that will swap adjacent diagonal blocks in (A, B). */
            if (k != ks - 1) {
                INT ifst_val = k;
                INT ilst_val = ks - 1;
                ztgexc(wantq, wantz, n, A, lda, B, ldb, Q, ldq,
                       Z, ldz, ifst_val, &ilst_val, &ierr);
            }

            if (ierr > 0) {

                /* Swap is rejected: exit. */
                *info = 1;
                if (wantp) {
                    *pl = ZERO;
                    *pr = ZERO;
                }
                if (wantd) {
                    dif[0] = ZERO;
                    dif[1] = ZERO;
                }
                goto L70;
            }
        }
    }
    if (wantp) {

        /* Solve generalized Sylvester equation for R and L:
                   A11 * R - L * A22 = A12
                   B11 * R - L * B22 = B12 */
        n1 = *m;
        n2 = n - *m;
        i = n1;
        zlacpy("F", n1, n2, &A[0 + i * lda], lda, work, n1);
        zlacpy("F", n1, n2, &B[0 + i * ldb], ldb, &work[n1 * n2], n1);
        ijb = 0;
        ztgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda, work, n1,
               B, ldb, &B[i + i * ldb], ldb, &work[n1 * n2], n1,
               &dscale, &dif[0], &work[n1 * n2 * 2],
               lwork - 2 * n1 * n2, iwork, &ierr);

        /* Estimate the reciprocal of norms of "projections" onto
           left and right eigenspaces */
        rdscal = ZERO;
        dsum = ONE;
        zlassq(n1 * n2, work, 1, &rdscal, &dsum);
        *pl = rdscal * sqrt(dsum);
        if (*pl == ZERO) {
            *pl = ONE;
        } else {
            *pl = dscale / (sqrt(dscale * dscale / (*pl) + (*pl)) * sqrt(*pl));
        }
        rdscal = ZERO;
        dsum = ONE;
        zlassq(n1 * n2, &work[n1 * n2], 1, &rdscal, &dsum);
        *pr = rdscal * sqrt(dsum);
        if (*pr == ZERO) {
            *pr = ONE;
        } else {
            *pr = dscale / (sqrt(dscale * dscale / (*pr) + (*pr)) * sqrt(*pr));
        }
    }
    if (wantd) {

        /* Compute estimates Difu and Difl. */
        if (wantd1) {
            n1 = *m;
            n2 = n - *m;
            i = n1;
            ijb = IDIFJB;

            /* Frobenius norm-based Difu estimate. */
            ztgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda,
                   work, n1, B, ldb, &B[i + i * ldb], ldb, &work[n1 * n2],
                   n1, &dscale, &dif[0], &work[n1 * n2 * 2],
                   lwork - 2 * n1 * n2, iwork, &ierr);

            /* Frobenius norm-based Difl estimate. */
            ztgsyl("N", ijb, n2, n1, &A[i + i * lda], lda, A, lda,
                   work, n2, &B[i + i * ldb], ldb, B, ldb, &work[n1 * n2],
                   n2, &dscale, &dif[1], &work[n1 * n2 * 2],
                   lwork - 2 * n1 * n2, iwork, &ierr);
        } else {

            /* Compute 1-norm-based estimates of Difu and Difl using
               reversed communication with ZLACN2. In each step a
               generalized Sylvester equation or a transposed variant
               is solved. */
            kase = 0;
            n1 = *m;
            n2 = n - *m;
            i = n1;
            ijb = 0;
            mn2 = 2 * n1 * n2;

            /* 1-norm-based estimate of Difu. */
L40:
            zlacn2(mn2, &work[mn2], work, &dif[0], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    /* Solve generalized Sylvester equation */
                    ztgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda,
                           work, n1, B, ldb, &B[i + i * ldb], ldb,
                           &work[n1 * n2], n1, &dscale, &dif[0],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                } else {
                    /* Solve the transposed variant. */
                    ztgsyl("C", ijb, n1, n2, A, lda, &A[i + i * lda], lda,
                           work, n1, B, ldb, &B[i + i * ldb], ldb,
                           &work[n1 * n2], n1, &dscale, &dif[0],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                }
                goto L40;
            }
            dif[0] = dscale / dif[0];

            /* 1-norm-based estimate of Difl. */
L50:
            zlacn2(mn2, &work[mn2], work, &dif[1], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    /* Solve generalized Sylvester equation */
                    ztgsyl("N", ijb, n2, n1, &A[i + i * lda], lda, A, lda,
                           work, n2, &B[i + i * ldb], ldb, B, ldb,
                           &work[n1 * n2], n2, &dscale, &dif[1],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                } else {
                    /* Solve the transposed variant. */
                    ztgsyl("C", ijb, n2, n1, &A[i + i * lda], lda, A, lda,
                           work, n2, B, ldb, &B[i + i * ldb], ldb,
                           &work[n1 * n2], n2, &dscale, &dif[1],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                }
                goto L50;
            }
            dif[1] = dscale / dif[1];
        }
    }

    /* If B(K,K) is complex, make it real and positive (normalization
       of the generalized Schur form) and Store the generalized
       eigenvalues of reordered pair (A, B) */
    for (k = 0; k < n; k++) {
        dscale = cabs(B[k + k * ldb]);
        if (dscale > safmin) {
            temp1 = conj(B[k + k * ldb] / dscale);
            temp2 = B[k + k * ldb] / dscale;
            B[k + k * ldb] = CMPLX(dscale, 0.0);
            cblas_zscal(n - k - 1, &temp1, &B[k + (k + 1) * ldb], ldb);
            cblas_zscal(n - k, &temp1, &A[k + k * lda], lda);
            if (wantq)
                cblas_zscal(n, &temp2, &Q[0 + k * ldq], 1);
        } else {
            B[k + k * ldb] = CMPLX(ZERO, ZERO);
        }

        alpha[k] = A[k + k * lda];
        beta[k] = B[k + k * ldb];
    }

L70:

    work[0] = CMPLX((f64)lwmin, 0.0);
    iwork[0] = liwmin;
}
