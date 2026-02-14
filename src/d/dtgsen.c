/**
 * @file dtgsen.c
 * @brief DTGSEN reorders the generalized real Schur decomposition.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DTGSEN reorders the generalized real Schur decomposition of a real
 * matrix pair (A, B) (in terms of an orthonormal equivalence trans-
 * formation Q**T * (A, B) * Z), so that a selected cluster of eigenvalues
 * appears in the leading diagonal blocks of the upper quasi-triangular
 * matrix A and the upper triangular B.
 *
 * @param[in]     ijob    Specifies whether condition numbers are required.
 *                        = 0: Only reorder w.r.t. SELECT. No extras.
 *                        = 1: Reciprocal of norms of "projections" (PL and PR).
 *                        = 2: Upper bounds on Difu and Difl.
 *                        = 3: Estimate of Difu and Difl.
 *                        = 4: Compute PL, PR and DIF (i.e. 0, 1 and 2 above).
 *                        = 5: Compute PL, PR and DIF (i.e. 0, 1 and 3 above).
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     select  Integer array of dimension (n). Specifies the eigenvalues
 *                        in the selected cluster.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Array of dimension (lda, n). Upper quasi-triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Array of dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    alphar  Array of dimension (n). Real parts of eigenvalues.
 * @param[out]    alphai  Array of dimension (n). Imaginary parts of eigenvalues.
 * @param[out]    beta    Array of dimension (n). Scale factors for eigenvalues.
 * @param[in,out] Q       Array of dimension (ldq, n). The orthogonal matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Array of dimension (ldz, n). The orthogonal matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[out]    m       The dimension of the specified pair of deflating subspaces.
 * @param[out]    pl      Lower bound on reciprocal of projection onto left eigenspace.
 * @param[out]    pr      Lower bound on reciprocal of projection onto right eigenspace.
 * @param[out]    dif     Array of dimension (2). Estimates of Difu and Difl.
 * @param[out]    work    Array of dimension (lwork).
 * @param[in]     lwork   The dimension of work. lwork >= 4*n+16.
 * @param[out]    iwork   Integer array of dimension (liwork).
 * @param[in]     liwork  The dimension of iwork. liwork >= 1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: reordering failed
 */
void dtgsen(
    const int ijob,
    const int wantq,
    const int wantz,
    const int* restrict select,
    const int n,
    f64* restrict A,
    const int lda,
    f64* restrict B,
    const int ldb,
    f64* restrict alphar,
    f64* restrict alphai,
    f64* restrict beta,
    f64* restrict Q,
    const int ldq,
    f64* restrict Z,
    const int ldz,
    int* m,
    f64* pl,
    f64* pr,
    f64* restrict dif,
    f64* restrict work,
    const int lwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const int IDIFJB = 3;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int lquery, pair, swap, wantd, wantd1, wantd2, wantp;
    int i, ierr, ijb, k, kase, kk, ks, liwmin, lwmin, mn2, n1, n2;
    f64 dscale, dsum, eps, rdscal, smlnum;
    int isave[3];

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
        *info = -14;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -16;
    }

    if (*info != 0) {
        xerbla("DTGSEN", -(*info));
        return;
    }

    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    ierr = 0;

    wantp = (ijob == 1 || ijob >= 4);
    wantd1 = (ijob == 2 || ijob == 4);
    wantd2 = (ijob == 3 || ijob == 5);
    wantd = (wantd1 || wantd2);

    /* Set M to the dimension of the specified pair of deflating
       subspaces. */
    *m = 0;
    pair = 0;
    if (!lquery || ijob != 0) {
        for (k = 0; k < n; k++) {
            if (pair) {
                pair = 0;
            } else {
                if (k < n - 1) {
                    if (A[(k + 1) + k * lda] == ZERO) {
                        if (select[k]) {
                            *m = *m + 1;
                        }
                    } else {
                        pair = 1;
                        if (select[k] || select[k + 1]) {
                            *m = *m + 2;
                        }
                    }
                } else {
                    if (select[n - 1]) {
                        *m = *m + 1;
                    }
                }
            }
        }
    }

    if (ijob == 1 || ijob == 2 || ijob == 4) {
        lwmin = 1 > 4 * n + 16 ? 1 : 4 * n + 16;
        lwmin = lwmin > 2 * (*m) * (n - *m) ? lwmin : 2 * (*m) * (n - *m);
        liwmin = 1 > n + 6 ? 1 : n + 6;
    } else if (ijob == 3 || ijob == 5) {
        lwmin = 1 > 4 * n + 16 ? 1 : 4 * n + 16;
        lwmin = lwmin > 4 * (*m) * (n - *m) ? lwmin : 4 * (*m) * (n - *m);
        liwmin = 1 > 2 * (*m) * (n - *m) ? 1 : 2 * (*m) * (n - *m);
        liwmin = liwmin > n + 6 ? liwmin : n + 6;
    } else {
        lwmin = 1 > 4 * n + 16 ? 1 : 4 * n + 16;
        liwmin = 1;
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;

    if (lwork < lwmin && !lquery) {
        *info = -22;
    } else if (liwork < liwmin && !lquery) {
        *info = -24;
    }

    if (*info != 0) {
        xerbla("DTGSEN", -(*info));
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
                dlassq(n, &A[0 + i * lda], 1, &dscale, &dsum);
                dlassq(n, &B[0 + i * ldb], 1, &dscale, &dsum);
            }
            dif[0] = dscale * sqrt(dsum);
            dif[1] = dif[0];
        }
        goto L60;
    }

    /* Collect the selected blocks at the top-left corner of (A, B). */
    ks = 0;
    pair = 0;
    for (k = 0; k < n; k++) {
        if (pair) {
            pair = 0;
        } else {

            swap = select[k];
            if (k < n - 1) {
                if (A[(k + 1) + k * lda] != ZERO) {
                    pair = 1;
                    swap = (swap || select[k + 1]);
                }
            }

            if (swap) {
                /* Swap the K-th block to position KS.
                   Perform the reordering of diagonal blocks in (A, B)
                   by orthogonal transformation matrices and update
                   Q and Z accordingly (if requested): */
                kk = k;
                if (k != ks) {
                    dtgexc(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                           &kk, &ks, work, lwork, &ierr);
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
                    goto L60;
                }

                if (pair) {
                    ks = ks + 1;
                }
                ks = ks + 1;
            }
        }
    }
    if (wantp) {

        /* Solve generalized Sylvester equation for R and L
           and compute PL and PR. */
        n1 = *m;
        n2 = n - *m;
        i = n1;
        ijb = 0;
        dlacpy("F", n1, n2, &A[0 + i * lda], lda, work, n1);
        dlacpy("F", n1, n2, &B[0 + i * ldb], ldb, &work[n1 * n2], n1);
        dtgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda, work, n1,
               B, ldb, &B[i + i * ldb], ldb, &work[n1 * n2], n1,
               &dscale, &dif[0], &work[2 * n1 * n2], lwork - 2 * n1 * n2,
               iwork, &ierr);

        /* Estimate the reciprocal of norms of "projections" onto left
           and right eigenspaces. */
        rdscal = ZERO;
        dsum = ONE;
        dlassq(n1 * n2, work, 1, &rdscal, &dsum);
        *pl = rdscal * sqrt(dsum);
        if (*pl == ZERO) {
            *pl = ONE;
        } else {
            *pl = dscale / (sqrt(dscale * dscale / (*pl) + (*pl)) * sqrt(*pl));
        }
        rdscal = ZERO;
        dsum = ONE;
        dlassq(n1 * n2, &work[n1 * n2], 1, &rdscal, &dsum);
        *pr = rdscal * sqrt(dsum);
        if (*pr == ZERO) {
            *pr = ONE;
        } else {
            *pr = dscale / (sqrt(dscale * dscale / (*pr) + (*pr)) * sqrt(*pr));
        }
    }

    if (wantd) {

        /* Compute estimates of Difu and Difl. */
        if (wantd1) {
            n1 = *m;
            n2 = n - *m;
            i = n1;
            ijb = IDIFJB;

            /* Frobenius norm-based Difu-estimate. */
            dtgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda, work, n1,
                   B, ldb, &B[i + i * ldb], ldb, &work[n1 * n2], n1,
                   &dscale, &dif[0], &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                   iwork, &ierr);

            /* Frobenius norm-based Difl-estimate. */
            dtgsyl("N", ijb, n2, n1, &A[i + i * lda], lda, A, lda, work, n2,
                   &B[i + i * ldb], ldb, B, ldb, &work[n1 * n2], n2,
                   &dscale, &dif[1], &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                   iwork, &ierr);
        } else {

            /* Compute 1-norm-based estimates of Difu and Difl using
               reversed communication with DLACN2. */
            kase = 0;
            n1 = *m;
            n2 = n - *m;
            i = n1;
            ijb = 0;
            mn2 = 2 * n1 * n2;

            /* 1-norm-based estimate of Difu. */
L40:
            dlacn2(mn2, &work[mn2], work, iwork, &dif[0], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    /* Solve generalized Sylvester equation. */
                    dtgsyl("N", ijb, n1, n2, A, lda, &A[i + i * lda], lda,
                           work, n1, B, ldb, &B[i + i * ldb], ldb,
                           &work[n1 * n2], n1, &dscale, &dif[0],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                } else {
                    /* Solve the transposed variant. */
                    dtgsyl("T", ijb, n1, n2, A, lda, &A[i + i * lda], lda,
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
            dlacn2(mn2, &work[mn2], work, iwork, &dif[1], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    /* Solve generalized Sylvester equation. */
                    dtgsyl("N", ijb, n2, n1, &A[i + i * lda], lda, A, lda,
                           work, n2, &B[i + i * ldb], ldb, B, ldb,
                           &work[n1 * n2], n2, &dscale, &dif[1],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                } else {
                    /* Solve the transposed variant. */
                    dtgsyl("T", ijb, n2, n1, &A[i + i * lda], lda, A, lda,
                           work, n2, &B[i + i * ldb], ldb, B, ldb,
                           &work[n1 * n2], n2, &dscale, &dif[1],
                           &work[2 * n1 * n2], lwork - 2 * n1 * n2,
                           iwork, &ierr);
                }
                goto L50;
            }
            dif[1] = dscale / dif[1];

        }
    }

L60:

    /* Compute generalized eigenvalues of reordered pair (A, B) and
       normalize the generalized Schur form. */
    pair = 0;
    for (k = 0; k < n; k++) {
        if (pair) {
            pair = 0;
        } else {

            if (k < n - 1) {
                if (A[(k + 1) + k * lda] != ZERO) {
                    pair = 1;
                }
            }

            if (pair) {

                /* Compute the eigenvalue(s) at position K. */
                work[0] = A[k + k * lda];
                work[1] = A[(k + 1) + k * lda];
                work[2] = A[k + (k + 1) * lda];
                work[3] = A[(k + 1) + (k + 1) * lda];
                work[4] = B[k + k * ldb];
                work[5] = B[(k + 1) + k * ldb];
                work[6] = B[k + (k + 1) * ldb];
                work[7] = B[(k + 1) + (k + 1) * ldb];
                dlag2(work, 2, &work[4], 2, smlnum * eps, &beta[k],
                      &beta[k + 1], &alphar[k], &alphar[k + 1], &alphai[k]);
                alphai[k + 1] = -alphai[k];

            } else {

                if ((B[k + k * ldb] < ZERO ? -ONE : ONE) < ZERO) {

                    /* If B(K,K) is negative, make it positive */
                    for (i = 0; i < n; i++) {
                        A[k + i * lda] = -A[k + i * lda];
                        B[k + i * ldb] = -B[k + i * ldb];
                        if (wantq) {
                            Q[i + k * ldq] = -Q[i + k * ldq];
                        }
                    }
                }

                alphar[k] = A[k + k * lda];
                alphai[k] = ZERO;
                beta[k] = B[k + k * ldb];

            }
        }
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
