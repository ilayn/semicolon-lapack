/**
 * @file ztgsna.c
 * @brief ZTGSNA estimates reciprocal condition numbers for eigenvalues/eigenvectors.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTGSNA estimates reciprocal condition numbers for specified
 * eigenvalues and/or eigenvectors of a matrix pair (A, B) in
 * generalized Schur canonical form (i.e., A and B are both
 * upper triangular).
 *
 * @param[in]     job     = 'E': condition numbers for eigenvalues only (S)
 *                        = 'V': condition numbers for eigenvectors only (DIF)
 *                        = 'B': condition numbers for both (S and DIF)
 * @param[in]     howmny  = 'A': compute for all eigenpairs
 *                        = 'S': compute for selected eigenpairs
 * @param[in]     select  Integer array of dimension (n). If howmny = 'S',
 *                        specifies the eigenpairs for which condition numbers
 *                        are required.
 * @param[in]     n       The order of the matrix pair (A, B). n >= 0.
 * @param[in]     A       Complex array of dimension (lda, n). Upper triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     B       Complex array of dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in]     VL      Complex array of dimension (ldvl, m). Left eigenvectors.
 * @param[in]     ldvl    The leading dimension of VL. ldvl >= 1; if job='E'/'B', ldvl >= n.
 * @param[in]     VR      Complex array of dimension (ldvr, m). Right eigenvectors.
 * @param[in]     ldvr    The leading dimension of VR. ldvr >= 1; if job='E'/'B', ldvr >= n.
 * @param[out]    S       Double precision array of dimension (mm). Reciprocal condition
 *                        numbers of eigenvalues.
 * @param[out]    dif     Double precision array of dimension (mm). Reciprocal condition
 *                        numbers of eigenvectors.
 * @param[in]     mm      The number of elements in S and dif. mm >= m.
 * @param[out]    m       The number of elements used in S and dif.
 * @param[out]    work    Complex workspace array of dimension (max(1, lwork)).
 * @param[in]     lwork   The dimension of work. lwork >= max(1, n).
 *                        If job = 'V' or 'B', lwork >= max(1, 2*n*n).
 * @param[out]    iwork   Integer array of dimension (n+2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ztgsna(
    const char* job,
    const char* howmny,
    const int* const restrict select,
    const int n,
    const c128* const restrict A,
    const int lda,
    const c128* const restrict B,
    const int ldb,
    const c128* const restrict VL,
    const int ldvl,
    const c128* const restrict VR,
    const int ldvr,
    f64* const restrict S,
    f64* const restrict dif,
    const int mm,
    int* m,
    c128* const restrict work,
    const int lwork,
    int* const restrict iwork,
    int* info)
{
    const int IDIFJB = 3;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    int lquery, somcon, wantbh, wantdf, wants;
    int i, ierr, ilst, k, ks, lwmin, n1, n2;
    f64 cond, eps, lnrm, rnrm, scale, smlnum;
    c128 yhax, yhbx;
    c128 dummy[1], dummy1[1];

    wantbh = (job[0] == 'B' || job[0] == 'b');
    wants = (job[0] == 'E' || job[0] == 'e') || wantbh;
    wantdf = (job[0] == 'V' || job[0] == 'v') || wantbh;

    somcon = (howmny[0] == 'S' || howmny[0] == 's');

    *info = 0;
    lquery = (lwork == -1);

    if (!wants && !wantdf) {
        *info = -1;
    } else if (!(howmny[0] == 'A' || howmny[0] == 'a') && !somcon) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (wants && ldvl < n) {
        *info = -10;
    } else if (wants && ldvr < n) {
        *info = -12;
    } else {

        /* Set M to the number of eigenpairs for which condition numbers
           are required, and test MM. */
        if (somcon) {
            *m = 0;
            for (k = 0; k < n; k++) {
                if (select[k])
                    *m = *m + 1;
            }
        } else {
            *m = n;
        }

        if (n == 0) {
            lwmin = 1;
        } else if ((job[0] == 'V' || job[0] == 'v') ||
                   (job[0] == 'B' || job[0] == 'b')) {
            lwmin = 2 * n * n;
        } else {
            lwmin = n;
        }
        work[0] = CMPLX((f64)lwmin, 0.0);

        if (mm < *m) {
            *info = -15;
        } else if (lwork < lwmin && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("ZTGSNA", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    (void)smlnum;
    ks = 0;

    for (k = 0; k < n; k++) {

        /* Determine whether condition numbers are required for the k-th
           eigenpair. */
        if (somcon) {
            if (!select[k])
                continue;
        }

        ks = ks + 1;

        if (wants) {

            /* Compute the reciprocal condition number of the k-th
               eigenvalue. */
            rnrm = cblas_dznrm2(n, &VR[0 + (ks - 1) * ldvr], 1);
            lnrm = cblas_dznrm2(n, &VL[0 + (ks - 1) * ldvl], 1);
            cblas_zgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, A, lda,
                        &VR[0 + (ks - 1) * ldvr], 1, &CZERO, work, 1);
            cblas_zdotc_sub(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1, &yhax);
            cblas_zgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, B, ldb,
                        &VR[0 + (ks - 1) * ldvr], 1, &CZERO, work, 1);
            cblas_zdotc_sub(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1, &yhbx);
            cond = dlapy2(cabs(yhax), cabs(yhbx));
            if (cond == ZERO) {
                S[ks - 1] = -ONE;
            } else {
                S[ks - 1] = cond / (rnrm * lnrm);
            }
        }

        if (wantdf) {
            if (n == 1) {
                dif[ks - 1] = dlapy2(cabs(A[0 + 0 * lda]), cabs(B[0 + 0 * ldb]));
                continue;
            }

            /* Estimate the reciprocal condition number of the k-th
               eigenvectors. */

            /* Copy the matrix (A, B) to the array WORK and move the
               (k,k)th pair to the (1,1) position. */
            zlacpy("F", n, n, A, lda, work, n);
            zlacpy("F", n, n, B, ldb, &work[n * n], n);
            ilst = 0;

            ztgexc(0, 0, n, work, n, &work[n * n], n,
                   dummy, 1, dummy1, 1, k, &ilst, &ierr);

            if (ierr > 0) {

                /* Ill-conditioned problem - swap rejected. */
                dif[ks - 1] = ZERO;
            } else {

                /* Reordering successful, solve generalized Sylvester
                   equation for R and L. */
                n1 = 1;
                n2 = n - n1;
                i = n * n;
                ztgsyl("N", IDIFJB, n2, n1,
                       &work[n * n1 + n1], n, work, n, &work[n1], n,
                       &work[n * n1 + n1 + i], n, &work[i], n,
                       &work[n1 + i], n, &scale, &dif[ks - 1],
                       dummy, 1, iwork, &ierr);
            }
        }
    }
    work[0] = CMPLX((f64)lwmin, 0.0);
}
