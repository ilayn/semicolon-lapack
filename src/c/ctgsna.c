/**
 * @file ctgsna.c
 * @brief CTGSNA estimates reciprocal condition numbers for eigenvalues/eigenvectors.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CTGSNA estimates reciprocal condition numbers for specified
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
 * @param[out]    S       Single precision array of dimension (mm). Reciprocal condition
 *                        numbers of eigenvalues.
 * @param[out]    dif     Single precision array of dimension (mm). Reciprocal condition
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
void ctgsna(
    const char* job,
    const char* howmny,
    const INT* restrict select,
    const INT n,
    const c64* restrict A,
    const INT lda,
    const c64* restrict B,
    const INT ldb,
    const c64* restrict VL,
    const INT ldvl,
    const c64* restrict VR,
    const INT ldvr,
    f32* restrict S,
    f32* restrict dif,
    const INT mm,
    INT* m,
    c64* restrict work,
    const INT lwork,
    INT* restrict iwork,
    INT* info)
{
    const INT IDIFJB = 3;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    INT lquery, somcon, wantbh, wantdf, wants;
    INT i, ierr, ilst, k, ks, lwmin, n1, n2;
    f32 cond, eps, lnrm, rnrm, scale, smlnum;
    c64 yhax, yhbx;
    c64 dummy[1], dummy1[1];

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
        work[0] = CMPLXF((f32)lwmin, 0.0f);

        if (mm < *m) {
            *info = -15;
        } else if (lwork < lwmin && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("CTGSNA", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* Get machine constants */
    eps = slamch("P");
    smlnum = slamch("S") / eps;
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
            rnrm = cblas_scnrm2(n, &VR[0 + (ks - 1) * ldvr], 1);
            lnrm = cblas_scnrm2(n, &VL[0 + (ks - 1) * ldvl], 1);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, A, lda,
                        &VR[0 + (ks - 1) * ldvr], 1, &CZERO, work, 1);
            cblas_cdotc_sub(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1, &yhax);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, B, ldb,
                        &VR[0 + (ks - 1) * ldvr], 1, &CZERO, work, 1);
            cblas_cdotc_sub(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1, &yhbx);
            cond = slapy2(cabsf(yhax), cabsf(yhbx));
            if (cond == ZERO) {
                S[ks - 1] = -ONE;
            } else {
                S[ks - 1] = cond / (rnrm * lnrm);
            }
        }

        if (wantdf) {
            if (n == 1) {
                dif[ks - 1] = slapy2(cabsf(A[0 + 0 * lda]), cabsf(B[0 + 0 * ldb]));
                continue;
            }

            /* Estimate the reciprocal condition number of the k-th
               eigenvectors. */

            /* Copy the matrix (A, B) to the array WORK and move the
               (k,k)th pair to the (1,1) position. */
            clacpy("F", n, n, A, lda, work, n);
            clacpy("F", n, n, B, ldb, &work[n * n], n);
            ilst = 0;

            ctgexc(0, 0, n, work, n, &work[n * n], n,
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
                ctgsyl("N", IDIFJB, n2, n1,
                       &work[n * n1 + n1], n, work, n, &work[n1], n,
                       &work[n * n1 + n1 + i], n, &work[i], n,
                       &work[n1 + i], n, &scale, &dif[ks - 1],
                       dummy, 1, iwork, &ierr);
            }
        }
    }
    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
