/**
 * @file dtgsna.c
 * @brief DTGSNA estimates reciprocal condition numbers for eigenvalues/eigenvectors.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTGSNA estimates reciprocal condition numbers for specified
 * eigenvalues and/or eigenvectors of a matrix pair (A, B) in
 * generalized real Schur canonical form.
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
 * @param[in]     A       Array of dimension (lda, n). Upper quasi-triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     B       Array of dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in]     VL      Array of dimension (ldvl, m). Left eigenvectors.
 * @param[in]     ldvl    The leading dimension of VL. ldvl >= 1; if job='E'/'B', ldvl >= n.
 * @param[in]     VR      Array of dimension (ldvr, m). Right eigenvectors.
 * @param[in]     ldvr    The leading dimension of VR. ldvr >= 1; if job='E'/'B', ldvr >= n.
 * @param[out]    S       Array of dimension (mm). Reciprocal condition numbers of eigenvalues.
 * @param[out]    dif     Array of dimension (mm). Reciprocal condition numbers of eigenvectors.
 * @param[in]     mm      The number of elements in S and dif. mm >= m.
 * @param[out]    m       The number of elements used in S and dif.
 * @param[out]    work    Array of dimension (lwork).
 * @param[in]     lwork   The dimension of work. lwork >= max(1, n).
 *                        If job = 'V' or 'B', lwork >= 2*n*(n+2)+16.
 * @param[out]    iwork   Integer array of dimension (n+6).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dtgsna(
    const char* job,
    const char* howmny,
    const INT* restrict select,
    const INT n,
    const f64* restrict A,
    const INT lda,
    const f64* restrict B,
    const INT ldb,
    const f64* restrict VL,
    const INT ldvl,
    const f64* restrict VR,
    const INT ldvr,
    f64* restrict S,
    f64* restrict dif,
    const INT mm,
    INT* m,
    f64* restrict work,
    const INT lwork,
    INT* restrict iwork,
    INT* info)
{
    const INT DIFDRI = 3;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;

    INT lquery, pair, somcon, wantbh, wantdf, wants;
    INT i, ierr, ifst, ilst, iz, k, ks, lwmin, n1, n2;
    f64 alphai, alphar, alprqt, beta, c1, c2, cond = 0.0;
    f64 eps, lnrm, rnrm, root1, root2, scale, smlnum;
    f64 tmpii, tmpir, tmpri, tmprr, uhav, uhavi, uhbv, uhbvi;
    f64 dummy[1], dummy1[1];

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
            pair = 0;
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
        } else {
            *m = n;
        }

        if (n == 0) {
            lwmin = 1;
        } else if ((job[0] == 'V' || job[0] == 'v') ||
                   (job[0] == 'B' || job[0] == 'b')) {
            lwmin = 2 * n * (n + 2) + 16;
        } else {
            lwmin = n;
        }
        work[0] = (f64)lwmin;

        if (mm < *m) {
            *info = -15;
        } else if (lwork < lwmin && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("DTGSNA", -(*info));
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
    ks = 0;
    pair = 0;

    for (k = 0; k < n; k++) {

        /* Determine whether A(k,k) begins a 1-by-1 or 2-by-2 block. */
        if (pair) {
            pair = 0;
            continue;
        } else {
            if (k < n - 1) {
                pair = (A[(k + 1) + k * lda] != ZERO);
            }
        }

        /* Determine whether condition numbers are required for the k-th
           eigenpair. */
        if (somcon) {
            if (pair) {
                if (!select[k] && !select[k + 1]) {
                    continue;
                }
            } else {
                if (!select[k]) {
                    continue;
                }
            }
        }

        ks = ks + 1;

        if (wants) {

            /* Compute the reciprocal condition number of the k-th
               eigenvalue. */
            if (pair) {

                /* Complex eigenvalue pair. */
                rnrm = dlapy2(cblas_dnrm2(n, &VR[0 + (ks - 1) * ldvr], 1),
                              cblas_dnrm2(n, &VR[0 + ks * ldvr], 1));
                lnrm = dlapy2(cblas_dnrm2(n, &VL[0 + (ks - 1) * ldvl], 1),
                              cblas_dnrm2(n, &VL[0 + ks * ldvl], 1));
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, A, lda,
                            &VR[0 + (ks - 1) * ldvr], 1, ZERO, work, 1);
                tmprr = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                tmpri = cblas_ddot(n, work, 1, &VL[0 + ks * ldvl], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, A, lda,
                            &VR[0 + ks * ldvr], 1, ZERO, work, 1);
                tmpii = cblas_ddot(n, work, 1, &VL[0 + ks * ldvl], 1);
                tmpir = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                uhav = tmprr + tmpii;
                uhavi = tmpir - tmpri;
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, B, ldb,
                            &VR[0 + (ks - 1) * ldvr], 1, ZERO, work, 1);
                tmprr = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                tmpri = cblas_ddot(n, work, 1, &VL[0 + ks * ldvl], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, B, ldb,
                            &VR[0 + ks * ldvr], 1, ZERO, work, 1);
                tmpii = cblas_ddot(n, work, 1, &VL[0 + ks * ldvl], 1);
                tmpir = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                uhbv = tmprr + tmpii;
                uhbvi = tmpir - tmpri;
                uhav = dlapy2(uhav, uhavi);
                uhbv = dlapy2(uhbv, uhbvi);
                cond = dlapy2(uhav, uhbv);
                S[ks - 1] = cond / (rnrm * lnrm);
                S[ks] = S[ks - 1];

            } else {

                /* Real eigenvalue. */
                rnrm = cblas_dnrm2(n, &VR[0 + (ks - 1) * ldvr], 1);
                lnrm = cblas_dnrm2(n, &VL[0 + (ks - 1) * ldvl], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, A, lda,
                            &VR[0 + (ks - 1) * ldvr], 1, ZERO, work, 1);
                uhav = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, B, ldb,
                            &VR[0 + (ks - 1) * ldvr], 1, ZERO, work, 1);
                uhbv = cblas_ddot(n, work, 1, &VL[0 + (ks - 1) * ldvl], 1);
                cond = dlapy2(uhav, uhbv);
                if (cond == ZERO) {
                    S[ks - 1] = -ONE;
                } else {
                    S[ks - 1] = cond / (rnrm * lnrm);
                }
            }
        }

        if (wantdf) {
            if (n == 1) {
                dif[ks - 1] = dlapy2(A[0 + 0 * lda], B[0 + 0 * ldb]);
                continue;
            }

            /* Estimate the reciprocal condition number of the k-th
               eigenvectors. */
            if (pair) {

                /* Copy the 2-by-2 pencil beginning at (A(k,k), B(k,k)).
                   Compute the eigenvalue(s) at position K. */
                work[0] = A[k + k * lda];
                work[1] = A[(k + 1) + k * lda];
                work[2] = A[k + (k + 1) * lda];
                work[3] = A[(k + 1) + (k + 1) * lda];
                work[4] = B[k + k * ldb];
                work[5] = B[(k + 1) + k * ldb];
                work[6] = B[k + (k + 1) * ldb];
                work[7] = B[(k + 1) + (k + 1) * ldb];
                dlag2(work, 2, &work[4], 2, smlnum * eps, &beta,
                      &dummy1[0], &alphar, &dummy[0], &alphai);
                alprqt = ONE;
                c1 = TWO * (alphar * alphar + alphai * alphai + beta * beta);
                c2 = FOUR * beta * beta * alphai * alphai;
                root1 = c1 + sqrt(c1 * c1 - 4.0 * c2);
                root1 = root1 / TWO;
                root2 = c2 / root1;
                cond = sqrt(root1) < sqrt(root2) ? sqrt(root1) : sqrt(root2);
            }

            /* Copy the matrix (A, B) to the array WORK and swap the
               diagonal block beginning at A(k,k) to the (1,1) position. */
            dlacpy("F", n, n, A, lda, work, n);
            dlacpy("F", n, n, B, ldb, &work[n * n], n);
            ifst = k;
            ilst = 0;

            dtgexc(0, 0, n, work, n, &work[n * n], n,
                   dummy, 1, dummy1, 1, &ifst, &ilst,
                   &work[2 * n * n], lwork - 2 * n * n, &ierr);

            if (ierr > 0) {

                /* Ill-conditioned problem - swap rejected. */
                dif[ks - 1] = ZERO;
            } else {

                /* Reordering successful, solve generalized Sylvester
                   equation for R and L. */
                n1 = 1;
                if (work[1] != ZERO) {
                    n1 = 2;
                }
                n2 = n - n1;
                if (n2 == 0) {
                    dif[ks - 1] = cond;
                } else {
                    i = n * n;
                    iz = 2 * n * n;
                    dtgsyl("N", DIFDRI, n2, n1,
                           &work[n * n1 + n1], n, work, n, &work[n1], n,
                           &work[n * n1 + n1 + i], n, &work[i], n,
                           &work[n1 + i], n, &scale, &dif[ks - 1],
                           &work[iz], lwork - 2 * n * n, iwork, &ierr);

                    if (pair) {
                        f64 tmp = (ONE > alprqt ? ONE : alprqt) * dif[ks - 1];
                        dif[ks - 1] = tmp < cond ? tmp : cond;
                    }
                }
            }
            if (pair) {
                dif[ks] = dif[ks - 1];
            }
        }
        if (pair) {
            ks = ks + 1;
        }
    }
    work[0] = (f64)lwmin;
}
