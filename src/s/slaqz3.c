/**
 * @file slaqz3.c
 * @brief SLAQZ3 performs aggressive early deflation (AED).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAQZ3 performs AED (Aggressive Early Deflation).
 *
 * @param[in]     ilschur  Determines whether or not to update the full Schur form.
 * @param[in]     ilq      Determines whether or not to update the matrix Q.
 * @param[in]     ilz      Determines whether or not to update the matrix Z.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      Lower bound of active submatrix (0-based).
 * @param[in]     ihi      Upper bound of active submatrix (0-based).
 * @param[in]     nw       The desired size of the deflation window.
 * @param[in,out] A        Matrix A.
 * @param[in]     lda      Leading dimension of A.
 * @param[in,out] B        Matrix B.
 * @param[in]     ldb      Leading dimension of B.
 * @param[in,out] Q        Matrix Q.
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in,out] Z        Matrix Z.
 * @param[in]     ldz      Leading dimension of Z.
 * @param[out]    ns       The number of unconverged eigenvalues available as shifts.
 * @param[out]    nd       The number of converged eigenvalues found.
 * @param[out]    alphar   Real parts of eigenvalues.
 * @param[out]    alphai   Imaginary parts of eigenvalues.
 * @param[out]    beta     Scale factors for eigenvalues.
 * @param[in,out] QC       Workspace for accumulated Q transformations.
 * @param[in]     ldqc     Leading dimension of QC.
 * @param[in,out] ZC       Workspace for accumulated Z transformations.
 * @param[in]     ldzc     Leading dimension of ZC.
 * @param[out]    work     Workspace array.
 * @param[in]     lwork    Dimension of workspace. If lwork = -1, workspace query.
 * @param[in]     rec      Current recursion level. Should be set to 0 on first call.
 * @param[out]    info
 *                         - = 0: successful exit, < 0: illegal argument.
 */
void slaqz3(
    const int ilschur,
    const int ilq,
    const int ilz,
    const int n,
    const int ilo,
    const int ihi,
    const int nw,
    f32* const restrict A,
    const int lda,
    f32* const restrict B,
    const int ldb,
    f32* const restrict Q,
    const int ldq,
    f32* const restrict Z,
    const int ldz,
    int* ns,
    int* nd,
    f32* const restrict alphar,
    f32* const restrict alphai,
    f32* const restrict beta,
    f32* const restrict QC,
    const int ldqc,
    f32* const restrict ZC,
    const int ldzc,
    f32* const restrict work,
    const int lwork,
    const int rec,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int bulge;
    int jw, kwtop, kwbot, istopm, istartm, k, k2, dtgexc_info;
    int ifst, ilst, lworkreq, qz_small_info;
    f32 s, smlnum, ulp, safmin, c1, s1, temp;

    *info = 0;

    /* Set up deflation window */
    jw = (nw < ihi - ilo + 1) ? nw : ihi - ilo + 1;
    kwtop = ihi - jw + 1;
    if (kwtop == ilo) {
        s = ZERO;
    } else {
        s = A[kwtop + (kwtop - 1) * lda];
    }

    /* Determine required workspace */
    ifst = 0;
    ilst = jw - 1;
    stgexc(1, 1, jw, A, lda, B, ldb, QC, ldqc, ZC, ldzc,
           &ifst, &ilst, work, -1, &dtgexc_info);
    lworkreq = (int)work[0];
    slaqz0("S", "V", "V", jw, 0, jw - 1, &A[kwtop + kwtop * lda], lda,
           &B[kwtop + kwtop * ldb], ldb, alphar, alphai, beta, QC, ldqc,
           ZC, ldzc, work, -1, rec + 1, &qz_small_info);
    lworkreq = (lworkreq > (int)work[0] + 2 * jw * jw) ? lworkreq : (int)work[0] + 2 * jw * jw;
    {
        int tmp1 = n * nw;
        int tmp2 = 2 * nw * nw + n;
        int tmp = (tmp1 > tmp2) ? tmp1 : tmp2;
        lworkreq = (lworkreq > tmp) ? lworkreq : tmp;
    }
    if (lwork == -1) {
        /* workspace query, quick return */
        work[0] = (f32)lworkreq;
        return;
    } else if (lwork < lworkreq) {
        *info = -26;
    }

    if (*info != 0) {
        xerbla("SLAQZ3", -(*info));
        return;
    }

    /* Get machine constants */
    safmin = slamch("S");
    (void)(ONE / safmin);  /* safmax computed in Fortran but unused */
    ulp = slamch("P");
    smlnum = safmin * ((f32)n / ulp);

    if (ihi == kwtop) {
        /* 1 by 1 deflation window, just try a regular deflation */
        alphar[kwtop] = A[kwtop + kwtop * lda];
        alphai[kwtop] = ZERO;
        beta[kwtop] = B[kwtop + kwtop * ldb];
        *ns = 1;
        *nd = 0;
        {
            f32 tmp = fabsf(A[kwtop + kwtop * lda]);
            if (smlnum > tmp) tmp = smlnum;
            tmp = ulp * tmp;
            if (fabsf(s) <= ((smlnum > tmp) ? smlnum : tmp)) {
                *ns = 0;
                *nd = 1;
                if (kwtop > ilo) {
                    A[kwtop + (kwtop - 1) * lda] = ZERO;
                }
            }
        }
        return;
    }

    /* Store window in case of convergence failure */
    slacpy("A", jw, jw, &A[kwtop + kwtop * lda], lda, work, jw);
    slacpy("A", jw, jw, &B[kwtop + kwtop * ldb], ldb, &work[jw * jw], jw);

    /* Transform window to real Schur form */
    slaset("F", jw, jw, ZERO, ONE, QC, ldqc);
    slaset("F", jw, jw, ZERO, ONE, ZC, ldzc);
    slaqz0("S", "V", "V", jw, 0, jw - 1, &A[kwtop + kwtop * lda], lda,
           &B[kwtop + kwtop * ldb], ldb, alphar, alphai, beta, QC, ldqc,
           ZC, ldzc, &work[2 * jw * jw], lwork - 2 * jw * jw, rec + 1,
           &qz_small_info);

    if (qz_small_info != 0) {
        /* Convergence failure, restore the window and exit */
        *nd = 0;
        *ns = jw - qz_small_info;
        slacpy("A", jw, jw, work, jw, &A[kwtop + kwtop * lda], lda);
        slacpy("A", jw, jw, &work[jw * jw], jw, &B[kwtop + kwtop * ldb], ldb);
        return;
    }

    /* Deflation detection loop */
    if (kwtop == ilo || s == ZERO) {
        kwbot = kwtop - 1;
    } else {
        kwbot = ihi;
        k = 0;
        k2 = 0;
        while (k <= jw - 1) {
            bulge = 0;
            if (kwbot - kwtop + 1 >= 2) {
                bulge = (A[kwbot + (kwbot - 1) * lda] != ZERO);
            }
            if (bulge) {
                /* Try to deflate complex conjugate eigenvalue pair */
                temp = fabsf(A[kwbot + kwbot * lda]) +
                       sqrtf(fabsf(A[kwbot + (kwbot - 1) * lda])) *
                       sqrtf(fabsf(A[(kwbot - 1) + kwbot * lda]));
                if (temp == ZERO) {
                    temp = fabsf(s);
                }
                {
                    f32 tmp1 = fabsf(s * QC[0 + (kwbot - kwtop - 1) * ldqc]);
                    f32 tmp2 = fabsf(s * QC[0 + (kwbot - kwtop) * ldqc]);
                    f32 maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
                    f32 thresh = (smlnum > ulp * temp) ? smlnum : ulp * temp;
                    if (maxval <= thresh) {
                        /* Deflatable */
                        kwbot = kwbot - 2;
                    } else {
                        /* Not deflatable, move out of the way */
                        ifst = kwbot - kwtop;
                        ilst = k2;
                        stgexc(1, 1, jw, &A[kwtop + kwtop * lda], lda,
                               &B[kwtop + kwtop * ldb], ldb, QC, ldqc,
                               ZC, ldzc, &ifst, &ilst, work, lwork, &dtgexc_info);
                        k2 = k2 + 2;
                    }
                }
                k = k + 2;
            } else {
                /* Try to deflate real eigenvalue */
                temp = fabsf(A[kwbot + kwbot * lda]);
                if (temp == ZERO) {
                    temp = fabsf(s);
                }
                {
                    f32 tmp = fabsf(s * QC[0 + (kwbot - kwtop) * ldqc]);
                    f32 thresh = (ulp * temp > smlnum) ? ulp * temp : smlnum;
                    if (tmp <= thresh) {
                        /* Deflatable */
                        kwbot = kwbot - 1;
                    } else {
                        /* Not deflatable, move out of the way */
                        ifst = kwbot - kwtop;
                        ilst = k2;
                        stgexc(1, 1, jw, &A[kwtop + kwtop * lda], lda,
                               &B[kwtop + kwtop * ldb], ldb, QC, ldqc,
                               ZC, ldzc, &ifst, &ilst, work, lwork, &dtgexc_info);
                        k2 = k2 + 1;
                    }
                }
                k = k + 1;
            }
        }
    }

    /* Store eigenvalues */
    *nd = ihi - kwbot;
    *ns = jw - *nd;
    k = kwtop;
    while (k <= ihi) {
        bulge = 0;
        if (k < ihi) {
            if (A[(k + 1) + k * lda] != ZERO) {
                bulge = 1;
            }
        }
        if (bulge) {
            /* 2x2 eigenvalue block */
            slag2(&A[k + k * lda], lda, &B[k + k * ldb], ldb, safmin,
                  &beta[k], &beta[k + 1], &alphar[k], &alphar[k + 1], &alphai[k]);
            alphai[k + 1] = -alphai[k];
            k = k + 2;
        } else {
            /* 1x1 eigenvalue block */
            alphar[k] = A[k + k * lda];
            alphai[k] = ZERO;
            beta[k] = B[k + k * ldb];
            k = k + 1;
        }
    }

    if (kwtop != ilo && s != ZERO) {
        /* Reflect spike back, this will create optimally packed bulges */
        /* A(KWTOP:KWBOT, KWTOP-1) = A(KWTOP, KWTOP-1) * QC(1, 1:JW-ND) */
        {
            f32 a_val = A[kwtop + (kwtop - 1) * lda];
            for (int i = kwtop; i <= kwbot; i++) {
                A[i + (kwtop - 1) * lda] = a_val * QC[0 + (i - kwtop) * ldqc];
            }
        }
        for (k = kwbot - 1; k >= kwtop; k--) {
            slartg(A[k + (kwtop - 1) * lda], A[(k + 1) + (kwtop - 1) * lda],
                   &c1, &s1, &temp);
            A[k + (kwtop - 1) * lda] = temp;
            A[(k + 1) + (kwtop - 1) * lda] = ZERO;
            k2 = (kwtop > k - 1) ? kwtop : k - 1;
            cblas_srot(ihi - k2 + 1, &A[k + k2 * lda], lda, &A[(k + 1) + k2 * lda], lda, c1, s1);
            cblas_srot(ihi - (k - 1) + 1, &B[k + (k - 1) * ldb], ldb,
                       &B[(k + 1) + (k - 1) * ldb], ldb, c1, s1);
            cblas_srot(jw, &QC[0 + (k - kwtop) * ldqc], 1,
                       &QC[0 + (k + 1 - kwtop) * ldqc], 1, c1, s1);
        }

        /* Chase bulges down */
        istartm = kwtop;
        istopm = ihi;
        k = kwbot - 1;
        while (k >= kwtop) {
            if ((k >= kwtop + 1) && A[(k + 1) + (k - 1) * lda] != ZERO) {
                /* Move double pole block down and remove it */
                for (k2 = k - 1; k2 <= kwbot - 2; k2++) {
                    slaqz2(1, 1, k2, kwtop, kwtop + jw - 1, kwbot,
                           A, lda, B, ldb, jw, kwtop, QC, ldqc, jw, kwtop, ZC, ldzc);
                }
                k = k - 2;
            } else {
                /* k points to single shift */
                for (k2 = k; k2 <= kwbot - 2; k2++) {
                    /* Move shift down */
                    slartg(B[(k2 + 1) + (k2 + 1) * ldb], B[(k2 + 1) + k2 * ldb],
                           &c1, &s1, &temp);
                    B[(k2 + 1) + (k2 + 1) * ldb] = temp;
                    B[(k2 + 1) + k2 * ldb] = ZERO;
                    cblas_srot(k2 + 2 - istartm + 1, &A[istartm + (k2 + 1) * lda], 1,
                               &A[istartm + k2 * lda], 1, c1, s1);
                    cblas_srot(k2 - istartm + 1, &B[istartm + (k2 + 1) * ldb], 1,
                               &B[istartm + k2 * ldb], 1, c1, s1);
                    cblas_srot(jw, &ZC[0 + (k2 + 1 - kwtop) * ldzc], 1,
                               &ZC[0 + (k2 - kwtop) * ldzc], 1, c1, s1);

                    slartg(A[(k2 + 1) + k2 * lda], A[(k2 + 2) + k2 * lda],
                           &c1, &s1, &temp);
                    A[(k2 + 1) + k2 * lda] = temp;
                    A[(k2 + 2) + k2 * lda] = ZERO;
                    cblas_srot(istopm - k2, &A[(k2 + 1) + (k2 + 1) * lda], lda,
                               &A[(k2 + 2) + (k2 + 1) * lda], lda, c1, s1);
                    cblas_srot(istopm - k2, &B[(k2 + 1) + (k2 + 1) * ldb], ldb,
                               &B[(k2 + 2) + (k2 + 1) * ldb], ldb, c1, s1);
                    cblas_srot(jw, &QC[0 + (k2 + 1 - kwtop) * ldqc], 1,
                               &QC[0 + (k2 + 2 - kwtop) * ldqc], 1, c1, s1);
                }

                /* Remove the shift */
                slartg(B[kwbot + kwbot * ldb], B[kwbot + (kwbot - 1) * ldb],
                       &c1, &s1, &temp);
                B[kwbot + kwbot * ldb] = temp;
                B[kwbot + (kwbot - 1) * ldb] = ZERO;
                cblas_srot(kwbot - istartm, &B[istartm + kwbot * ldb], 1,
                           &B[istartm + (kwbot - 1) * ldb], 1, c1, s1);
                cblas_srot(kwbot - istartm + 1, &A[istartm + kwbot * lda], 1,
                           &A[istartm + (kwbot - 1) * lda], 1, c1, s1);
                cblas_srot(jw, &ZC[0 + (kwbot - kwtop) * ldzc], 1,
                           &ZC[0 + (kwbot - 1 - kwtop) * ldzc], 1, c1, s1);

                k = k - 1;
            }
        }
    }

    /* Apply Qc and Zc to rest of the matrix */
    if (ilschur) {
        istartm = 0;
        istopm = n - 1;
    } else {
        istartm = ilo;
        istopm = ihi;
    }

    if (istopm - ihi > 0) {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, jw, istopm - ihi, jw,
                    ONE, QC, ldqc, &A[kwtop + (ihi + 1) * lda], lda, ZERO, work, jw);
        slacpy("A", jw, istopm - ihi, work, jw, &A[kwtop + (ihi + 1) * lda], lda);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, jw, istopm - ihi, jw,
                    ONE, QC, ldqc, &B[kwtop + (ihi + 1) * ldb], ldb, ZERO, work, jw);
        slacpy("A", jw, istopm - ihi, work, jw, &B[kwtop + (ihi + 1) * ldb], ldb);
    }
    if (ilq) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, jw, jw,
                    ONE, &Q[0 + kwtop * ldq], ldq, QC, ldqc, ZERO, work, n);
        slacpy("A", n, jw, work, n, &Q[0 + kwtop * ldq], ldq);
    }

    if (kwtop - istartm > 0) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, kwtop - istartm, jw, jw,
                    ONE, &A[istartm + kwtop * lda], lda, ZC, ldzc, ZERO, work, kwtop - istartm);
        slacpy("A", kwtop - istartm, jw, work, kwtop - istartm, &A[istartm + kwtop * lda], lda);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, kwtop - istartm, jw, jw,
                    ONE, &B[istartm + kwtop * ldb], ldb, ZC, ldzc, ZERO, work, kwtop - istartm);
        slacpy("A", kwtop - istartm, jw, work, kwtop - istartm, &B[istartm + kwtop * ldb], ldb);
    }
    if (ilz) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, jw, jw,
                    ONE, &Z[0 + kwtop * ldz], ldz, ZC, ldzc, ZERO, work, n);
        slacpy("A", n, jw, work, n, &Z[0 + kwtop * ldz], ldz);
    }
}
