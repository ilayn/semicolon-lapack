/**
 * @file zlaqz2.c
 * @brief ZLAQZ2 performs AED.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZLAQZ2 performs AED
 *
 * @param[in]     ilschur  Determines whether or not to update the full Schur form.
 * @param[in]     ilq      Determines whether or not to update the matrix Q.
 * @param[in]     ilz      Determines whether or not to update the matrix Z.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      0-based lower bound of active submatrix.
 * @param[in]     ihi      0-based upper bound of active submatrix.
 * @param[in]     nw       The desired size of the deflation window.
 * @param[in,out] A        Complex array, dimension (lda, n).
 * @param[in]     lda      The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B        Complex array, dimension (ldb, n).
 * @param[in]     ldb      The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q        Complex array, dimension (ldq, n).
 * @param[in]     ldq      The leading dimension of Q.
 * @param[in,out] Z        Complex array, dimension (ldz, n).
 * @param[in]     ldz      The leading dimension of Z.
 * @param[out]    ns       The number of unconverged eigenvalues available as shifts.
 * @param[out]    nd       The number of converged eigenvalues found.
 * @param[out]    alpha    Complex array, dimension (n). Eigenvalue numerators.
 * @param[out]    beta     Complex array, dimension (n). Eigenvalue denominators.
 * @param[in,out] QC       Complex array, dimension (ldqc, nw).
 * @param[in]     ldqc     The leading dimension of QC.
 * @param[in,out] ZC       Complex array, dimension (ldzc, nw).
 * @param[in]     ldzc     The leading dimension of ZC.
 * @param[out]    work     Complex array, dimension (max(1, lwork)).
 *                         On exit, if info >= 0, work[0] returns the optimal lwork.
 * @param[in]     lwork    The dimension of the array work.
 *                         If lwork = -1, workspace query.
 * @param[out]    rwork    Double precision array, dimension (n).
 * @param[in]     rec      Current recursion level. Should be set to 0 on first call.
 * @param[out]    info     = 0: successful exit.
 *                         < 0: if info = -i, the i-th argument had an illegal value.
 */
void zlaqz2(const INT ilschur, const INT ilq, const INT ilz,
            const INT n, const INT ilo, const INT ihi, const INT nw,
            c128* restrict A, const INT lda,
            c128* restrict B, const INT ldb,
            c128* restrict Q, const INT ldq,
            c128* restrict Z, const INT ldz,
            INT* ns, INT* nd,
            c128* restrict alpha,
            c128* restrict beta,
            c128* restrict QC, const INT ldqc,
            c128* restrict ZC, const INT ldzc,
            c128* restrict work, const INT lwork,
            f64* restrict rwork, const INT rec, INT* info)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ZERO = 0.0;

    INT jw, kwtop, kwbot, istopm, istartm, k, k2, ztgexc_info,
        ifst, ilst, lworkreq, qz_small_info;
    f64 smlnum, ulp, safmin, c1, tempr;
    c128 s, s1, temp;

    *info = 0;

    /* Set up deflation window */
    jw = (nw < ihi - ilo + 1) ? nw : ihi - ilo + 1;
    kwtop = ihi - jw + 1;
    if (kwtop == ilo) {
        s = CZERO;
    } else {
        s = A[kwtop + (kwtop - 1) * lda];
    }

    /* Determine required workspace */
    ilst = jw - 1;
    zlaqz0("S", "V", "V", jw, 0, jw - 1, &A[kwtop + kwtop * lda], lda,
           &B[kwtop + kwtop * ldb], ldb, alpha, beta, QC, ldqc, ZC,
           ldzc, work, -1, rwork, rec + 1, &qz_small_info);
    lworkreq = (INT)creal(work[0]) + 2 * jw * jw;
    {
        INT t1 = n * nw;
        INT t2 = 2 * nw * nw + n;
        if (t1 > lworkreq) lworkreq = t1;
        if (t2 > lworkreq) lworkreq = t2;
    }
    if (lwork == -1) {
        /* workspace query, quick return */
        work[0] = CMPLX((f64)lworkreq, 0.0);
        return;
    } else if (lwork < lworkreq) {
        *info = -26;
    }

    if (*info != 0) {
        xerbla("ZLAQZ2", -(*info));
        return;
    }

    /* Get machine constants */
    safmin = dlamch("S");
    ulp = dlamch("P");
    smlnum = safmin * ((f64)n / ulp);

    if (ihi == kwtop) {
        /* 1 by 1 deflation window, just try a regular deflation */
        alpha[kwtop] = A[kwtop + kwtop * lda];
        beta[kwtop] = B[kwtop + kwtop * ldb];
        *ns = 1;
        *nd = 0;
        if (cabs(s) <= fmax(smlnum, ulp * cabs(A[kwtop + kwtop * lda]))) {
            *ns = 0;
            *nd = 1;
            if (kwtop > ilo) {
                A[kwtop + (kwtop - 1) * lda] = CZERO;
            }
        }
    }

    /* Store window in case of convergence failure */
    zlacpy("A", jw, jw, &A[kwtop + kwtop * lda], lda, work, jw);
    zlacpy("A", jw, jw, &B[kwtop + kwtop * ldb], ldb,
           &work[jw * jw], jw);

    /* Transform window to real schur form */
    zlaset("F", jw, jw, CZERO, CONE, QC, ldqc);
    zlaset("F", jw, jw, CZERO, CONE, ZC, ldzc);
    zlaqz0("S", "V", "V", jw, 0, jw - 1, &A[kwtop + kwtop * lda], lda,
           &B[kwtop + kwtop * ldb], ldb, alpha, beta, QC, ldqc, ZC,
           ldzc, &work[2 * jw * jw], lwork - 2 * jw * jw, rwork,
           rec + 1, &qz_small_info);

    if (qz_small_info != 0) {
        /* Convergence failure, restore the window and exit */
        *nd = 0;
        *ns = jw - qz_small_info;
        zlacpy("A", jw, jw, work, jw, &A[kwtop + kwtop * lda], lda);
        zlacpy("A", jw, jw, &work[jw * jw], jw,
               &B[kwtop + kwtop * ldb], ldb);
        return;
    }

    /* Deflation detection loop */
    if (kwtop == ilo || s == CZERO) {
        kwbot = kwtop - 1;
    } else {
        kwbot = ihi;
        k = 0;
        k2 = 0;
        while (k < jw) {
            /* Try to deflate eigenvalue */
            tempr = cabs(A[kwbot + kwbot * lda]);
            if (tempr == ZERO) {
                tempr = cabs(s);
            }
            if (cabs(s * QC[0 + (kwbot - kwtop) * ldqc]) <=
                fmax(ulp * tempr, smlnum)) {
                /* Deflatable */
                kwbot = kwbot - 1;
            } else {
                /* Not deflatable, move out of the way */
                ifst = kwbot - kwtop;
                ilst = k2;
                ztgexc(1, 1, jw, &A[kwtop + kwtop * lda],
                       lda, &B[kwtop + kwtop * ldb], ldb, QC, ldqc,
                       ZC, ldzc, ifst, &ilst, &ztgexc_info);
                k2 = k2 + 1;
            }

            k = k + 1;
        }
    }

    /* Store eigenvalues */
    *nd = ihi - kwbot;
    *ns = jw - *nd;
    k = kwtop;
    while (k <= ihi) {
        alpha[k] = A[k + k * lda];
        beta[k] = B[k + k * ldb];
        k = k + 1;
    }

    if (kwtop != ilo && s != CZERO) {
        /* Reflect spike back, this will create optimally packed bulges */
        for (INT i = kwtop; i <= kwbot; i++) {
            A[i + (kwtop - 1) * lda] = A[kwtop + (kwtop - 1) * lda] *
                                        conj(QC[0 + (i - kwtop) * ldqc]);
        }
        for (k = kwbot - 1; k >= kwtop; k--) {
            zlartg(A[k + (kwtop - 1) * lda], A[k + 1 + (kwtop - 1) * lda],
                   &c1, &s1, &temp);
            A[k + (kwtop - 1) * lda] = temp;
            A[k + 1 + (kwtop - 1) * lda] = CZERO;
            k2 = (kwtop > k - 1) ? kwtop : k - 1;
            zrot(ihi - k2 + 1, &A[k + k2 * lda], lda,
                 &A[k + 1 + k2 * lda], lda, c1, s1);
            zrot(ihi - k + 2, &B[k + (k - 1) * ldb], ldb,
                 &B[k + 1 + (k - 1) * ldb], ldb, c1, s1);
            zrot(jw, &QC[0 + (k - kwtop) * ldqc], 1,
                 &QC[0 + (k + 1 - kwtop) * ldqc], 1, c1, conj(s1));
        }

        /* Chase bulges down */
        k = kwbot - 1;
        while (k >= kwtop) {

            /* Move bulge down and remove it */
            for (k2 = k; k2 <= kwbot - 1; k2++) {
                zlaqz1(1, 1, k2, kwtop, kwtop + jw - 1,
                        kwbot, A, lda, B, ldb, jw, kwtop, QC, ldqc,
                        jw, kwtop, ZC, ldzc);
            }

            k = k - 1;
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
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    jw, istopm - ihi, jw, &CONE, QC, ldqc,
                    &A[kwtop + (ihi + 1) * lda], lda, &CZERO, work, jw);
        zlacpy("A", jw, istopm - ihi, work, jw,
               &A[kwtop + (ihi + 1) * lda], lda);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    jw, istopm - ihi, jw, &CONE, QC, ldqc,
                    &B[kwtop + (ihi + 1) * ldb], ldb, &CZERO, work, jw);
        zlacpy("A", jw, istopm - ihi, work, jw,
               &B[kwtop + (ihi + 1) * ldb], ldb);
    }
    if (ilq) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, jw, jw, &CONE, &Q[0 + kwtop * ldq], ldq, QC,
                    ldqc, &CZERO, work, n);
        zlacpy("A", n, jw, work, n, &Q[0 + kwtop * ldq], ldq);
    }

    if (kwtop - istartm > 0) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    kwtop - istartm, jw, jw, &CONE,
                    &A[istartm + kwtop * lda], lda, ZC, ldzc, &CZERO, work,
                    kwtop - istartm);
        zlacpy("A", kwtop - istartm, jw, work, kwtop - istartm,
               &A[istartm + kwtop * lda], lda);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    kwtop - istartm, jw, jw, &CONE,
                    &B[istartm + kwtop * ldb], ldb, ZC, ldzc, &CZERO, work,
                    kwtop - istartm);
        zlacpy("A", kwtop - istartm, jw, work, kwtop - istartm,
               &B[istartm + kwtop * ldb], ldb);
    }
    if (ilz) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, jw, jw, &CONE, &Z[0 + kwtop * ldz], ldz, ZC,
                    ldzc, &CZERO, work, n);
        zlacpy("A", n, jw, work, n, &Z[0 + kwtop * ldz], ldz);
    }
}
