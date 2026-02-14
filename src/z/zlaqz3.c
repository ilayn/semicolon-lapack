/**
 * @file zlaqz3.c
 * @brief ZLAQZ3 executes a single multishift QZ sweep.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQZ3 executes a single multishift QZ sweep.
 *
 * @param[in]     ilschur  Determines whether or not to update the full Schur form.
 * @param[in]     ilq      Determines whether or not to update the matrix Q.
 * @param[in]     ilz      Determines whether or not to update the matrix Z.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      0-based lower bound of active submatrix.
 * @param[in]     ihi      0-based upper bound of active submatrix.
 * @param[in]     nshifts  The desired number of shifts to use.
 * @param[in]     nblock_desired  The desired size of the computational windows.
 * @param[in,out] alpha    Complex array. Alpha parts of the shifts.
 * @param[in,out] beta     Complex array. Beta parts of the shifts.
 * @param[in,out] A        Complex array, dimension (lda, n).
 * @param[in]     lda      Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B        Complex array, dimension (ldb, n).
 * @param[in]     ldb      Leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q        Complex array, dimension (ldq, n).
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in,out] Z        Complex array, dimension (ldz, n).
 * @param[in]     ldz      Leading dimension of Z.
 * @param[in,out] QC       Complex array, dimension (ldqc, nblock_desired).
 * @param[in]     ldqc     Leading dimension of QC.
 * @param[in,out] ZC       Complex array, dimension (ldzc, nblock_desired).
 * @param[in]     ldzc     Leading dimension of ZC.
 * @param[out]    work     Complex array, dimension (max(1, lwork)).
 * @param[in]     lwork    Dimension of work. If lwork = -1, workspace query.
 * @param[out]    info     = 0: successful exit.
 */
void zlaqz3(
    const int ilschur,
    const int ilq,
    const int ilz,
    const int n,
    const int ilo,
    const int ihi,
    const int nshifts,
    const int nblock_desired,
    c128* const restrict alpha,
    c128* const restrict beta,
    c128* const restrict A,
    const int lda,
    c128* const restrict B,
    const int ldb,
    c128* const restrict Q,
    const int ldq,
    c128* const restrict Z,
    const int ldz,
    c128* const restrict QC,
    const int ldqc,
    c128* const restrict ZC,
    const int ldzc,
    c128* const restrict work,
    const int lwork,
    int* info)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ONE = 1.0;

    int i, j, ns, istartm, istopm, sheight, swidth, k, np;
    int istartb, istopb, ishift, nblock, npos;
    f64 safmin, safmax, c, scale;
    c128 s, temp, temp2, temp3;

    *info = 0;
    if (nblock_desired < nshifts + 1) {
        *info = -8;
    }
    if (lwork == -1) {
        /* workspace query, quick return */
        work[0] = CMPLX((f64)(n * nblock_desired), 0.0);
        return;
    } else if (lwork < n * nblock_desired) {
        *info = -25;
    }

    if (*info != 0) {
        xerbla("ZLAQZ3", -(*info));
        return;
    }

    /* Get machine constants */
    safmin = dlamch("S");
    safmax = ONE / safmin;

    if (ilo >= ihi) {
        return;
    }

    if (ilschur) {
        istartm = 0;
        istopm = n - 1;
    } else {
        istartm = ilo;
        istopm = ihi;
    }

    ns = nshifts;
    npos = (nblock_desired - ns > 1) ? nblock_desired - ns : 1;

    /* The following block introduces the shifts and chases
       them down one by one just enough to make space for
       the other shifts. The near-the-diagonal block is
       of size (ns+1) x ns. */

    zlaset("F", ns + 1, ns + 1, CZERO, CONE, QC, ldqc);
    zlaset("F", ns, ns, CZERO, CONE, ZC, ldzc);

    for (i = 0; i < ns; i++) {
        /* Introduce the shift */
        scale = sqrt(cabs(alpha[i])) * sqrt(cabs(beta[i]));
        if (scale >= safmin && scale <= safmax) {
            alpha[i] = alpha[i] / scale;
            beta[i] = beta[i] / scale;
        }

        temp2 = beta[i] * A[ilo + ilo * lda] - alpha[i] * B[ilo + ilo * ldb];
        temp3 = beta[i] * A[(ilo + 1) + ilo * lda];

        if (cabs(temp2) > safmax || cabs(temp3) > safmax) {
            temp2 = CONE;
            temp3 = CZERO;
        }

        zlartg(temp2, temp3, &c, &s, &temp);
        zrot(ns, &A[ilo + ilo * lda], lda, &A[(ilo + 1) + ilo * lda], lda, c, s);
        zrot(ns, &B[ilo + ilo * ldb], ldb, &B[(ilo + 1) + ilo * ldb], ldb, c, s);
        zrot(ns + 1, &QC[0], 1, &QC[ldqc], 1, c, conj(s));

        /* Chase the shift down */
        for (j = 0; j < ns - 1 - i; j++) {
            zlaqz1(1, 1, j, 0, ns - 1, ihi - ilo,
                    &A[ilo + ilo * lda], lda, &B[ilo + ilo * ldb], ldb,
                    ns + 1, 0, QC, ldqc, ns, 0, ZC, ldzc);
        }
    }

    /* Update the rest of the pencil */

    /* Update A(ilo:ilo+ns,ilo+ns:istopm) and B(ilo:ilo+ns,ilo+ns:istopm)
       from the left with Qc(1:ns+1,1:ns+1)' */
    sheight = ns + 1;
    swidth = istopm - (ilo + ns) + 1;
    if (swidth > 0) {
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                    &CONE, QC, ldqc, &A[ilo + (ilo + ns) * lda], lda, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &A[ilo + (ilo + ns) * lda], lda);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                    &CONE, QC, ldqc, &B[ilo + (ilo + ns) * ldb], ldb, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &B[ilo + (ilo + ns) * ldb], ldb);
    }
    if (ilq) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, sheight, sheight,
                    &CONE, &Q[0 + ilo * ldq], ldq, QC, ldqc, &CZERO, work, n);
        zlacpy("A", n, sheight, work, n, &Q[0 + ilo * ldq], ldq);
    }

    /* Update A(istartm:ilo-1,ilo:ilo+ns-1) and B(istartm:ilo-1,ilo:ilo+ns-1)
       from the right with Zc(1:ns,1:ns) */
    sheight = ilo - istartm;
    swidth = ns;
    if (sheight > 0) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    &CONE, &A[istartm + ilo * lda], lda, ZC, ldzc, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &A[istartm + ilo * lda], lda);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    &CONE, &B[istartm + ilo * ldb], ldb, ZC, ldzc, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &B[istartm + ilo * ldb], ldb);
    }
    if (ilz) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, swidth, swidth,
                    &CONE, &Z[0 + ilo * ldz], ldz, ZC, ldzc, &CZERO, work, n);
        zlacpy("A", n, swidth, work, n, &Z[0 + ilo * ldz], ldz);
    }

    /* The following block chases the shifts down to the bottom
       right block. If possible, a shift is moved down npos
       positions at a time */

    k = ilo;
    while (k < ihi - ns) {
        np = ((ihi - ns - k) < npos) ? (ihi - ns - k) : npos;
        /* Size of the near-the-diagonal block */
        nblock = ns + np;
        /* istartb points to the first row we will be updating */
        istartb = k + 1;
        /* istopb points to the last column we will be updating */
        istopb = k + nblock - 1;

        zlaset("F", ns + np, ns + np, CZERO, CONE, QC, ldqc);
        zlaset("F", ns + np, ns + np, CZERO, CONE, ZC, ldzc);

        /* Near the diagonal shift chase */
        for (i = ns - 1; i >= 0; i--) {
            for (j = 0; j <= np - 1; j++) {
                zlaqz1(1, 1, k + i + j, istartb, istopb, ihi,
                        A, lda, B, ldb, nblock, k + 1, QC, ldqc,
                        nblock, k, ZC, ldzc);
            }
        }

        /* Update rest of the pencil */

        /* Update A(k+1:k+ns+np, k+ns+np:istopm) and
           B(k+1:k+ns+np, k+ns+np:istopm)
           from the left with Qc(1:ns+np,1:ns+np)' */
        sheight = ns + np;
        swidth = istopm - (k + ns + np) + 1;
        if (swidth > 0) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                        &CONE, QC, ldqc, &A[(k + 1) + (k + ns + np) * lda], lda, &CZERO, work, sheight);
            zlacpy("A", sheight, swidth, work, sheight, &A[(k + 1) + (k + ns + np) * lda], lda);
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                        &CONE, QC, ldqc, &B[(k + 1) + (k + ns + np) * ldb], ldb, &CZERO, work, sheight);
            zlacpy("A", sheight, swidth, work, sheight, &B[(k + 1) + (k + ns + np) * ldb], ldb);
        }
        if (ilq) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nblock, nblock,
                        &CONE, &Q[0 + (k + 1) * ldq], ldq, QC, ldqc, &CZERO, work, n);
            zlacpy("A", n, nblock, work, n, &Q[0 + (k + 1) * ldq], ldq);
        }

        /* Update A(istartm:k,k:k+ns+npos-1) and B(istartm:k,k:k+ns+npos-1)
           from the right with Zc(1:ns+np,1:ns+np) */
        sheight = k - istartm + 1;
        swidth = nblock;
        if (sheight > 0) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                        &CONE, &A[istartm + k * lda], lda, ZC, ldzc, &CZERO, work, sheight);
            zlacpy("A", sheight, swidth, work, sheight, &A[istartm + k * lda], lda);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                        &CONE, &B[istartm + k * ldb], ldb, ZC, ldzc, &CZERO, work, sheight);
            zlacpy("A", sheight, swidth, work, sheight, &B[istartm + k * ldb], ldb);
        }
        if (ilz) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nblock, nblock,
                        &CONE, &Z[0 + k * ldz], ldz, ZC, ldzc, &CZERO, work, n);
            zlacpy("A", n, nblock, work, n, &Z[0 + k * ldz], ldz);
        }

        k = k + np;
    }

    /* The following block removes the shifts from the bottom right corner
       one by one. Updates are initially applied to A(ihi-ns+1:ihi,ihi-ns:ihi). */

    zlaset("F", ns, ns, CZERO, CONE, QC, ldqc);
    zlaset("F", ns + 1, ns + 1, CZERO, CONE, ZC, ldzc);

    /* istartb points to the first row we will be updating */
    istartb = ihi - ns + 1;
    /* istopb points to the last column we will be updating */
    istopb = ihi;

    for (i = 0; i < ns; i++) {
        /* Chase the shift down to the bottom right corner */
        for (ishift = ihi - i - 1; ishift <= ihi - 1; ishift++) {
            zlaqz1(1, 1, ishift, istartb, istopb, ihi,
                    A, lda, B, ldb, ns, ihi - ns + 1, QC, ldqc,
                    ns + 1, ihi - ns, ZC, ldzc);
        }
    }

    /* Update rest of the pencil */

    /* Update A(ihi-ns+1:ihi, ihi+1:istopm) from the left with Qc(1:ns,1:ns)' */
    sheight = ns;
    swidth = istopm - ihi;
    if (swidth > 0) {
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                    &CONE, QC, ldqc, &A[(ihi - ns + 1) + (ihi + 1) * lda], lda, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &A[(ihi - ns + 1) + (ihi + 1) * lda], lda);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sheight, swidth, sheight,
                    &CONE, QC, ldqc, &B[(ihi - ns + 1) + (ihi + 1) * ldb], ldb, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &B[(ihi - ns + 1) + (ihi + 1) * ldb], ldb);
    }
    if (ilq) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, ns, ns,
                    &CONE, &Q[0 + (ihi - ns + 1) * ldq], ldq, QC, ldqc, &CZERO, work, n);
        zlacpy("A", n, ns, work, n, &Q[0 + (ihi - ns + 1) * ldq], ldq);
    }

    /* Update A(istartm:ihi-ns,ihi-ns:ihi) from the right with Zc(1:ns+1,1:ns+1) */
    sheight = ihi - ns - istartm + 1;
    swidth = ns + 1;
    if (sheight > 0) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    &CONE, &A[istartm + (ihi - ns) * lda], lda, ZC, ldzc, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &A[istartm + (ihi - ns) * lda], lda);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    &CONE, &B[istartm + (ihi - ns) * ldb], ldb, ZC, ldzc, &CZERO, work, sheight);
        zlacpy("A", sheight, swidth, work, sheight, &B[istartm + (ihi - ns) * ldb], ldb);
    }
    if (ilz) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, ns + 1, ns + 1,
                    &CONE, &Z[0 + (ihi - ns) * ldz], ldz, ZC, ldzc, &CZERO, work, n);
        zlacpy("A", n, ns + 1, work, n, &Z[0 + (ihi - ns) * ldz], ldz);
    }
}
