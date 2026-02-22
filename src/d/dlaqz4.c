/**
 * @file dlaqz4.c
 * @brief DLAQZ4 executes a single multishift QZ sweep.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAQZ4 executes a single multishift QZ sweep.
 *
 * @param[in]     ilschur  Determines whether or not to update the full Schur form.
 * @param[in]     ilq      Determines whether or not to update the matrix Q.
 * @param[in]     ilz      Determines whether or not to update the matrix Z.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      Lower bound of active submatrix (0-based).
 * @param[in]     ihi      Upper bound of active submatrix (0-based).
 * @param[in]     nshifts  The desired number of shifts to use.
 * @param[in]     nblock_desired  The desired size of the computational windows.
 * @param[in,out] sr       Real parts of the shifts to use.
 * @param[in,out] si       Imaginary parts of the shifts to use.
 * @param[in,out] ss       Scale of the shifts to use.
 * @param[in,out] A        Matrix A.
 * @param[in]     lda      Leading dimension of A.
 * @param[in,out] B        Matrix B.
 * @param[in]     ldb      Leading dimension of B.
 * @param[in,out] Q        Matrix Q.
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in,out] Z        Matrix Z.
 * @param[in]     ldz      Leading dimension of Z.
 * @param[in,out] QC       Workspace for accumulated Q transformations.
 * @param[in]     ldqc     Leading dimension of QC.
 * @param[in,out] ZC       Workspace for accumulated Z transformations.
 * @param[in]     ldzc     Leading dimension of ZC.
 * @param[out]    work     Workspace array.
 * @param[in]     lwork    Dimension of workspace. If lwork = -1, workspace query.
 * @param[out]    info
 *                         - = 0: successful exit, < 0: illegal argument.
 */
void dlaqz4(
    const INT ilschur,
    const INT ilq,
    const INT ilz,
    const INT n,
    const INT ilo,
    const INT ihi,
    const INT nshifts,
    const INT nblock_desired,
    f64* restrict sr,
    f64* restrict si,
    f64* restrict ss,
    f64* restrict A,
    const INT lda,
    f64* restrict B,
    const INT ldb,
    f64* restrict Q,
    const INT ldq,
    f64* restrict Z,
    const INT ldz,
    f64* restrict QC,
    const INT ldqc,
    f64* restrict ZC,
    const INT ldzc,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, ns, istartm, istopm, sheight, swidth, k, np;
    INT istartb, istopb, ishift, nblock, npos;
    f64 temp, v[3], c1, s1, c2, s2, swap;

    *info = 0;
    if (nblock_desired < nshifts + 1) {
        *info = -8;
    }
    if (lwork == -1) {
        /* workspace query, quick return */
        work[0] = (f64)(n * nblock_desired);
        return;
    } else if (lwork < n * nblock_desired) {
        *info = -25;
    }

    if (*info != 0) {
        xerbla("DLAQZ4", -(*info));
        return;
    }

    /* Executable statements */

    if (nshifts < 2) {
        return;
    }

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

    /* Shuffle shifts into pairs of real shifts and pairs
       of complex conjugate shifts assuming complex
       conjugate shifts are already adjacent to one another */
    for (i = 0; i <= nshifts - 3; i += 2) {
        if (si[i] != -si[i + 1]) {
            swap = sr[i];
            sr[i] = sr[i + 1];
            sr[i + 1] = sr[i + 2];
            sr[i + 2] = swap;

            swap = si[i];
            si[i] = si[i + 1];
            si[i + 1] = si[i + 2];
            si[i + 2] = swap;

            swap = ss[i];
            ss[i] = ss[i + 1];
            ss[i + 1] = ss[i + 2];
            ss[i + 2] = swap;
        }
    }

    /* NSHFTS is supposed to be even, but if it is odd,
       then simply reduce it by one. The shuffle above
       ensures that the dropped shift is real and that
       the remaining shifts are paired. */
    ns = nshifts - (nshifts % 2);
    npos = (nblock_desired - ns > 1) ? nblock_desired - ns : 1;

    /* The following block introduces the shifts and chases
       them down one by one just enough to make space for
       the other shifts. The near-the-diagonal block is
       of size (ns+1) x ns. */
    dlaset("F", ns + 1, ns + 1, ZERO, ONE, QC, ldqc);
    dlaset("F", ns, ns, ZERO, ONE, ZC, ldzc);

    for (i = 0; i <= ns - 1; i += 2) {
        /* Introduce the shift */
        dlaqz1(&A[ilo + ilo * lda], lda, &B[ilo + ilo * ldb], ldb,
               sr[i], sr[i + 1], si[i], ss[i], ss[i + 1], v);

        temp = v[1];
        dlartg(temp, v[2], &c1, &s1, &v[1]);
        dlartg(v[0], v[1], &c2, &s2, &temp);

        cblas_drot(ns, &A[(ilo + 1) + ilo * lda], lda, &A[(ilo + 2) + ilo * lda], lda, c1, s1);
        cblas_drot(ns, &A[ilo + ilo * lda], lda, &A[(ilo + 1) + ilo * lda], lda, c2, s2);
        cblas_drot(ns, &B[(ilo + 1) + ilo * ldb], ldb, &B[(ilo + 2) + ilo * ldb], ldb, c1, s1);
        cblas_drot(ns, &B[ilo + ilo * ldb], ldb, &B[(ilo + 1) + ilo * ldb], ldb, c2, s2);
        cblas_drot(ns + 1, &QC[0 + 1 * ldqc], 1, &QC[0 + 2 * ldqc], 1, c1, s1);
        cblas_drot(ns + 1, &QC[0 + 0 * ldqc], 1, &QC[0 + 1 * ldqc], 1, c2, s2);

        /* Chase the shift down */
        for (j = 0; j <= ns - 2 - i; j++) {
            /* Fortran: DLAQZ2(.TRUE., .TRUE., J, 1, NS, IHI-ILO+1, A(ILO,ILO), ...) */
            /* j is 0-based in C, maps to Fortran J which is 1-based */
            /* In Fortran: J goes 1 to NS-1-I, in C: j goes 0 to NS-2-I */
            /* The k parameter to dlaqz2 in Fortran is J (1-based), in C it's j (0-based) */
            dlaqz2(1, 1, j, 0, ns - 1, ihi - ilo,
                   &A[ilo + ilo * lda], lda, &B[ilo + ilo * ldb], ldb,
                   ns + 1, 0, QC, ldqc, ns, 0, ZC, ldzc);
        }
    }

    /* Update the rest of the pencil */

    /* Update A(ilo:ilo+ns, ilo+ns:istopm) and B(ilo:ilo+ns, ilo+ns:istopm)
       from the left with Qc(1:ns+1,1:ns+1)' */
    sheight = ns + 1;
    swidth = istopm - (ilo + ns) + 1;
    if (swidth > 0) {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                    ONE, QC, ldqc, &A[ilo + (ilo + ns) * lda], lda, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &A[ilo + (ilo + ns) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                    ONE, QC, ldqc, &B[ilo + (ilo + ns) * ldb], ldb, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &B[ilo + (ilo + ns) * ldb], ldb);
    }
    if (ilq) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, sheight, sheight,
                    ONE, &Q[0 + ilo * ldq], ldq, QC, ldqc, ZERO, work, n);
        dlacpy("A", n, sheight, work, n, &Q[0 + ilo * ldq], ldq);
    }

    /* Update A(istartm:ilo-1, ilo:ilo+ns-1) and B(istartm:ilo-1, ilo:ilo+ns-1)
       from the right with Zc(1:ns,1:ns) */
    sheight = ilo - istartm;
    swidth = ns;
    if (sheight > 0) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    ONE, &A[istartm + ilo * lda], lda, ZC, ldzc, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &A[istartm + ilo * lda], lda);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    ONE, &B[istartm + ilo * ldb], ldb, ZC, ldzc, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &B[istartm + ilo * ldb], ldb);
    }
    if (ilz) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, swidth, swidth,
                    ONE, &Z[0 + ilo * ldz], ldz, ZC, ldzc, ZERO, work, n);
        dlacpy("A", n, swidth, work, n, &Z[0 + ilo * ldz], ldz);
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

        dlaset("F", ns + np, ns + np, ZERO, ONE, QC, ldqc);
        dlaset("F", ns + np, ns + np, ZERO, ONE, ZC, ldzc);

        /* Near the diagonal shift chase */
        for (i = ns - 1; i >= 0; i -= 2) {
            for (j = 0; j <= np - 1; j++) {
                /* Move down the block with index k+i+j-1, updating
                   the (ns+np x ns+np) block: (k:k+ns+np, k:k+ns+np-1)
                   Fortran: DLAQZ2(.TRUE., .TRUE., K+I+J-1, ISTARTB, ISTOPB, IHI, ...)
                   where K, I, J are all 1-based
                   In C: k, i, j are 0-based equivalents
                   k_fortran = k + 1, i_fortran = i + 1 (but Fortran loop goes NS-1 down to 0)
                   Actually in Fortran: I goes from NS-1 to 0 (by -2), J goes from 0 to NP-1
                   The index K+I+J-1 in Fortran (1-based K,I,J):
                   - K is 1-based loop variable starting at ILO
                   - I is loop counter NS-1, NS-3, ..., 1 (or 0)
                   - J is loop counter 0, 1, ..., NP-1
                   In C with 0-based k: Fortran K = C k + 1
                   Fortran K+I+J-1 = (k+1)+I+J-1 = k+I+J
                   But we also need to convert I,J. Looking at Fortran loop:
                   I goes NS-1, NS-3, ... which is the same values in C
                   J goes 0 to NP-1 which is the same
                   So Fortran index K+I+J-1 with Fortran K = k_c+1 gives us k_c+I+J
                   But wait, I need to be more careful about what dlaqz2 expects.
                   dlaqz2's k parameter is the bulge position (0-based in C).
                   So we pass k + i + j directly (all 0-based). */
                dlaqz2(1, 1, k + i + j, istartb, istopb, ihi,
                       A, lda, B, ldb, nblock, k + 1, QC, ldqc,
                       nblock, k, ZC, ldzc);
            }
        }

        /* Update rest of the pencil */

        /* Update A(k+1:k+ns+np, k+ns+np:istopm) and B(k+1:k+ns+np, k+ns+np:istopm)
           from the left with Qc(1:ns+np,1:ns+np)' */
        sheight = ns + np;
        swidth = istopm - (k + ns + np) + 1;
        if (swidth > 0) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                        ONE, QC, ldqc, &A[(k + 1) + (k + ns + np) * lda], lda, ZERO, work, sheight);
            dlacpy("A", sheight, swidth, work, sheight, &A[(k + 1) + (k + ns + np) * lda], lda);
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                        ONE, QC, ldqc, &B[(k + 1) + (k + ns + np) * ldb], ldb, ZERO, work, sheight);
            dlacpy("A", sheight, swidth, work, sheight, &B[(k + 1) + (k + ns + np) * ldb], ldb);
        }
        if (ilq) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nblock, nblock,
                        ONE, &Q[0 + (k + 1) * ldq], ldq, QC, ldqc, ZERO, work, n);
            dlacpy("A", n, nblock, work, n, &Q[0 + (k + 1) * ldq], ldq);
        }

        /* Update A(istartm:k, k:k+ns+npos-1) and B(istartm:k, k:k+ns+npos-1)
           from the right with Zc(1:ns+np,1:ns+np) */
        sheight = k - istartm + 1;
        swidth = nblock;
        if (sheight > 0) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                        ONE, &A[istartm + k * lda], lda, ZC, ldzc, ZERO, work, sheight);
            dlacpy("A", sheight, swidth, work, sheight, &A[istartm + k * lda], lda);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                        ONE, &B[istartm + k * ldb], ldb, ZC, ldzc, ZERO, work, sheight);
            dlacpy("A", sheight, swidth, work, sheight, &B[istartm + k * ldb], ldb);
        }
        if (ilz) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nblock, nblock,
                        ONE, &Z[0 + k * ldz], ldz, ZC, ldzc, ZERO, work, n);
            dlacpy("A", n, nblock, work, n, &Z[0 + k * ldz], ldz);
        }

        k = k + np;
    }

    /* The following block removes the shifts from the bottom right corner
       one by one. Updates are initially applied to A(ihi-ns+1:ihi, ihi-ns:ihi). */
    dlaset("F", ns, ns, ZERO, ONE, QC, ldqc);
    dlaset("F", ns + 1, ns + 1, ZERO, ONE, ZC, ldzc);

    /* istartb points to the first row we will be updating */
    istartb = ihi - ns + 1;
    /* istopb points to the last column we will be updating */
    istopb = ihi;

    for (i = 0; i <= ns - 1; i += 2) {
        /* Chase the shift down to the bottom right corner */
        for (ishift = ihi - i - 2; ishift <= ihi - 2; ishift++) {
            /* Fortran: DO ISHIFT = IHI-I-1, IHI-2 (1-based IHI, I goes 1,3,5,...)
               C: ihi is 0-based, i goes 0,2,4,...
               Fortran ISHIFT range: IHI-I-1 to IHI-2
               With Fortran IHI = C ihi + 1 and Fortran I = C i + 1:
               Fortran ISHIFT = (ihi+1)-(i+1)-1 to (ihi+1)-2 = ihi-i-1 to ihi-1
               But ISHIFT is the k parameter to dlaqz2, which should be 0-based.
               Fortran ISHIFT is 1-based position, C needs 0-based.
               So C ishift should go from ihi-i-2 to ihi-2. */
            dlaqz2(1, 1, ishift, istartb, istopb, ihi,
                   A, lda, B, ldb, ns, ihi - ns + 1, QC, ldqc,
                   ns + 1, ihi - ns, ZC, ldzc);
        }
    }

    /* Update rest of the pencil */

    /* Update A(ihi-ns+1:ihi, ihi+1:istopm) from the left with Qc(1:ns,1:ns)' */
    sheight = ns;
    swidth = istopm - ihi;
    if (swidth > 0) {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                    ONE, QC, ldqc, &A[(ihi - ns + 1) + (ihi + 1) * lda], lda, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &A[(ihi - ns + 1) + (ihi + 1) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, sheight, swidth, sheight,
                    ONE, QC, ldqc, &B[(ihi - ns + 1) + (ihi + 1) * ldb], ldb, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &B[(ihi - ns + 1) + (ihi + 1) * ldb], ldb);
    }
    if (ilq) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, ns, ns,
                    ONE, &Q[0 + (ihi - ns + 1) * ldq], ldq, QC, ldqc, ZERO, work, n);
        dlacpy("A", n, ns, work, n, &Q[0 + (ihi - ns + 1) * ldq], ldq);
    }

    /* Update A(istartm:ihi-ns, ihi-ns:ihi) from the right with Zc(1:ns+1,1:ns+1) */
    sheight = ihi - ns - istartm + 1;
    swidth = ns + 1;
    if (sheight > 0) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    ONE, &A[istartm + (ihi - ns) * lda], lda, ZC, ldzc, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &A[istartm + (ihi - ns) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sheight, swidth, swidth,
                    ONE, &B[istartm + (ihi - ns) * ldb], ldb, ZC, ldzc, ZERO, work, sheight);
        dlacpy("A", sheight, swidth, work, sheight, &B[istartm + (ihi - ns) * ldb], ldb);
    }
    if (ilz) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, ns + 1, ns + 1,
                    ONE, &Z[0 + (ihi - ns) * ldz], ldz, ZC, ldzc, ZERO, work, n);
        dlacpy("A", n, ns + 1, work, n, &Z[0 + (ihi - ns) * ldz], ldz);
    }
}
