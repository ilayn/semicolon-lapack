/**
 * @file dgghd3.c
 * @brief DGGHD3 reduces a pair of real matrices (A,B) to generalized upper Hessenberg form.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DGGHD3 reduces a pair of real matrices (A,B) to generalized upper
 * Hessenberg form using orthogonal transformations, where A is a
 * general matrix and B is upper triangular. The form of the
 * generalized eigenvalue problem is
 *    A*x = lambda*B*x,
 * and B is typically made upper triangular by computing its QR
 * factorization and moving the orthogonal matrix Q to the left side
 * of the equation.
 *
 * This subroutine simultaneously reduces A to a Hessenberg matrix H:
 *    Q**T*A*Z = H
 * and transforms B to another upper triangular matrix T:
 *    Q**T*B*Z = T
 * in order to reduce the problem to its standard form
 *    H*y = lambda*T*y
 * where y = Z**T*x.
 *
 * The orthogonal matrices Q and Z are determined as products of Givens
 * rotations. They may either be formed explicitly, or they may be
 * postmultiplied into input matrices Q1 and Z1, so that
 *
 *      Q1 * A * Z1**T = (Q1*Q) * H * (Z1*Z)**T
 *
 *      Q1 * B * Z1**T = (Q1*Q) * T * (Z1*Z)**T
 *
 * If Q1 is the orthogonal matrix from the QR factorization of B in the
 * original equation A*x = lambda*B*x, then DGGHD3 reduces the original
 * problem to generalized Hessenberg form.
 *
 * This is a blocked variant of DGGHRD, using matrix-matrix
 * multiplications for parts of the computation to enhance performance.
 *
 * @param[in]     compq   = 'N': do not compute Q;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'V': Q must contain an orthogonal matrix Q1 on entry.
 * @param[in]     compz   = 'N': do not compute Z;
 *                        = 'I': Z is initialized to the unit matrix;
 *                        = 'V': Z must contain an orthogonal matrix Z1 on entry.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in]     ilo     See ihi.
 * @param[in]     ihi     ILO and IHI mark the rows and columns of A which are to be
 *                        reduced. It is assumed that A is already upper triangular in
 *                        rows and columns 0:ilo-1 and ihi+1:n-1. ILO and IHI are normally
 *                        set by a previous call to DGGBAL; otherwise they should be set
 *                        to 0 and n-1 respectively.
 *                        0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 * @param[in,out] A       Array of dimension (lda, n). On entry, the N-by-N general
 *                        matrix. On exit, the upper triangle and the first subdiagonal
 *                        of A are overwritten with the upper Hessenberg matrix H, and
 *                        the rest is set to zero.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Array of dimension (ldb, n). On entry, the upper triangular
 *                        matrix B. On exit, the upper triangular matrix T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[in,out] Q       Array of dimension (ldq, n). The orthogonal matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= N if COMPQ='V' or 'I'.
 * @param[in,out] Z       Array of dimension (ldz, n). The orthogonal matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= N if COMPZ='V' or 'I'.
 * @param[out]    work    Array of dimension (max(1,lwork)). On exit, if info = 0,
 *                        work[0] returns the optimal lwork.
 * @param[in]     lwork   The length of the array work. lwork >= 1.
 *                        For optimum performance lwork >= 6*n*nb, where nb is the
 *                        optimal blocksize.
 *                        If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgghd3(
    const char* compq,
    const char* compz,
    const INT n,
    const INT ilo,
    const INT ihi,
    f64* restrict A,
    const INT lda,
    f64* restrict B,
    const INT ldb,
    f64* restrict Q,
    const INT ldq,
    f64* restrict Z,
    const INT ldz,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT blk22, initq, initz, lquery, wantq, wantz;
    INT cola, i, ierr, j, j0, jcol, jj, jrow, k;
    INT kacc22, len, lwkopt, n2nb, nb, nblst, nbmin;
    INT nh, nnb, nx, ppw, ppwo, pw, top = 0, topq;
    f64 c, c1, c2, s, s1, s2, temp, temp1, temp2, temp3;

    *info = 0;
    nb = lapack_get_nb("GGHD3");
    nh = ihi - ilo + 1;
    if (nh <= 1) {
        lwkopt = 1;
    } else {
        lwkopt = 6 * n * nb;
    }
    work[0] = (f64)lwkopt;
    initq = (compq[0] == 'I' || compq[0] == 'i');
    wantq = initq || (compq[0] == 'V' || compq[0] == 'v');
    initz = (compz[0] == 'I' || compz[0] == 'i');
    wantz = initz || (compz[0] == 'V' || compz[0] == 'v');
    lquery = (lwork == -1);

    if (compq[0] != 'N' && compq[0] != 'n' && !wantq) {
        *info = -1;
    } else if (compz[0] != 'N' && compz[0] != 'n' && !wantz) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 0 || (n > 0 && ilo > n - 1)) {
        *info = -4;
    } else if (ihi > n - 1 || ihi < ilo - 1) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if ((wantq && ldq < n) || ldq < 1) {
        *info = -11;
    } else if ((wantz && ldz < n) || ldz < 1) {
        *info = -13;
    } else if (lwork < 1 && !lquery) {
        *info = -15;
    }
    if (*info != 0) {
        xerbla("DGGHD3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (initq)
        dlaset("A", n, n, ZERO, ONE, Q, ldq);
    if (initz)
        dlaset("A", n, n, ZERO, ONE, Z, ldz);

    if (n > 1)
        dlaset("L", n - 1, n - 1, ZERO, ZERO, &B[1], ldb);

    if (nh <= 1) {
        work[0] = ONE;
        return;
    }

    nbmin = lapack_get_nbmin("GGHD3");
    if (nb > 1 && nb < nh) {
        nx = lapack_get_nx("GGHD3");
        if (nx < nb)
            nx = nb;
        if (nx < nh) {
            if (lwork < lwkopt) {
                nbmin = lapack_get_nbmin("GGHD3");
                if (nbmin < 2)
                    nbmin = 2;
                if (lwork >= 6 * n * nbmin) {
                    nb = lwork / (6 * n);
                } else {
                    nb = 1;
                }
            }
        }
    }

    if (nb < nbmin || nb >= nh) {

        jcol = ilo;

    } else {

        kacc22 = (nh >= 14) ? 2 : 1;
        blk22 = (kacc22 == 2);

        for (jcol = ilo; jcol <= ihi - 2; jcol += nb) {
            nnb = nb < (ihi - jcol - 1) ? nb : (ihi - jcol - 1);

            n2nb = (ihi - jcol - 1) / nnb - 1;
            nblst = ihi - jcol - n2nb * nnb;

            dlaset("A", nblst, nblst, ZERO, ONE, work, nblst);
            pw = nblst * nblst;
            for (i = 0; i < n2nb; i++) {
                dlaset("A", 2 * nnb, 2 * nnb, ZERO, ONE, &work[pw], 2 * nnb);
                pw += 4 * nnb * nnb;
            }

            for (j = jcol; j < jcol + nnb; j++) {

                for (i = ihi; i >= j + 2; i--) {
                    temp = A[i - 1 + j * lda];
                    dlartg(temp, A[i + j * lda], &c, &s, &A[i - 1 + j * lda]);
                    A[i + j * lda] = c;
                    B[i + j * ldb] = s;
                }

                ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                len = 2 + j - jcol;
                jrow = j + n2nb * nnb + 2;
                for (i = ihi; i >= jrow; i--) {
                    c = A[i + j * lda];
                    s = B[i + j * ldb];
                    for (jj = ppw; jj < ppw + len; jj++) {
                        temp = work[jj + nblst];
                        work[jj + nblst] = c * temp - s * work[jj];
                        work[jj] = s * temp + c * work[jj];
                    }
                    len++;
                    ppw = ppw - nblst - 1;
                }

                ppwo = nblst * nblst + (nnb + j - jcol - 1) * 2 * nnb + nnb - 1;
                j0 = jrow - nnb;
                for (jrow = j0; jrow >= j + 2; jrow -= nnb) {
                    ppw = ppwo;
                    len = 2 + j - jcol;
                    for (i = jrow + nnb - 1; i >= jrow; i--) {
                        c = A[i + j * lda];
                        s = B[i + j * ldb];
                        for (jj = ppw; jj < ppw + len; jj++) {
                            temp = work[jj + 2 * nnb];
                            work[jj + 2 * nnb] = c * temp - s * work[jj];
                            work[jj] = s * temp + c * work[jj];
                        }
                        len++;
                        ppw = ppw - 2 * nnb - 1;
                    }
                    ppwo += 4 * nnb * nnb;
                }

                if (jcol <= 1) {
                    top = 0;
                } else {
                    top = jcol + 1;
                }

                for (jj = n - 1; jj >= j + 1; jj--) {

                    for (i = (jj + 1 < ihi ? jj + 1 : ihi); i >= j + 2; i--) {
                        c = A[i + j * lda];
                        s = B[i + j * ldb];
                        temp = B[i + jj * ldb];
                        B[i + jj * ldb] = c * temp - s * B[i - 1 + jj * ldb];
                        B[i - 1 + jj * ldb] = s * temp + c * B[i - 1 + jj * ldb];
                    }

                    if (jj < ihi) {
                        temp = B[jj + 1 + (jj + 1) * ldb];
                        dlartg(temp, B[jj + 1 + jj * ldb], &c, &s,
                               &B[jj + 1 + (jj + 1) * ldb]);
                        B[jj + 1 + jj * ldb] = ZERO;
                        cblas_drot(jj + 1 - top, &B[top + (jj + 1) * ldb], 1,
                                   &B[top + jj * ldb], 1, c, s);
                        A[jj + 1 + j * lda] = c;
                        B[jj + 1 + j * ldb] = -s;
                    }
                }

                jj = (ihi - j - 1) % 3;
                for (i = ihi - j - 3; i >= jj + 1; i -= 3) {
                    c = A[j + 1 + i + j * lda];
                    s = -B[j + 1 + i + j * ldb];
                    c1 = A[j + 2 + i + j * lda];
                    s1 = -B[j + 2 + i + j * ldb];
                    c2 = A[j + 3 + i + j * lda];
                    s2 = -B[j + 3 + i + j * ldb];

                    for (k = top; k <= ihi; k++) {
                        temp = A[k + (j + i) * lda];
                        temp1 = A[k + (j + i + 1) * lda];
                        temp2 = A[k + (j + i + 2) * lda];
                        temp3 = A[k + (j + i + 3) * lda];
                        A[k + (j + i + 3) * lda] = c2 * temp3 + s2 * temp2;
                        temp2 = -s2 * temp3 + c2 * temp2;
                        A[k + (j + i + 2) * lda] = c1 * temp2 + s1 * temp1;
                        temp1 = -s1 * temp2 + c1 * temp1;
                        A[k + (j + i + 1) * lda] = c * temp1 + s * temp;
                        A[k + (j + i) * lda] = -s * temp1 + c * temp;
                    }
                }

                if (jj > 0) {
                    for (i = jj; i >= 1; i--) {
                        cblas_drot(ihi + 1 - top,
                                   &A[top + (j + i + 1) * lda], 1,
                                   &A[top + (j + i) * lda], 1,
                                   A[j + 1 + i + j * lda],
                                   -B[j + 1 + i + j * ldb]);
                    }
                }

                if (j < jcol + nnb - 1) {
                    len = 1 + j - jcol;

                    jrow = ihi - nblst + 1;
                    cblas_dgemv(CblasColMajor, CblasTrans, nblst, len, ONE,
                                work, nblst, &A[jrow + (j + 1) * lda], 1,
                                ZERO, &work[pw], 1);
                    ppw = pw + len;
                    for (i = jrow; i <= jrow + nblst - len - 1; i++) {
                        work[ppw] = A[i + (j + 1) * lda];
                        ppw++;
                    }
                    cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, nblst - len,
                                &work[len * nblst], nblst, &work[pw + len], 1);
                    cblas_dgemv(CblasColMajor, CblasTrans, len, nblst - len,
                                ONE,
                                &work[(len + 1) * nblst - len], nblst,
                                &A[jrow + nblst - len + (j + 1) * lda], 1,
                                ONE, &work[pw + len], 1);
                    ppw = pw;
                    for (i = jrow; i <= jrow + nblst - 1; i++) {
                        A[i + (j + 1) * lda] = work[ppw];
                        ppw++;
                    }

                    ppwo = nblst * nblst;
                    j0 = jrow - nnb;
                    for (jrow = j0; jrow >= jcol + 1; jrow -= nnb) {
                        ppw = pw + len;
                        for (i = jrow; i <= jrow + nnb - 1; i++) {
                            work[ppw] = A[i + (j + 1) * lda];
                            ppw++;
                        }
                        ppw = pw;
                        for (i = jrow + nnb; i <= jrow + nnb + len - 1; i++) {
                            work[ppw] = A[i + (j + 1) * lda];
                            ppw++;
                        }
                        cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans,
                                    CblasNonUnit, len,
                                    &work[ppwo + nnb], 2 * nnb,
                                    &work[pw], 1);
                        cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans,
                                    CblasNonUnit, nnb,
                                    &work[ppwo + 2 * len * nnb],
                                    2 * nnb, &work[pw + len], 1);
                        cblas_dgemv(CblasColMajor, CblasTrans, nnb, len, ONE,
                                    &work[ppwo], 2 * nnb,
                                    &A[jrow + (j + 1) * lda], 1,
                                    ONE, &work[pw], 1);
                        cblas_dgemv(CblasColMajor, CblasTrans, len, nnb, ONE,
                                    &work[ppwo + 2 * len * nnb + nnb], 2 * nnb,
                                    &A[jrow + nnb + (j + 1) * lda], 1,
                                    ONE, &work[pw + len], 1);
                        ppw = pw;
                        for (i = jrow; i <= jrow + len + nnb - 1; i++) {
                            A[i + (j + 1) * lda] = work[ppw];
                            ppw++;
                        }
                        ppwo += 4 * nnb * nnb;
                    }
                }
            }

            cola = n - jcol - nnb;
            j = ihi - nblst + 1;
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        nblst, cola, nblst, ONE, work, nblst,
                        &A[j + (jcol + nnb) * lda], lda, ZERO,
                        &work[pw], nblst);
            dlacpy("A", nblst, cola, &work[pw], nblst,
                   &A[j + (jcol + nnb) * lda], lda);
            ppwo = nblst * nblst;
            j0 = j - nnb;
            for (j = j0; j >= jcol + 1; j -= nnb) {
                if (blk22) {
                    dorm22("L", "T", 2 * nnb, cola, nnb, nnb,
                           &work[ppwo], 2 * nnb,
                           &A[j + (jcol + nnb) * lda], lda,
                           &work[pw], lwork - pw, &ierr);
                } else {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                2 * nnb, cola, 2 * nnb, ONE,
                                &work[ppwo], 2 * nnb,
                                &A[j + (jcol + nnb) * lda], lda, ZERO,
                                &work[pw], 2 * nnb);
                    dlacpy("A", 2 * nnb, cola, &work[pw], 2 * nnb,
                           &A[j + (jcol + nnb) * lda], lda);
                }
                ppwo += 4 * nnb * nnb;
            }

            if (wantq) {
                j = ihi - nblst + 1;
                if (initq) {
                    topq = (j - jcol > 1) ? (j - jcol) : 1;
                    nh = ihi - topq + 1;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            nh, nblst, nblst, ONE,
                            &Q[topq + j * ldq], ldq,
                            work, nblst, ZERO, &work[pw], nh);
                dlacpy("A", nh, nblst, &work[pw], nh,
                       &Q[topq + j * ldq], ldq);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (j - jcol > 1) ? (j - jcol) : 1;
                        nh = ihi - topq + 1;
                    }
                    if (blk22) {
                        dorm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb,
                               &Q[topq + j * ldq], ldq,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, ONE,
                                    &Q[topq + j * ldq], ldq,
                                    &work[ppwo], 2 * nnb, ZERO,
                                    &work[pw], nh);
                        dlacpy("A", nh, 2 * nnb, &work[pw], nh,
                               &Q[topq + j * ldq], ldq);
                    }
                    ppwo += 4 * nnb * nnb;
                }
            }

            if (wantz || top > 0) {

                dlaset("A", nblst, nblst, ZERO, ONE, work, nblst);
                pw = nblst * nblst;
                for (i = 0; i < n2nb; i++) {
                    dlaset("A", 2 * nnb, 2 * nnb, ZERO, ONE,
                           &work[pw], 2 * nnb);
                    pw += 4 * nnb * nnb;
                }

                for (j = jcol; j < jcol + nnb; j++) {
                    ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                    len = 2 + j - jcol;
                    jrow = j + n2nb * nnb + 2;
                    for (i = ihi; i >= jrow; i--) {
                        c = A[i + j * lda];
                        A[i + j * lda] = ZERO;
                        s = B[i + j * ldb];
                        B[i + j * ldb] = ZERO;
                        for (jj = ppw; jj < ppw + len; jj++) {
                            temp = work[jj + nblst];
                            work[jj + nblst] = c * temp - s * work[jj];
                            work[jj] = s * temp + c * work[jj];
                        }
                        len++;
                        ppw = ppw - nblst - 1;
                    }

                    ppwo = nblst * nblst + (nnb + j - jcol - 1) * 2 * nnb + nnb - 1;
                    j0 = jrow - nnb;
                    for (jrow = j0; jrow >= j + 2; jrow -= nnb) {
                        ppw = ppwo;
                        len = 2 + j - jcol;
                        for (i = jrow + nnb - 1; i >= jrow; i--) {
                            c = A[i + j * lda];
                            A[i + j * lda] = ZERO;
                            s = B[i + j * ldb];
                            B[i + j * ldb] = ZERO;
                            for (jj = ppw; jj < ppw + len; jj++) {
                                temp = work[jj + 2 * nnb];
                                work[jj + 2 * nnb] = c * temp - s * work[jj];
                                work[jj] = s * temp + c * work[jj];
                            }
                            len++;
                            ppw = ppw - 2 * nnb - 1;
                        }
                        ppwo += 4 * nnb * nnb;
                    }
                }
            } else {

                dlaset("L", ihi - jcol - 1, nnb, ZERO, ZERO,
                       &A[jcol + 2 + jcol * lda], lda);
                dlaset("L", ihi - jcol - 1, nnb, ZERO, ZERO,
                       &B[jcol + 2 + jcol * ldb], ldb);
            }

            if (top > 0) {
                j = ihi - nblst + 1;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            top, nblst, nblst, ONE,
                            &A[j * lda], lda, work, nblst, ZERO,
                            &work[pw], top);
                dlacpy("A", top, nblst, &work[pw], top,
                       &A[j * lda], lda);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        dorm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb,
                               &A[j * lda], lda, &work[pw],
                               lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, ONE,
                                    &A[j * lda], lda,
                                    &work[ppwo], 2 * nnb, ZERO,
                                    &work[pw], top);
                        dlacpy("A", top, 2 * nnb, &work[pw], top,
                               &A[j * lda], lda);
                    }
                    ppwo += 4 * nnb * nnb;
                }

                j = ihi - nblst + 1;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            top, nblst, nblst, ONE,
                            &B[j * ldb], ldb, work, nblst, ZERO,
                            &work[pw], top);
                dlacpy("A", top, nblst, &work[pw], top,
                       &B[j * ldb], ldb);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        dorm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb,
                               &B[j * ldb], ldb, &work[pw],
                               lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, ONE,
                                    &B[j * ldb], ldb,
                                    &work[ppwo], 2 * nnb, ZERO,
                                    &work[pw], top);
                        dlacpy("A", top, 2 * nnb, &work[pw], top,
                               &B[j * ldb], ldb);
                    }
                    ppwo += 4 * nnb * nnb;
                }
            }

            if (wantz) {
                j = ihi - nblst + 1;
                if (initq) {
                    topq = (j - jcol > 1) ? (j - jcol) : 1;
                    nh = ihi - topq + 1;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            nh, nblst, nblst, ONE,
                            &Z[topq + j * ldz], ldz,
                            work, nblst, ZERO, &work[pw], nh);
                dlacpy("A", nh, nblst, &work[pw], nh,
                       &Z[topq + j * ldz], ldz);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (j - jcol > 1) ? (j - jcol) : 1;
                        nh = ihi - topq + 1;
                    }
                    if (blk22) {
                        dorm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb,
                               &Z[topq + j * ldz], ldz,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, ONE,
                                    &Z[topq + j * ldz], ldz,
                                    &work[ppwo], 2 * nnb, ZERO,
                                    &work[pw], nh);
                        dlacpy("A", nh, 2 * nnb, &work[pw], nh,
                               &Z[topq + j * ldz], ldz);
                    }
                    ppwo += 4 * nnb * nnb;
                }
            }
        }
    }

    {
        const char* compq2 = compq;
        const char* compz2 = compz;
        if (jcol != ilo) {
            if (wantq)
                compq2 = "V";
            if (wantz)
                compz2 = "V";
        }

        if (jcol < ihi)
            dgghrd(compq2, compz2, n, jcol, ihi, A, lda, B, ldb,
                   Q, ldq, Z, ldz, &ierr);
    }

    work[0] = (f64)lwkopt;
}
