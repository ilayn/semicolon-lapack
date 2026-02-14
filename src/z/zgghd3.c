/**
 * @file zgghd3.c
 * @brief ZGGHD3 reduces a pair of complex matrices (A,B) to generalized upper Hessenberg form.
 */

#include <cblas.h>
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZGGHD3 reduces a pair of complex matrices (A,B) to generalized upper
 * Hessenberg form using unitary transformations, where A is a
 * general matrix and B is upper triangular.  The form of the
 * generalized eigenvalue problem is
 *    A*x = lambda*B*x,
 * and B is typically made upper triangular by computing its QR
 * factorization and moving the unitary matrix Q to the left side
 * of the equation.
 *
 * This subroutine simultaneously reduces A to a Hessenberg matrix H:
 *    Q**H*A*Z = H
 * and transforms B to another upper triangular matrix T:
 *    Q**H*B*Z = T
 * in order to reduce the problem to its standard form
 *    H*y = lambda*T*y
 * where y = Z**H*x.
 *
 * The unitary matrices Q and Z are determined as products of Givens
 * rotations.  They may either be formed explicitly, or they may be
 * postmultiplied into input matrices Q1 and Z1, so that
 *      Q1 * A * Z1**H = (Q1*Q) * H * (Z1*Z)**H
 *      Q1 * B * Z1**H = (Q1*Q) * T * (Z1*Z)**H
 * If Q1 is the unitary matrix from the QR factorization of B in the
 * original equation A*x = lambda*B*x, then ZGGHD3 reduces the original
 * problem to generalized Hessenberg form.
 *
 * This is a blocked variant of ZGGHRD, using matrix-matrix
 * multiplications for parts of the computation to enhance performance.
 *
 * @param[in] compq  = 'N': do not compute Q;
 *                    = 'I': Q is initialized to the unit matrix, and the
 *                           unitary matrix Q is returned;
 *                    = 'V': Q must contain a unitary matrix Q1 on entry,
 *                           and the product Q1*Q is returned.
 * @param[in] compz  = 'N': do not compute Z;
 *                    = 'I': Z is initialized to the unit matrix, and the
 *                           unitary matrix Z is returned;
 *                    = 'V': Z must contain a unitary matrix Z1 on entry,
 *                           and the product Z1*Z is returned.
 * @param[in] n      The order of the matrices A and B. n >= 0.
 * @param[in] ilo    ILO and IHI mark the rows and columns of A which are
 *                   to be reduced. It is assumed that A is already upper
 *                   triangular in rows and columns 1:ILO-1 and IHI+1:N.
 *                   ILO and IHI are normally set by a previous call to
 *                   ZGGBAL; otherwise they should be set to 1 and N
 *                   respectively.
 *                   1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 * @param[in] ihi    See ilo.
 * @param[in,out] A  Complex array, dimension (lda, n).
 *                   On entry, the N-by-N general matrix to be reduced.
 *                   On exit, the upper triangle and the first subdiagonal of A
 *                   are overwritten with the upper Hessenberg matrix H, and the
 *                   rest is set to zero.
 * @param[in] lda    The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B  Complex array, dimension (ldb, n).
 *                   On entry, the N-by-N upper triangular matrix B.
 *                   On exit, the upper triangular matrix T = Q**H B Z. The
 *                   elements below the diagonal are set to zero.
 * @param[in] ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q  Complex array, dimension (ldq, n).
 *                   On entry, if compq = 'V', the unitary matrix Q1,
 *                   typically from the QR factorization of B.
 *                   On exit, if compq = 'I', the unitary matrix Q, and if
 *                   compq = 'V', the product Q1*Q.
 *                   Not referenced if compq = 'N'.
 * @param[in] ldq    The leading dimension of Q.
 *                   ldq >= n if compq = 'V' or 'I'; ldq >= 1 otherwise.
 * @param[in,out] Z  Complex array, dimension (ldz, n).
 *                   On entry, if compz = 'V', the unitary matrix Z1.
 *                   On exit, if compz = 'I', the unitary matrix Z, and if
 *                   compz = 'V', the product Z1*Z.
 *                   Not referenced if compz = 'N'.
 * @param[in] ldz    The leading dimension of Z.
 *                   ldz >= n if compz = 'V' or 'I'; ldz >= 1 otherwise.
 * @param[out] work  Complex workspace array, dimension (max(1, lwork)).
 *                   On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in] lwork  The length of the array work. lwork >= 1.
 *                   For optimum performance lwork >= 6*N*NB, where NB is the
 *                   optimal blocksize.
 *                   If lwork = -1, then a workspace query is assumed.
 * @param[out] info  = 0: successful exit.
 *                   < 0: if info = -i, the i-th argument had an illegal value.
 */
void zgghd3(
    const char* compq,
    const char* compz,
    const int n,
    const int ilo,
    const int ihi,
    c128* A,
    const int lda,
    c128* B,
    const int ldb,
    c128* Q,
    const int ldq,
    c128* Z,
    const int ldz,
    c128* work,
    const int lwork,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    int blk22, initq, initz, lquery, wantq, wantz;
    char compq2, compz2;
    int cola, i, ierr, j, j0, jcol, jj, jrow, k,
        kacc22, len, lwkopt, n2nb, nb, nblst, nbmin,
        nh, nnb, nx, ppw, ppwo, pw, top, topq;
    f64 c;
    c128 c1, c2, ctemp, s, s1, s2, temp, temp1, temp2, temp3;

    *info = 0;
    nb = lapack_get_nb("GGHD3");
    nh = ihi - ilo + 1;
    if (nh <= 1) {
        lwkopt = 1;
    } else {
        lwkopt = 6 * n * nb;
    }
    work[0] = CMPLX((f64)lwkopt, 0.0);
    initq = (compq[0] == 'I' || compq[0] == 'i');
    wantq = initq || (compq[0] == 'V' || compq[0] == 'v');
    initz = (compz[0] == 'I' || compz[0] == 'i');
    wantz = initz || (compz[0] == 'V' || compz[0] == 'v');
    lquery = (lwork == -1);

    if (!(compq[0] == 'N' || compq[0] == 'n') && !wantq) {
        *info = -1;
    } else if (!(compz[0] == 'N' || compz[0] == 'n') && !wantz) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1) {
        *info = -4;
    } else if (ihi > n || ihi < ilo - 1) {
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
        xerbla("ZGGHD3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (initq)
        zlaset("A", n, n, CZERO, CONE, Q, ldq);
    if (initz)
        zlaset("A", n, n, CZERO, CONE, Z, ldz);

    if (n > 1)
        zlaset("L", n - 1, n - 1, CZERO, CZERO, &B[1], ldb);

    if (nh <= 1) {
        work[0] = CONE;
        return;
    }

    nbmin = lapack_get_nbmin("GGHD3");
    if (nb > 1 && nb < nh) {
        nx = lapack_get_nx("GGHD3");
        if (nx > nb) nx = nb;
        if (nx < nh) {
            if (lwork < lwkopt) {
                nbmin = lapack_get_nbmin("GGHD3");
                if (nbmin < 2) nbmin = 2;
                if (lwork >= 6 * n * nbmin) {
                    nb = lwork / (6 * n);
                } else {
                    nb = 1;
                }
            }
        }
    }

    if (nb < nbmin || nb >= nh) {
        jcol = ilo - 1;
    } else {
        kacc22 = (nh >= 14) ? 2 : 1;
        blk22 = (kacc22 == 2);
        top = 0;

        for (jcol = ilo - 1; jcol <= ihi - 3; jcol += nb) {
            nnb = nb;
            if (nnb > ihi - jcol - 2) nnb = ihi - jcol - 2;

            n2nb = (ihi - jcol - 2) / nnb - 1;
            nblst = ihi - jcol - 1 - n2nb * nnb;
            zlaset("A", nblst, nblst, CZERO, CONE, work, nblst);
            pw = nblst * nblst;
            for (i = 0; i < n2nb; i++) {
                zlaset("A", 2 * nnb, 2 * nnb, CZERO, CONE, &work[pw], 2 * nnb);
                pw = pw + 4 * nnb * nnb;
            }

            for (j = jcol; j <= jcol + nnb - 1; j++) {
                for (i = ihi - 1; i >= j + 2; i--) {
                    temp = A[(i - 1) + j * lda];
                    zlartg(temp, A[i + j * lda], &c, &s, &A[(i - 1) + j * lda]);
                    A[i + j * lda] = CMPLX(c, 0.0);
                    B[i + j * ldb] = s;
                }

                ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                len = 2 + j - jcol;
                jrow = j + n2nb * nnb + 2;
                for (i = ihi - 1; i >= jrow; i--) {
                    ctemp = A[i + j * lda];
                    s = B[i + j * ldb];
                    for (jj = ppw; jj <= ppw + len - 1; jj++) {
                        temp = work[jj + nblst];
                        work[jj + nblst] = ctemp * temp - s * work[jj];
                        work[jj] = conj(s) * temp + ctemp * work[jj];
                    }
                    len = len + 1;
                    ppw = ppw - nblst - 1;
                }

                ppwo = nblst * nblst + (nnb + j - jcol - 1) * 2 * nnb + nnb;
                j0 = jrow - nnb;
                for (jrow = j0; jrow >= j + 2; jrow -= nnb) {
                    ppw = ppwo;
                    len = 2 + j - jcol;
                    for (i = jrow + nnb - 1; i >= jrow; i--) {
                        ctemp = A[i + j * lda];
                        s = B[i + j * ldb];
                        for (jj = ppw; jj <= ppw + len - 1; jj++) {
                            temp = work[jj + 2 * nnb];
                            work[jj + 2 * nnb] = ctemp * temp - s * work[jj];
                            work[jj] = conj(s) * temp + ctemp * work[jj];
                        }
                        len = len + 1;
                        ppw = ppw - 2 * nnb - 1;
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }

                if (jcol <= 1) {
                    top = 0;
                } else {
                    top = jcol;
                }

                for (jj = n - 1; jj >= j + 1; jj--) {
                    for (i = (jj + 1 < ihi - 1 ? jj + 1 : ihi - 1); i >= j + 2; i--) {
                        ctemp = A[i + j * lda];
                        s = B[i + j * ldb];
                        temp = B[i + jj * ldb];
                        B[i + jj * ldb] = ctemp * temp - conj(s) * B[(i - 1) + jj * ldb];
                        B[(i - 1) + jj * ldb] = s * temp + ctemp * B[(i - 1) + jj * ldb];
                    }

                    if (jj < ihi - 1) {
                        temp = B[(jj + 1) + (jj + 1) * ldb];
                        zlartg(temp, B[(jj + 1) + jj * ldb], &c, &s, &B[(jj + 1) + (jj + 1) * ldb]);
                        B[(jj + 1) + jj * ldb] = CZERO;
                        zrot(jj - top, &B[top + (jj + 1) * ldb], 1,
                             &B[top + jj * ldb], 1, c, s);
                        A[(jj + 1) + j * lda] = CMPLX(c, 0.0);
                        B[(jj + 1) + j * ldb] = -conj(s);
                    }
                }

                jj = (ihi - j - 2) % 3;
                for (i = ihi - j - 4; i >= jj + 1; i -= 3) {
                    ctemp = A[(j + 1 + i) + j * lda];
                    s = -B[(j + 1 + i) + j * ldb];
                    c1 = A[(j + 2 + i) + j * lda];
                    s1 = -B[(j + 2 + i) + j * ldb];
                    c2 = A[(j + 3 + i) + j * lda];
                    s2 = -B[(j + 3 + i) + j * ldb];

                    for (k = top; k <= ihi - 1; k++) {
                        temp = A[k + (j + i) * lda];
                        temp1 = A[k + (j + i + 1) * lda];
                        temp2 = A[k + (j + i + 2) * lda];
                        temp3 = A[k + (j + i + 3) * lda];
                        A[k + (j + i + 3) * lda] = c2 * temp3 + conj(s2) * temp2;
                        temp2 = -s2 * temp3 + c2 * temp2;
                        A[k + (j + i + 2) * lda] = c1 * temp2 + conj(s1) * temp1;
                        temp1 = -s1 * temp2 + c1 * temp1;
                        A[k + (j + i + 1) * lda] = ctemp * temp1 + conj(s) * temp;
                        A[k + (j + i) * lda] = -s * temp1 + ctemp * temp;
                    }
                }

                if (jj > 0) {
                    for (i = jj; i >= 1; i--) {
                        c = creal(A[(j + 1 + i) + j * lda]);
                        zrot(ihi - top, &A[top + (j + i + 1) * lda], 1,
                             &A[top + (j + i) * lda], 1, c,
                             -conj(B[(j + 1 + i) + j * ldb]));
                    }
                }

                if (j < jcol + nnb - 1) {
                    len = 1 + j - jcol;

                    jrow = ihi - nblst;
                    cblas_zgemv(CblasColMajor, CblasConjTrans, nblst, len, &CONE, work,
                                nblst, &A[jrow + (j + 1) * lda], 1, &CZERO,
                                &work[pw], 1);
                    ppw = pw + len;
                    for (i = jrow; i <= jrow + nblst - len - 1; i++) {
                        work[ppw] = A[i + (j + 1) * lda];
                        ppw = ppw + 1;
                    }
                    cblas_ztrmv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                                nblst - len, &work[len * nblst], nblst,
                                &work[pw + len], 1);
                    cblas_zgemv(CblasColMajor, CblasConjTrans, len, nblst - len, &CONE,
                                &work[(len + 1) * nblst - len], nblst,
                                &A[(jrow + nblst - len) + (j + 1) * lda], 1, &CONE,
                                &work[pw + len], 1);
                    ppw = pw;
                    for (i = jrow; i <= jrow + nblst - 1; i++) {
                        A[i + (j + 1) * lda] = work[ppw];
                        ppw = ppw + 1;
                    }

                    ppwo = nblst * nblst;
                    j0 = jrow - nnb;
                    for (jrow = j0; jrow >= jcol + 1; jrow -= nnb) {
                        ppw = pw + len;
                        for (i = jrow; i <= jrow + nnb - 1; i++) {
                            work[ppw] = A[i + (j + 1) * lda];
                            ppw = ppw + 1;
                        }
                        ppw = pw;
                        for (i = jrow + nnb; i <= jrow + nnb + len - 1; i++) {
                            work[ppw] = A[i + (j + 1) * lda];
                            ppw = ppw + 1;
                        }
                        cblas_ztrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                                    len, &work[ppwo + nnb], 2 * nnb, &work[pw], 1);
                        cblas_ztrmv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                                    nnb, &work[ppwo + 2 * len * nnb],
                                    2 * nnb, &work[pw + len], 1);
                        cblas_zgemv(CblasColMajor, CblasConjTrans, nnb, len, &CONE,
                                    &work[ppwo], 2 * nnb, &A[jrow + (j + 1) * lda], 1,
                                    &CONE, &work[pw], 1);
                        cblas_zgemv(CblasColMajor, CblasConjTrans, len, nnb, &CONE,
                                    &work[ppwo + 2 * len * nnb + nnb], 2 * nnb,
                                    &A[(jrow + nnb) + (j + 1) * lda], 1, &CONE,
                                    &work[pw + len], 1);
                        ppw = pw;
                        for (i = jrow; i <= jrow + len + nnb - 1; i++) {
                            A[i + (j + 1) * lda] = work[ppw];
                            ppw = ppw + 1;
                        }
                        ppwo = ppwo + 4 * nnb * nnb;
                    }
                }
            }

            cola = n - jcol - nnb;
            j = ihi - nblst;
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nblst, cola, nblst,
                        &CONE, work, nblst, &A[j + (jcol + nnb) * lda], lda,
                        &CZERO, &work[pw], nblst);
            zlacpy("A", nblst, cola, &work[pw], nblst, &A[j + (jcol + nnb) * lda], lda);
            ppwo = nblst * nblst;
            j0 = j - nnb;
            for (j = j0; j >= jcol + 1; j -= nnb) {
                if (blk22) {
                    zunm22("L", "C", 2 * nnb, cola, nnb, nnb,
                           &work[ppwo], 2 * nnb, &A[j + (jcol + nnb) * lda], lda,
                           &work[pw], lwork - pw, &ierr);
                } else {
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                2 * nnb, cola, 2 * nnb, &CONE, &work[ppwo], 2 * nnb,
                                &A[j + (jcol + nnb) * lda], lda, &CZERO, &work[pw], 2 * nnb);
                    zlacpy("A", 2 * nnb, cola, &work[pw], 2 * nnb,
                           &A[j + (jcol + nnb) * lda], lda);
                }
                ppwo = ppwo + 4 * nnb * nnb;
            }

            if (wantq) {
                j = ihi - nblst;
                if (initq) {
                    topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                    nh = ihi - topq;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nh, nblst, nblst,
                            &CONE, &Q[topq + j * ldq], ldq, work, nblst,
                            &CZERO, &work[pw], nh);
                zlacpy("A", nh, nblst, &work[pw], nh, &Q[topq + j * ldq], ldq);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                        nh = ihi - topq;
                    }
                    if (blk22) {
                        zunm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &Q[topq + j * ldq], ldq,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, &CONE, &Q[topq + j * ldq], ldq,
                                    &work[ppwo], 2 * nnb, &CZERO, &work[pw], nh);
                        zlacpy("A", nh, 2 * nnb, &work[pw], nh, &Q[topq + j * ldq], ldq);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }

            if (wantz || top > 0) {
                zlaset("A", nblst, nblst, CZERO, CONE, work, nblst);
                pw = nblst * nblst;
                for (i = 0; i < n2nb; i++) {
                    zlaset("A", 2 * nnb, 2 * nnb, CZERO, CONE, &work[pw], 2 * nnb);
                    pw = pw + 4 * nnb * nnb;
                }

                for (j = jcol; j <= jcol + nnb - 1; j++) {
                    ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                    len = 2 + j - jcol;
                    jrow = j + n2nb * nnb + 2;
                    for (i = ihi - 1; i >= jrow; i--) {
                        ctemp = A[i + j * lda];
                        A[i + j * lda] = CZERO;
                        s = B[i + j * ldb];
                        B[i + j * ldb] = CZERO;
                        for (jj = ppw; jj <= ppw + len - 1; jj++) {
                            temp = work[jj + nblst];
                            work[jj + nblst] = ctemp * temp - conj(s) * work[jj];
                            work[jj] = s * temp + ctemp * work[jj];
                        }
                        len = len + 1;
                        ppw = ppw - nblst - 1;
                    }

                    ppwo = nblst * nblst + (nnb + j - jcol - 1) * 2 * nnb + nnb;
                    j0 = jrow - nnb;
                    for (jrow = j0; jrow >= j + 2; jrow -= nnb) {
                        ppw = ppwo;
                        len = 2 + j - jcol;
                        for (i = jrow + nnb - 1; i >= jrow; i--) {
                            ctemp = A[i + j * lda];
                            A[i + j * lda] = CZERO;
                            s = B[i + j * ldb];
                            B[i + j * ldb] = CZERO;
                            for (jj = ppw; jj <= ppw + len - 1; jj++) {
                                temp = work[jj + 2 * nnb];
                                work[jj + 2 * nnb] = ctemp * temp - conj(s) * work[jj];
                                work[jj] = s * temp + ctemp * work[jj];
                            }
                            len = len + 1;
                            ppw = ppw - 2 * nnb - 1;
                        }
                        ppwo = ppwo + 4 * nnb * nnb;
                    }
                }
            } else {
                zlaset("L", ihi - jcol - 2, nnb, CZERO, CZERO, &A[(jcol + 2) + jcol * lda], lda);
                zlaset("L", ihi - jcol - 2, nnb, CZERO, CZERO, &B[(jcol + 2) + jcol * ldb], ldb);
            }

            if (top > 0) {
                j = ihi - nblst;
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, top, nblst, nblst,
                            &CONE, &A[j * lda], lda, work, nblst, &CZERO, &work[pw], top);
                zlacpy("A", top, nblst, &work[pw], top, &A[j * lda], lda);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        zunm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &A[j * lda], lda,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, &CONE, &A[j * lda], lda,
                                    &work[ppwo], 2 * nnb, &CZERO, &work[pw], top);
                        zlacpy("A", top, 2 * nnb, &work[pw], top, &A[j * lda], lda);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }

                j = ihi - nblst;
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, top, nblst, nblst,
                            &CONE, &B[j * ldb], ldb, work, nblst, &CZERO, &work[pw], top);
                zlacpy("A", top, nblst, &work[pw], top, &B[j * ldb], ldb);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        zunm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &B[j * ldb], ldb,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, &CONE, &B[j * ldb], ldb,
                                    &work[ppwo], 2 * nnb, &CZERO, &work[pw], top);
                        zlacpy("A", top, 2 * nnb, &work[pw], top, &B[j * ldb], ldb);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }

            if (wantz) {
                j = ihi - nblst;
                if (initq) {
                    topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                    nh = ihi - topq;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nh, nblst, nblst,
                            &CONE, &Z[topq + j * ldz], ldz, work, nblst,
                            &CZERO, &work[pw], nh);
                zlacpy("A", nh, nblst, &work[pw], nh, &Z[topq + j * ldz], ldz);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                        nh = ihi - topq;
                    }
                    if (blk22) {
                        zunm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &Z[topq + j * ldz], ldz,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, &CONE, &Z[topq + j * ldz], ldz,
                                    &work[ppwo], 2 * nnb, &CZERO, &work[pw], nh);
                        zlacpy("A", nh, 2 * nnb, &work[pw], nh, &Z[topq + j * ldz], ldz);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }
        }
    }

    compq2 = compq[0];
    compz2 = compz[0];
    if (jcol != ilo - 1) {
        if (wantq)
            compq2 = 'V';
        if (wantz)
            compz2 = 'V';
    }

    if (jcol < ihi - 1) {
        char cq2[2] = {compq2, '\0'};
        char cz2[2] = {compz2, '\0'};
        zgghrd(cq2, cz2, n, jcol + 1, ihi, A, lda, B, ldb, Q, ldq, Z, ldz, &ierr);
    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
