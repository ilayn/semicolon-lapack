/**
 * @file dgghd3.c
 * @brief DGGHD3 reduces a pair of real matrices (A,B) to generalized upper Hessenberg form.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

void dgghd3(
    const char* compq,
    const char* compz,
    const int n,
    const int ilo,
    const int ihi,
    f64* restrict A,
    const int lda,
    f64* restrict B,
    const int ldb,
    f64* restrict Q,
    const int ldq,
    f64* restrict Z,
    const int ldz,
    f64* restrict work,
    const int lwork,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int blk22, initq, initz, lquery, wantq, wantz;
    char compq2, compz2;
    int cola, i, ierr, j, j0, jcol, jj, jrow, k,
        kacc22, len, lwkopt, n2nb, nb, nblst, nbmin,
        nh, nnb, nx, ppw, ppwo, pw, top, topq;
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

    /* Chunk 2 begins (Fortran line 367) */

    if (nb < nbmin || nb >= nh) {
        /* Use unblocked code below (Fortran lines 367-372) */
        jcol = ilo - 1;  /* 0-based: Fortran ILO becomes ilo-1 */
    } else {
        /* Use blocked code (Fortran lines 373-879) */
        /* KACC22 = ILAENV(16, ...) - from iparmq.f: returns 2 if NH >= 14 (K22MIN) */
        kacc22 = (nh >= 14) ? 2 : 1;
        blk22 = (kacc22 == 2);
        top = 0;

        for (jcol = ilo - 1; jcol <= ihi - 3; jcol += nb) {
            nnb = nb;
            if (nnb > ihi - jcol - 2) nnb = ihi - jcol - 2;

            /* Initialize small orthogonal factors in workspace (Fortran lines 382-397) */
            n2nb = (ihi - jcol - 2) / nnb - 1;
            nblst = ihi - jcol - 1 - n2nb * nnb;
            dlaset("A", nblst, nblst, ZERO, ONE, work, nblst);
            pw = nblst * nblst;
            for (i = 0; i < n2nb; i++) {
                dlaset("A", 2 * nnb, 2 * nnb, ZERO, ONE, &work[pw], 2 * nnb);
                pw = pw + 4 * nnb * nnb;
            }

            /* Reduce columns JCOL:JCOL+NNB-1 of A to Hessenberg form (Fortran lines 399-611) */
            for (j = jcol; j <= jcol + nnb - 1; j++) {
                /* Reduce Jth column of A (Fortran lines 406-411) */
                for (i = ihi - 1; i >= j + 2; i--) {
                    temp = A[(i - 1) + j * lda];
                    dlartg(temp, A[i + j * lda], &c, &s, &A[(i - 1) + j * lda]);
                    A[i + j * lda] = c;
                    B[i + j * ldb] = s;
                }

                /* Accumulate Givens rotations into workspace array (Fortran lines 413-447) */
                ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                len = 2 + j - jcol;
                jrow = j + n2nb * nnb + 2;
                for (i = ihi - 1; i >= jrow; i--) {
                    c = A[i + j * lda];
                    s = B[i + j * ldb];
                    for (jj = ppw; jj <= ppw + len - 1; jj++) {
                        temp = work[jj + nblst];
                        work[jj + nblst] = c * temp - s * work[jj];
                        work[jj] = s * temp + c * work[jj];
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
                        c = A[i + j * lda];
                        s = B[i + j * ldb];
                        for (jj = ppw; jj <= ppw + len - 1; jj++) {
                            temp = work[jj + 2 * nnb];
                            work[jj + 2 * nnb] = c * temp - s * work[jj];
                            work[jj] = s * temp + c * work[jj];
                        }
                        len = len + 1;
                        ppw = ppw - 2 * nnb - 1;
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }

                /* TOP denotes number of top rows not updated (Fortran lines 449-456) */
                if (jcol <= 1) {
                    top = 0;
                } else {
                    top = jcol;
                }

                /* Propagate transformations through B (Fortran lines 458-485) */
                for (jj = n - 1; jj >= j + 1; jj--) {
                    for (i = (jj + 1 < ihi - 1 ? jj + 1 : ihi - 1); i >= j + 2; i--) {
                        c = A[i + j * lda];
                        s = B[i + j * ldb];
                        temp = B[i + jj * ldb];
                        B[i + jj * ldb] = c * temp - s * B[(i - 1) + jj * ldb];
                        B[(i - 1) + jj * ldb] = s * temp + c * B[(i - 1) + jj * ldb];
                    }

                    if (jj < ihi - 1) {
                        temp = B[(jj + 1) + (jj + 1) * ldb];
                        dlartg(temp, B[(jj + 1) + jj * ldb], &c, &s, &B[(jj + 1) + (jj + 1) * ldb]);
                        B[(jj + 1) + jj * ldb] = ZERO;
                        cblas_drot(jj - top, &B[top + (jj + 1) * ldb], 1,
                                   &B[top + jj * ldb], 1, c, s);
                        A[(jj + 1) + j * lda] = c;
                        B[(jj + 1) + j * ldb] = -s;
                    }
                }

                /* Update A by transformations from right (Fortran lines 487-523) */
                jj = (ihi - j - 2) % 3;
                for (i = ihi - j - 4; i >= jj + 1; i -= 3) {
                    c = A[(j + 1 + i) + j * lda];
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
                        cblas_drot(ihi - top, &A[top + (j + i + 1) * lda], 1,
                                   &A[top + (j + i) * lda], 1,
                                   A[(j + 1 + i) + j * lda], -B[(j + 1 + i) + j * ldb]);
                    }
                }

                /* Update (J+1)th column of A by transformations from left (Fortran lines 525-610) */
                if (j < jcol + nnb - 1) {
                    len = 1 + j - jcol;

                    jrow = ihi - nblst;
                    cblas_dgemv(CblasColMajor, CblasTrans, nblst, len, ONE, work,
                                nblst, &A[jrow + (j + 1) * lda], 1, ZERO,
                                &work[pw], 1);
                    ppw = pw + len;
                    for (i = jrow; i <= jrow + nblst - len - 1; i++) {
                        work[ppw] = A[i + (j + 1) * lda];
                        ppw = ppw + 1;
                    }
                    cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                                nblst - len, &work[len * nblst], nblst,
                                &work[pw + len], 1);
                    cblas_dgemv(CblasColMajor, CblasTrans, len, nblst - len, ONE,
                                &work[(len + 1) * nblst - len], nblst,
                                &A[(jrow + nblst - len) + (j + 1) * lda], 1, ONE,
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
                        cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                                    len, &work[ppwo + nnb], 2 * nnb, &work[pw], 1);
                        cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                                    nnb, &work[ppwo + 2 * len * nnb],
                                    2 * nnb, &work[pw + len], 1);
                        cblas_dgemv(CblasColMajor, CblasTrans, nnb, len, ONE,
                                    &work[ppwo], 2 * nnb, &A[jrow + (j + 1) * lda], 1,
                                    ONE, &work[pw], 1);
                        cblas_dgemv(CblasColMajor, CblasTrans, len, nnb, ONE,
                                    &work[ppwo + 2 * len * nnb + nnb], 2 * nnb,
                                    &A[(jrow + nnb) + (j + 1) * lda], 1, ONE,
                                    &work[pw + len], 1);
                        ppw = pw;
                        for (i = jrow; i <= jrow + len + nnb - 1; i++) {
                            A[i + (j + 1) * lda] = work[ppw];
                            ppw = ppw + 1;
                        }
                        ppwo = ppwo + 4 * nnb * nnb;
                    }
                }
            } /* end for j */

            /* Apply accumulated orthogonal matrices to A (Fortran lines 613-653) */
            cola = n - jcol - nnb;
            j = ihi - nblst;
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nblst, cola, nblst,
                        ONE, work, nblst, &A[j + (jcol + nnb) * lda], lda,
                        ZERO, &work[pw], nblst);
            dlacpy("A", nblst, cola, &work[pw], nblst, &A[j + (jcol + nnb) * lda], lda);
            ppwo = nblst * nblst;
            j0 = j - nnb;
            for (j = j0; j >= jcol + 1; j -= nnb) {
                if (blk22) {
                    dorm22("L", "T", 2 * nnb, cola, nnb, nnb,
                           &work[ppwo], 2 * nnb, &A[j + (jcol + nnb) * lda], lda,
                           &work[pw], lwork - pw, &ierr);
                } else {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                2 * nnb, cola, 2 * nnb, ONE, &work[ppwo], 2 * nnb,
                                &A[j + (jcol + nnb) * lda], lda, ZERO, &work[pw], 2 * nnb);
                    dlacpy("A", 2 * nnb, cola, &work[pw], 2 * nnb,
                           &A[j + (jcol + nnb) * lda], lda);
                }
                ppwo = ppwo + 4 * nnb * nnb;
            }

            /* Apply accumulated orthogonal matrices to Q (Fortran lines 655-699) */
            if (wantq) {
                j = ihi - nblst;
                if (initq) {
                    topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                    nh = ihi - topq;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nh, nblst, nblst,
                            ONE, &Q[topq + j * ldq], ldq, work, nblst,
                            ZERO, &work[pw], nh);
                dlacpy("A", nh, nblst, &work[pw], nh, &Q[topq + j * ldq], ldq);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                        nh = ihi - topq;
                    }
                    if (blk22) {
                        dorm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &Q[topq + j * ldq], ldq,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, ONE, &Q[topq + j * ldq], ldq,
                                    &work[ppwo], 2 * nnb, ZERO, &work[pw], nh);
                        dlacpy("A", nh, 2 * nnb, &work[pw], nh, &Q[topq + j * ldq], ldq);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }

            /* Chunk 3 begins (Fortran line 700) */

            /* Accumulate right Givens rotations if required (Fortran lines 700-764) */
            if (wantz || top > 0) {
                /* Initialize small orthogonal factors (Fortran lines 705-715) */
                dlaset("A", nblst, nblst, ZERO, ONE, work, nblst);
                pw = nblst * nblst;
                for (i = 0; i < n2nb; i++) {
                    dlaset("A", 2 * nnb, 2 * nnb, ZERO, ONE, &work[pw], 2 * nnb);
                    pw = pw + 4 * nnb * nnb;
                }

                /* Accumulate Givens rotations into workspace array (Fortran lines 717-757) */
                for (j = jcol; j <= jcol + nnb - 1; j++) {
                    ppw = (nblst + 1) * (nblst - 2) - j + jcol;
                    len = 2 + j - jcol;
                    jrow = j + n2nb * nnb + 2;
                    for (i = ihi - 1; i >= jrow; i--) {
                        c = A[i + j * lda];
                        A[i + j * lda] = ZERO;
                        s = B[i + j * ldb];
                        B[i + j * ldb] = ZERO;
                        for (jj = ppw; jj <= ppw + len - 1; jj++) {
                            temp = work[jj + nblst];
                            work[jj + nblst] = c * temp - s * work[jj];
                            work[jj] = s * temp + c * work[jj];
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
                            c = A[i + j * lda];
                            A[i + j * lda] = ZERO;
                            s = B[i + j * ldb];
                            B[i + j * ldb] = ZERO;
                            for (jj = ppw; jj <= ppw + len - 1; jj++) {
                                temp = work[jj + 2 * nnb];
                                work[jj + 2 * nnb] = c * temp - s * work[jj];
                                work[jj] = s * temp + c * work[jj];
                            }
                            len = len + 1;
                            ppw = ppw - 2 * nnb - 1;
                        }
                        ppwo = ppwo + 4 * nnb * nnb;
                    }
                }
            } else {
                /* Zero out stored rotations (Fortran lines 758-764) */
                dlaset("L", ihi - jcol - 2, nnb, ZERO, ZERO, &A[(jcol + 2) + jcol * lda], lda);
                dlaset("L", ihi - jcol - 2, nnb, ZERO, ZERO, &B[(jcol + 2) + jcol * ldb], ldb);
            }

            /* Apply accumulated orthogonal matrices to A and B (Fortran lines 766-832) */
            if (top > 0) {
                j = ihi - nblst;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, top, nblst, nblst,
                            ONE, &A[j * lda], lda, work, nblst, ZERO, &work[pw], top);
                dlacpy("A", top, nblst, &work[pw], top, &A[j * lda], lda);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        dorm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &A[j * lda], lda,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, ONE, &A[j * lda], lda,
                                    &work[ppwo], 2 * nnb, ZERO, &work[pw], top);
                        dlacpy("A", top, 2 * nnb, &work[pw], top, &A[j * lda], lda);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }

                j = ihi - nblst;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, top, nblst, nblst,
                            ONE, &B[j * ldb], ldb, work, nblst, ZERO, &work[pw], top);
                dlacpy("A", top, nblst, &work[pw], top, &B[j * ldb], ldb);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (blk22) {
                        dorm22("R", "N", top, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &B[j * ldb], ldb,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    top, 2 * nnb, 2 * nnb, ONE, &B[j * ldb], ldb,
                                    &work[ppwo], 2 * nnb, ZERO, &work[pw], top);
                        dlacpy("A", top, 2 * nnb, &work[pw], top, &B[j * ldb], ldb);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }

            /* Apply accumulated orthogonal matrices to Z (Fortran lines 834-878) */
            if (wantz) {
                j = ihi - nblst;
                if (initq) {
                    topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                    nh = ihi - topq;
                } else {
                    topq = 0;
                    nh = n;
                }
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nh, nblst, nblst,
                            ONE, &Z[topq + j * ldz], ldz, work, nblst,
                            ZERO, &work[pw], nh);
                dlacpy("A", nh, nblst, &work[pw], nh, &Z[topq + j * ldz], ldz);
                ppwo = nblst * nblst;
                j0 = j - nnb;
                for (j = j0; j >= jcol + 1; j -= nnb) {
                    if (initq) {
                        topq = (2 > j - jcol + 1) ? 2 : j - jcol + 1;
                        nh = ihi - topq;
                    }
                    if (blk22) {
                        dorm22("R", "N", nh, 2 * nnb, nnb, nnb,
                               &work[ppwo], 2 * nnb, &Z[topq + j * ldz], ldz,
                               &work[pw], lwork - pw, &ierr);
                    } else {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    nh, 2 * nnb, 2 * nnb, ONE, &Z[topq + j * ldz], ldz,
                                    &work[ppwo], 2 * nnb, ZERO, &work[pw], nh);
                        dlacpy("A", nh, 2 * nnb, &work[pw], nh, &Z[topq + j * ldz], ldz);
                    }
                    ppwo = ppwo + 4 * nnb * nnb;
                }
            }
        } /* end for jcol (blocked loop) - Fortran line 879 */
    } /* end if blocked vs unblocked - Fortran line 880 */

    /* Unblocked tail (Fortran lines 882-897) */
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
        dgghrd(cq2, cz2, n, jcol + 1, ihi, A, lda, B, ldb, Q, ldq, Z, ldz, &ierr);
    }

    work[0] = (f64)lwkopt;
}
