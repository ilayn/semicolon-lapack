/**
 * @file zlalsd.c
 * @brief ZLALSD uses the singular value decomposition of A to solve
 *        the least squares problem.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZLALSD uses the singular value decomposition of A to solve the least
 * squares problem of finding X to minimize the Euclidean norm of each
 * column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
 * are N-by-NRHS. The solution X overwrites B.
 *
 * The singular values of A smaller than RCOND times the largest
 * singular value are treated as zero in solving the least squares
 * problem; in this case a minimum norm solution is returned.
 * The actual singular values are returned in D in ascending order.
 *
 * @param[in]     uplo    = 'U': D and E define an upper bidiagonal matrix.
 *                        = 'L': D and E define a lower bidiagonal matrix.
 * @param[in]     smlsiz  The maximum size of the subproblems at the bottom of the
 *                        computation tree.
 * @param[in]     n       The dimension of the bidiagonal matrix. n >= 0.
 * @param[in]     nrhs    The number of columns of B. nrhs must be at least 1.
 * @param[in,out] D       Double array, dimension (N).
 *                        On entry D contains the main diagonal of the bidiagonal
 *                        matrix. On exit, if info = 0, D contains its singular values.
 * @param[in,out] E       Double array, dimension (N-1).
 *                        Contains the super-diagonal entries of the bidiagonal matrix.
 *                        On exit, E has been destroyed.
 * @param[in,out] B       Complex array, dimension (ldb, nrhs).
 *                        On input, B contains the right hand sides of the least
 *                        squares problem. On output, B contains the solution X.
 * @param[in]     ldb     The leading dimension of B in the calling subprogram.
 *                        ldb must be at least max(1, N).
 * @param[in]     rcond   The singular values of A less than or equal to RCOND times
 *                        the largest singular value are treated as zero in solving
 *                        the least squares problem. If RCOND is negative,
 *                        machine precision is used instead.
 *                        For example, if diag(S)*X=B were the least squares problem,
 *                        where diag(S) is a diagonal matrix of singular values, the
 *                        solution would be X(i) = B(i) / S(i) if S(i) is greater than
 *                        RCOND*max(S), and X(i) = 0 if S(i) is less than or equal to
 *                        RCOND*max(S).
 * @param[out]    rank    The number of singular values of A greater than RCOND times
 *                        the largest singular value.
 * @param[out]    work    Complex array, dimension (N * NRHS).
 * @param[out]    rwork   Double array, dimension at least
 *                        (9*N + 2*N*SMLSIZ + 8*N*NLVL + 3*SMLSIZ*NRHS +
 *                        max((SMLSIZ+1)**2, N*(1+NRHS) + 2*NRHS)),
 *                        where NLVL = max(0, INT(LOG_2(MIN(M,N)/(SMLSIZ+1))) + 1).
 * @param[out]    iwork   Integer array, dimension at least (3*N*NLVL + 11*N).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 *                        > 0: The algorithm failed to compute a singular value while
 *                             working on the submatrix lying in rows and columns
 *                             INFO/(N+1) through MOD(INFO,N+1).
 */
void zlalsd(const char* uplo, const INT smlsiz, const INT n, const INT nrhs,
            f64* restrict D, f64* restrict E,
            c128* restrict B, const INT ldb, const f64 rcond,
            INT* rank, c128* restrict work,
            f64* restrict rwork, INT* restrict iwork, INT* info)
{
    INT bx, bxst, c_idx, difl_idx, difr_idx, givcol, givnum;
    INT givptr, i, icmpq1, icmpq2, irwb, irwib, irwrb;
    INT irwu, irwvt, irwwrk, iwk, j, jcol, jimag;
    INT jreal, jrow, k_idx, nlvl, nm1, nrwork, nsize, nsub;
    INT perm, poles, s_idx, sizei, smlszp, sqre, st, st1;
    INT u_idx, vt_idx, z_idx;
    f64 cs, eps, orgnrm, r, rcnd, sn, tol;
    const c128 CZERO = CMPLX(0.0, 0.0);

    *info = 0;

    if (n < 0) {
        *info = -3;
    } else if (nrhs < 1) {
        *info = -4;
    } else if (ldb < 1 || ldb < n) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("ZLALSD", -(*info));
        return;
    }

    eps = dlamch("Epsilon");

    if (rcond <= 0.0 || rcond >= 1.0) {
        rcnd = eps;
    } else {
        rcnd = rcond;
    }

    *rank = 0;

    if (n == 0) {
        return;
    } else if (n == 1) {
        if (D[0] == 0.0) {
            zlaset("A", 1, nrhs, CZERO, CZERO, B, ldb);
        } else {
            *rank = 1;
            zlascl("G", 0, 0, D[0], 1.0, 1, nrhs, B, ldb, info);
            D[0] = fabs(D[0]);
        }
        return;
    }

    /* Rotate the matrix if it is lower bidiagonal. */

    if (uplo[0] == 'L' || uplo[0] == 'l') {
        for (i = 0; i < n - 1; i++) {
            dlartg(D[i], E[i], &cs, &sn, &r);
            D[i] = r;
            E[i] = sn * D[i + 1];
            D[i + 1] = cs * D[i + 1];
            if (nrhs == 1) {
                cblas_zdrot(1, &B[i], 1, &B[i + 1], 1, cs, sn);
            } else {
                rwork[i * 2] = cs;
                rwork[i * 2 + 1] = sn;
            }
        }
        if (nrhs > 1) {
            for (i = 0; i < nrhs; i++) {
                for (j = 0; j < n - 1; j++) {
                    cs = rwork[j * 2];
                    sn = rwork[j * 2 + 1];
                    cblas_zdrot(1, &B[j + i * ldb], 1, &B[j + 1 + i * ldb], 1, cs, sn);
                }
            }
        }
    }

    /* Scale. */

    nm1 = n - 1;
    orgnrm = dlanst("M", n, D, E);
    if (orgnrm == 0.0) {
        zlaset("A", n, nrhs, CZERO, CZERO, B, ldb);
        return;
    }

    dlascl("G", 0, 0, orgnrm, 1.0, n, 1, D, n, info);
    dlascl("G", 0, 0, orgnrm, 1.0, nm1, 1, E, nm1, info);

    /* If N is smaller than the minimum divide size SMLSIZ, then solve
     * the problem with another solver. */

    if (n <= smlsiz) {
        irwu = 0;
        irwvt = irwu + n * n;
        irwwrk = irwvt + n * n;
        irwrb = irwwrk;
        irwib = irwrb + n * nrhs;
        irwb = irwib + n * nrhs;
        dlaset("A", n, n, 0.0, 1.0, &rwork[irwu], n);
        dlaset("A", n, n, 0.0, 1.0, &rwork[irwvt], n);
        dlasdq("U", 0, n, n, n, 0, D, E, &rwork[irwvt], n,
               &rwork[irwu], n, NULL, 1, &rwork[irwwrk], info);
        if (*info != 0) {
            return;
        }

        /* In the real version, B is passed to DLASDQ and multiplied
         * internally by Q**H. Here B is complex and that product is
         * computed below in two steps (real and imaginary parts). */

        j = irwb;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                rwork[j] = creal(B[jrow + jcol * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, n, 1.0, &rwork[irwu], n,
                    &rwork[irwb], n, 0.0, &rwork[irwrb], n);
        j = irwb;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                rwork[j] = cimag(B[jrow + jcol * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, n, 1.0, &rwork[irwu], n,
                    &rwork[irwb], n, 0.0, &rwork[irwib], n);
        jreal = irwrb;
        jimag = irwib;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                B[jrow + jcol * ldb] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }

        tol = rcnd * fabs(D[cblas_idamax(n, D, 1)]);
        for (i = 0; i < n; i++) {
            if (D[i] <= tol) {
                zlaset("A", 1, nrhs, CZERO, CZERO, &B[i], ldb);
            } else {
                zlascl("G", 0, 0, D[i], 1.0, 1, nrhs, &B[i], ldb, info);
                *rank = *rank + 1;
            }
        }

        /* Since B is complex, the following call to DGEMM is performed
         * in two steps (real and imaginary parts). That is for V * B
         * (in the real version of the code V**H is stored in WORK). */

        j = irwb;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                rwork[j] = creal(B[jrow + jcol * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, n, 1.0, &rwork[irwvt], n,
                    &rwork[irwb], n, 0.0, &rwork[irwrb], n);
        j = irwb;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                rwork[j] = cimag(B[jrow + jcol * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, nrhs, n, 1.0, &rwork[irwvt], n,
                    &rwork[irwb], n, 0.0, &rwork[irwib], n);
        jreal = irwrb;
        jimag = irwib;
        for (jcol = 0; jcol < nrhs; jcol++) {
            for (jrow = 0; jrow < n; jrow++) {
                B[jrow + jcol * ldb] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }

        /* Unscale. */

        dlascl("G", 0, 0, 1.0, orgnrm, n, 1, D, n, info);
        dlasrt("D", n, D, info);
        zlascl("G", 0, 0, orgnrm, 1.0, n, nrhs, B, ldb, info);

        return;
    }

    /* Book-keeping and setting up some constants. */

    nlvl = (INT)(log((f64)n / (f64)(smlsiz + 1)) / log(2.0)) + 1;

    smlszp = smlsiz + 1;

    u_idx = 0;
    vt_idx = smlsiz * n;
    difl_idx = vt_idx + smlszp * n;
    difr_idx = difl_idx + nlvl * n;
    z_idx = difr_idx + nlvl * n * 2;
    c_idx = z_idx + nlvl * n;
    s_idx = c_idx + n;
    poles = s_idx + n;
    givnum = poles + 2 * nlvl * n;
    nrwork = givnum + 2 * nlvl * n;
    bx = 0;

    irwrb = nrwork;
    irwib = irwrb + smlsiz * nrhs;
    irwb = irwib + smlsiz * nrhs;

    sizei = n;
    k_idx = sizei + n;
    givptr = k_idx + n;
    perm = givptr + n;
    givcol = perm + nlvl * n;
    iwk = givcol + nlvl * n * 2;

    st = 0;
    sqre = 0;
    icmpq1 = 1;
    icmpq2 = 0;
    nsub = 0;

    for (i = 0; i < n; i++) {
        if (fabs(D[i]) < eps) {
            D[i] = copysign(eps, D[i]);
        }
    }

    for (i = 0; i < nm1; i++) {
        if (fabs(E[i]) < eps || i == nm1 - 1) {
            nsub = nsub + 1;
            iwork[nsub - 1] = st;

            if (i < nm1 - 1) {
                nsize = i - st + 1;
                iwork[sizei + nsub - 1] = nsize;
            } else if (fabs(E[i]) >= eps) {
                nsize = n - st;
                iwork[sizei + nsub - 1] = nsize;
            } else {
                nsize = i - st + 1;
                iwork[sizei + nsub - 1] = nsize;
                nsub = nsub + 1;
                iwork[nsub - 1] = n - 1;
                iwork[sizei + nsub - 1] = 1;
                cblas_zcopy(nrhs, &B[n - 1], ldb, &work[bx + n - 1], n);
            }
            st1 = st;
            if (nsize == 1) {
                cblas_zcopy(nrhs, &B[st], ldb, &work[bx + st1], n);
            } else if (nsize <= smlsiz) {
                dlaset("A", nsize, nsize, 0.0, 1.0, &rwork[vt_idx + st1], n);
                dlaset("A", nsize, nsize, 0.0, 1.0, &rwork[u_idx + st1], n);
                dlasdq("U", 0, nsize, nsize, nsize, 0, &D[st], &E[st],
                       &rwork[vt_idx + st1], n, &rwork[u_idx + st1], n,
                       NULL, 1, &rwork[nrwork], info);
                if (*info != 0) {
                    return;
                }

                /* In the real version, B is passed to DLASDQ and multiplied
                 * internally by Q**H. Here B is complex and that product is
                 * computed below in two steps (real and imaginary parts). */

                j = irwb;
                for (jcol = 0; jcol < nrhs; jcol++) {
                    for (jrow = st; jrow < st + nsize; jrow++) {
                        rwork[j] = creal(B[jrow + jcol * ldb]);
                        j++;
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            nsize, nrhs, nsize, 1.0, &rwork[u_idx + st1], n,
                            &rwork[irwb], nsize, 0.0, &rwork[irwrb], nsize);
                j = irwb;
                for (jcol = 0; jcol < nrhs; jcol++) {
                    for (jrow = st; jrow < st + nsize; jrow++) {
                        rwork[j] = cimag(B[jrow + jcol * ldb]);
                        j++;
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            nsize, nrhs, nsize, 1.0, &rwork[u_idx + st1], n,
                            &rwork[irwb], nsize, 0.0, &rwork[irwib], nsize);
                jreal = irwrb;
                jimag = irwib;
                for (jcol = 0; jcol < nrhs; jcol++) {
                    for (jrow = st; jrow < st + nsize; jrow++) {
                        B[jrow + jcol * ldb] = CMPLX(rwork[jreal], rwork[jimag]);
                        jreal++;
                        jimag++;
                    }
                }

                zlacpy("A", nsize, nrhs, &B[st], ldb, &work[bx + st1], n);
            } else {
                dlasda(icmpq1, smlsiz, nsize, sqre, &D[st], &E[st],
                       &rwork[u_idx + st1], n, &rwork[vt_idx + st1],
                       &iwork[k_idx + st1], &rwork[difl_idx + st1],
                       &rwork[difr_idx + st1], &rwork[z_idx + st1],
                       &rwork[poles + st1], &iwork[givptr + st1],
                       &iwork[givcol + st1], n, &iwork[perm + st1],
                       &rwork[givnum + st1], &rwork[c_idx + st1],
                       &rwork[s_idx + st1], &rwork[nrwork], &iwork[iwk], info);
                if (*info != 0) {
                    return;
                }
                bxst = bx + st1;
                zlalsa(icmpq2, smlsiz, nsize, nrhs, &B[st], ldb,
                       &work[bxst], n, &rwork[u_idx + st1], n,
                       &rwork[vt_idx + st1], &iwork[k_idx + st1],
                       &rwork[difl_idx + st1], &rwork[difr_idx + st1],
                       &rwork[z_idx + st1], &rwork[poles + st1],
                       &iwork[givptr + st1], &iwork[givcol + st1], n,
                       &iwork[perm + st1], &rwork[givnum + st1],
                       &rwork[c_idx + st1], &rwork[s_idx + st1],
                       &rwork[nrwork], &iwork[iwk], info);
                if (*info != 0) {
                    return;
                }
            }
            st = i + 1;
        }
    }

    /* Apply the singular values and treat the tiny ones as zero. */

    tol = rcnd * fabs(D[cblas_idamax(n, D, 1)]);

    for (i = 0; i < n; i++) {
        if (fabs(D[i]) <= tol) {
            zlaset("A", 1, nrhs, CZERO, CZERO, &work[bx + i], n);
        } else {
            *rank = *rank + 1;
            zlascl("G", 0, 0, D[i], 1.0, 1, nrhs, &work[bx + i], n, info);
        }
        D[i] = fabs(D[i]);
    }

    /* Now apply back the right singular vectors. */

    icmpq2 = 1;
    for (i = 0; i < nsub; i++) {
        st = iwork[i];
        st1 = st;
        nsize = iwork[sizei + i];
        bxst = bx + st1;
        if (nsize == 1) {
            cblas_zcopy(nrhs, &work[bxst], n, &B[st], ldb);
        } else if (nsize <= smlsiz) {

            /* Since B and BX are complex, the following call to DGEMM
             * is performed in two steps (real and imaginary parts). */

            j = bxst - n;
            jreal = irwb;
            for (jcol = 0; jcol < nrhs; jcol++) {
                j = j + n;
                for (jrow = 0; jrow < nsize; jrow++) {
                    rwork[jreal] = creal(work[j + jrow]);
                    jreal++;
                }
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        nsize, nrhs, nsize, 1.0, &rwork[vt_idx + st1], n,
                        &rwork[irwb], nsize, 0.0, &rwork[irwrb], nsize);
            j = bxst - n;
            jimag = irwb;
            for (jcol = 0; jcol < nrhs; jcol++) {
                j = j + n;
                for (jrow = 0; jrow < nsize; jrow++) {
                    rwork[jimag] = cimag(work[j + jrow]);
                    jimag++;
                }
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        nsize, nrhs, nsize, 1.0, &rwork[vt_idx + st1], n,
                        &rwork[irwb], nsize, 0.0, &rwork[irwib], nsize);
            jreal = irwrb;
            jimag = irwib;
            for (jcol = 0; jcol < nrhs; jcol++) {
                for (jrow = st; jrow < st + nsize; jrow++) {
                    B[jrow + jcol * ldb] = CMPLX(rwork[jreal], rwork[jimag]);
                    jreal++;
                    jimag++;
                }
            }
        } else {
            zlalsa(icmpq2, smlsiz, nsize, nrhs, &work[bxst], n,
                   &B[st], ldb, &rwork[u_idx + st1], n,
                   &rwork[vt_idx + st1], &iwork[k_idx + st1],
                   &rwork[difl_idx + st1], &rwork[difr_idx + st1],
                   &rwork[z_idx + st1], &rwork[poles + st1],
                   &iwork[givptr + st1], &iwork[givcol + st1], n,
                   &iwork[perm + st1], &rwork[givnum + st1],
                   &rwork[c_idx + st1], &rwork[s_idx + st1],
                   &rwork[nrwork], &iwork[iwk], info);
            if (*info != 0) {
                return;
            }
        }
    }

    /* Unscale and sort the singular values. */

    dlascl("G", 0, 0, 1.0, orgnrm, n, 1, D, n, info);
    dlasrt("D", n, D, info);
    zlascl("G", 0, 0, orgnrm, 1.0, n, nrhs, B, ldb, info);
}
