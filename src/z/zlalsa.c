/**
 * @file zlalsa.c
 * @brief ZLALSA computes the SVD of the coefficient matrix in compact form.
 *        Used by zgelsd.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZLALSA is an intermediate step in solving the least squares problem
 * by computing the SVD of the coefficient matrix in compact form (The
 * singular vectors are computed as products of simple orthogonal
 * matrices.).
 *
 * If ICOMPQ = 0, ZLALSA applies the inverse of the left singular vector
 * matrix of an upper bidiagonal matrix to the right hand side; and if
 * ICOMPQ = 1, ZLALSA applies the right singular vector matrix to the
 * right hand side. The singular vector matrices were generated in
 * compact form by ZLALSA.
 *
 * @param[in]     icompq  Specifies whether the left or the right singular vector
 *                        matrix is involved.
 *                        = 0: Left singular vector matrix
 *                        = 1: Right singular vector matrix
 * @param[in]     smlsiz  The maximum size of the subproblems at the bottom of the
 *                        computation tree.
 * @param[in]     n       The row and column dimensions of the upper bidiagonal matrix.
 * @param[in]     nrhs    The number of columns of B and BX. nrhs must be at least 1.
 * @param[in,out] B       Complex array, dimension (ldb, nrhs).
 *                        On input, B contains the right hand sides of the least
 *                        squares problem in rows 1 through M.
 *                        On output, B contains the solution X in rows 1 through N.
 * @param[in]     ldb     The leading dimension of B in the calling subprogram.
 *                        ldb must be at least max(1, max(M, N)).
 * @param[out]    BX      Complex array, dimension (ldbx, nrhs).
 *                        On exit, the result of applying the left or right singular
 *                        vector matrix to B.
 * @param[in]     ldbx    The leading dimension of BX.
 * @param[in]     U       Double array, dimension (ldu, smlsiz).
 *                        On entry, U contains the left singular vector matrices of all
 *                        subproblems at the bottom level.
 * @param[in]     ldu     The leading dimension of arrays U, VT, DIFL, DIFR,
 *                        POLES, GIVNUM, and Z. ldu >= N.
 * @param[in]     VT      Double array, dimension (ldu, smlsiz+1).
 *                        On entry, VT**H contains the right singular vector matrices of
 *                        all subproblems at the bottom level.
 * @param[in]     K       Integer array, dimension (N).
 * @param[in]     difl    Double array, dimension (ldu, nlvl).
 *                        where NLVL = INT(log_2 (N/(SMLSIZ+1))) + 1.
 * @param[in]     difr    Double array, dimension (ldu, 2 * nlvl).
 *                        On entry, DIFL(*, I) and DIFR(*, 2 * I - 1) record
 *                        distances between singular values on the I-th level and
 *                        singular values on the (I-1)-th level, and DIFR(*, 2 * I)
 *                        record the normalizing factors of the right singular vectors
 *                        matrices of subproblems on I-th level.
 * @param[in]     Z       Double array, dimension (ldu, nlvl).
 *                        On entry, Z(1, I) contains the components of the deflation-
 *                        adjusted updating row vector for subproblems on the I-th
 *                        level.
 * @param[in]     poles   Double array, dimension (ldu, 2 * nlvl).
 *                        On entry, POLES(*, 2 * I - 1: 2 * I) contains the new and old
 *                        singular values involved in the secular equations on the I-th
 *                        level.
 * @param[in]     givptr  Integer array, dimension (N).
 *                        On entry, GIVPTR(I) records the number of Givens
 *                        rotations performed on the I-th problem on the computation
 *                        tree.
 * @param[in]     givcol  Integer array, dimension (ldgcol, 2 * nlvl).
 *                        On entry, for each I, GIVCOL(*, 2 * I - 1: 2 * I) records the
 *                        locations of Givens rotations performed on the I-th level on
 *                        the computation tree.
 * @param[in]     ldgcol  The leading dimension of arrays GIVCOL and PERM.
 *                        ldgcol >= N.
 * @param[in]     perm    Integer array, dimension (ldgcol, nlvl).
 *                        On entry, PERM(*, I) records permutations done on the I-th
 *                        level of the computation tree.
 * @param[in]     givnum  Double array, dimension (ldu, 2 * nlvl).
 *                        On entry, GIVNUM(*, 2 * I - 1: 2 * I) records the C- and S-
 *                        values of Givens rotations performed on the I-th level on the
 *                        computation tree.
 * @param[in]     C       Double array, dimension (N).
 *                        On entry, if the I-th subproblem is not square,
 *                        C(I) contains the C-value of a Givens rotation related to
 *                        the right null space of the I-th subproblem.
 * @param[in]     S       Double array, dimension (N).
 *                        On entry, if the I-th subproblem is not square,
 *                        S(I) contains the S-value of a Givens rotation related to
 *                        the right null space of the I-th subproblem.
 * @param[out]    rwork   Double array, dimension at least
 *                        max((SMLSIZ+1)*NRHS*3, N*(1+NRHS) + 2*NRHS).
 * @param[out]    iwork   Integer array, dimension (3*N).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 */
void zlalsa(const INT icompq, const INT smlsiz, const INT n, const INT nrhs,
            c128* restrict B, const INT ldb,
            c128* restrict BX, const INT ldbx,
            const f64* restrict U, const INT ldu,
            const f64* restrict VT, const INT* restrict K,
            const f64* restrict difl, const f64* restrict difr,
            const f64* restrict Z, const f64* restrict poles,
            const INT* restrict givptr, const INT* restrict givcol,
            const INT ldgcol, const INT* restrict perm,
            const f64* restrict givnum,
            const f64* restrict C, const f64* restrict S,
            f64* restrict rwork, INT* restrict iwork, INT* info)
{
    INT i, i1, ic, im1, inode, j, jcol, jimag, jreal;
    INT jrow, lf, ll, lvl, lvl2, nd, ndb1, ndiml;
    INT ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre;

    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (smlsiz < 3) {
        *info = -2;
    } else if (n < smlsiz) {
        *info = -3;
    } else if (nrhs < 1) {
        *info = -4;
    } else if (ldb < n) {
        *info = -6;
    } else if (ldbx < n) {
        *info = -8;
    } else if (ldu < n) {
        *info = -10;
    } else if (ldgcol < n) {
        *info = -19;
    }
    if (*info != 0) {
        xerbla("ZLALSA", -(*info));
        return;
    }

    /* Book-keeping and setting up the computation tree. */

    inode = 0;
    ndiml = n;
    ndimr = ndiml + n;

    dlasdt(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], smlsiz);

    if (icompq == 1) {
        goto L170;
    }

    /* The nodes on the bottom level of the tree were solved
     * by DLASDQ. The corresponding left and right singular vector
     * matrices are in explicit form. First apply back the left
     * singular vector matrices. */

    ndb1 = (nd + 1) / 2;
    for (i = ndb1; i <= nd; i++) {
        i1 = i - 1;
        ic = iwork[inode + i1];
        nl = iwork[ndiml + i1];
        nr = iwork[ndimr + i1];
        nlf = ic - nl;
        nrf = ic + 1;

        /* Since B and BX are complex, the following call to DGEMM
         * is performed in two steps (real and imaginary parts). */

        j = nl * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nl - 1; jrow++) {
                rwork[j] = creal(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nl, nrhs, nl, 1.0, &U[nlf], ldu,
                    &rwork[nl * nrhs * 2], nl, 0.0, rwork, nl);
        j = nl * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nl - 1; jrow++) {
                rwork[j] = cimag(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nl, nrhs, nl, 1.0, &U[nlf], ldu,
                    &rwork[nl * nrhs * 2], nl, 0.0, &rwork[nl * nrhs], nl);
        jreal = 0;
        jimag = nl * nrhs;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nl - 1; jrow++) {
                BX[jrow + (jcol - 1) * ldbx] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }

        /* Since B and BX are complex, the following call to DGEMM
         * is performed in two steps (real and imaginary parts). */

        j = nr * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nr - 1; jrow++) {
                rwork[j] = creal(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nr, nrhs, nr, 1.0, &U[nrf], ldu,
                    &rwork[nr * nrhs * 2], nr, 0.0, rwork, nr);
        j = nr * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nr - 1; jrow++) {
                rwork[j] = cimag(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nr, nrhs, nr, 1.0, &U[nrf], ldu,
                    &rwork[nr * nrhs * 2], nr, 0.0, &rwork[nr * nrhs], nr);
        jreal = 0;
        jimag = nr * nrhs;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nr - 1; jrow++) {
                BX[jrow + (jcol - 1) * ldbx] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }
    }

    /* Next copy the rows of B that correspond to unchanged rows
     * in the bidiagonal matrix to BX. */

    for (i = 1; i <= nd; i++) {
        ic = iwork[inode + i - 1];
        cblas_zcopy(nrhs, &B[ic], ldb, &BX[ic], ldbx);
    }

    /* Finally go through the left singular vector matrices of all
     * the other subproblems bottom-up on the tree. */

    j = 1 << nlvl;
    sqre = 0;

    for (lvl = nlvl; lvl >= 1; lvl--) {
        lvl2 = 2 * lvl - 1;

        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }
        for (i = lf; i <= ll; i++) {
            im1 = i - 1;
            ic = iwork[inode + im1];
            nl = iwork[ndiml + im1];
            nr = iwork[ndimr + im1];
            nlf = ic - nl;
            j = j - 1;
            zlals0(icompq, nl, nr, sqre, nrhs,
                   &BX[nlf], ldbx, &B[nlf], ldb,
                   &perm[nlf + (lvl - 1) * ldgcol],
                   givptr[j - 1],
                   &givcol[nlf + (lvl2 - 1) * ldgcol], ldgcol,
                   &givnum[nlf + (lvl2 - 1) * ldu], ldu,
                   &poles[nlf + (lvl2 - 1) * ldu],
                   &difl[nlf + (lvl - 1) * ldu],
                   &difr[nlf + (lvl2 - 1) * ldu],
                   &Z[nlf + (lvl - 1) * ldu],
                   K[j - 1], C[j - 1], S[j - 1], rwork, info);
        }
    }
    goto L330;

    /* ICOMPQ = 1: applying back the right singular vector factors. */

L170:

    /* First now go through the right singular vector matrices of all
     * the tree nodes top-down. */

    j = 0;
    for (lvl = 1; lvl <= nlvl; lvl++) {
        lvl2 = 2 * lvl - 1;

        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }
        for (i = ll; i >= lf; i--) {
            im1 = i - 1;
            ic = iwork[inode + im1];
            nl = iwork[ndiml + im1];
            nr = iwork[ndimr + im1];
            nlf = ic - nl;
            if (i == ll) {
                sqre = 0;
            } else {
                sqre = 1;
            }
            j = j + 1;
            zlals0(icompq, nl, nr, sqre, nrhs,
                   &B[nlf], ldb, &BX[nlf], ldbx,
                   &perm[nlf + (lvl - 1) * ldgcol],
                   givptr[j - 1],
                   &givcol[nlf + (lvl2 - 1) * ldgcol], ldgcol,
                   &givnum[nlf + (lvl2 - 1) * ldu], ldu,
                   &poles[nlf + (lvl2 - 1) * ldu],
                   &difl[nlf + (lvl - 1) * ldu],
                   &difr[nlf + (lvl2 - 1) * ldu],
                   &Z[nlf + (lvl - 1) * ldu],
                   K[j - 1], C[j - 1], S[j - 1], rwork, info);
        }
    }

    /* The nodes on the bottom level of the tree were solved
     * by DLASDQ. The corresponding right singular vector
     * matrices are in explicit form. Apply them back. */

    ndb1 = (nd + 1) / 2;
    for (i = ndb1; i <= nd; i++) {
        i1 = i - 1;
        ic = iwork[inode + i1];
        nl = iwork[ndiml + i1];
        nr = iwork[ndimr + i1];
        nlp1 = nl + 1;
        if (i == nd) {
            nrp1 = nr;
        } else {
            nrp1 = nr + 1;
        }
        nlf = ic - nl;
        nrf = ic + 1;

        /* Since B and BX are complex, the following call to DGEMM is
         * performed in two steps (real and imaginary parts). */

        j = nlp1 * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nlp1 - 1; jrow++) {
                rwork[j] = creal(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nlp1, nrhs, nlp1, 1.0, &VT[nlf], ldu,
                    &rwork[nlp1 * nrhs * 2], nlp1, 0.0, rwork, nlp1);
        j = nlp1 * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nlp1 - 1; jrow++) {
                rwork[j] = cimag(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nlp1, nrhs, nlp1, 1.0, &VT[nlf], ldu,
                    &rwork[nlp1 * nrhs * 2], nlp1, 0.0, &rwork[nlp1 * nrhs], nlp1);
        jreal = 0;
        jimag = nlp1 * nrhs;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nlf; jrow <= nlf + nlp1 - 1; jrow++) {
                BX[jrow + (jcol - 1) * ldbx] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }

        /* Since B and BX are complex, the following call to DGEMM is
         * performed in two steps (real and imaginary parts). */

        j = nrp1 * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nrp1 - 1; jrow++) {
                rwork[j] = creal(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nrp1, nrhs, nrp1, 1.0, &VT[nrf], ldu,
                    &rwork[nrp1 * nrhs * 2], nrp1, 0.0, rwork, nrp1);
        j = nrp1 * nrhs * 2;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nrp1 - 1; jrow++) {
                rwork[j] = cimag(B[jrow + (jcol - 1) * ldb]);
                j++;
            }
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    nrp1, nrhs, nrp1, 1.0, &VT[nrf], ldu,
                    &rwork[nrp1 * nrhs * 2], nrp1, 0.0, &rwork[nrp1 * nrhs], nrp1);
        jreal = 0;
        jimag = nrp1 * nrhs;
        for (jcol = 1; jcol <= nrhs; jcol++) {
            for (jrow = nrf; jrow <= nrf + nrp1 - 1; jrow++) {
                BX[jrow + (jcol - 1) * ldbx] = CMPLX(rwork[jreal], rwork[jimag]);
                jreal++;
                jimag++;
            }
        }
    }

L330:
    return;
}
