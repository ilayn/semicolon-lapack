/**
 * @file ctrsyl3.c
 * @brief CTRSYL3 solves the complex Sylvester matrix equation (blocked version).
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CTRSYL3 solves the complex Sylvester matrix equation:
 *
 *    op(A)*X + X*op(B) = scale*C or
 *    op(A)*X - X*op(B) = scale*C,
 *
 * where op(A) = A or A**H, and A and B are both upper triangular. A is
 * M-by-M and B is N-by-N; the right hand side C and the solution X are
 * M-by-N; and scale is an output scale factor, set <= 1 to avoid
 * overflow in X.
 *
 * This is the block version of the algorithm.
 *
 * @param[in] trana  Specifies the option op(A):
 *                   = 'N': op(A) = A    (No transpose)
 *                   = 'C': op(A) = A**H (Conjugate transpose)
 * @param[in] tranb  Specifies the option op(B):
 *                   = 'N': op(B) = B    (No transpose)
 *                   = 'C': op(B) = B**H (Conjugate transpose)
 * @param[in] isgn   Specifies the sign in the equation:
 *                   = +1: solve op(A)*X + X*op(B) = scale*C
 *                   = -1: solve op(A)*X - X*op(B) = scale*C
 * @param[in] m      The order of the matrix A, and the number of rows in the
 *                   matrices X and C. m >= 0.
 * @param[in] n      The order of the matrix B, and the number of columns in the
 *                   matrices X and C. n >= 0.
 * @param[in] A      Complex*16 array, dimension (lda, m).
 *                   The upper triangular matrix A.
 * @param[in] lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[in] B      Complex*16 array, dimension (ldb, n).
 *                   The upper triangular matrix B.
 * @param[in] ldb    The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] C  Complex*16 array, dimension (ldc, n).
 *                   On entry, the M-by-N right hand side matrix C.
 *                   On exit, C is overwritten by the solution matrix X.
 * @param[in] ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out] scale The scale factor, scale, set <= 1 to avoid overflow in X.
 * @param[out] swork Single precision array, dimension (MAX(2, ROWS), MAX(1, COLS)).
 *                   On exit, if info = 0, swork[0] returns the optimal value ROWS
 *                   and swork[1] returns the optimal COLS.
 * @param[in] ldswork The leading dimension of the array swork.
 *                    ldswork >= MAX(2, ROWS) + MAX(m, n),
 *                    where ROWS = ((m + nb - 1) / nb + 1).
 *                    If ldswork = -1, then a workspace query is assumed.
 * @param[out] info
 *                   = 0: successful exit
 *                   < 0: if info = -i, the i-th argument had an illegal value
 *                   = 1: A and B have common or very close eigenvalues; perturbed
 *                        values were used to solve the equation (but the matrices
 *                        A and B are unchanged).
 */
void ctrsyl3(const char* trana, const char* tranb, const int isgn,
             const int m, const int n,
             const c64* A, const int lda,
             const c64* B, const int ldb,
             c64* C, const int ldc,
             f32* scale,
             f32* swork, const int ldswork,
             int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int notrna, notrnb, lquery;
    int awrk, bwrk, i, i1, i2, iinfo, j, j1, j2, jj;
    int k, k1, k2, l, l1, l2, ll, nba, nb, nbb;
    f32 anrm, bignum, bnrm, cnrm, scal, scaloc;
    f32 scamin, sgn, xnrm, buf, smlnum;
    c64 csgn;

    int max_mn = (m > n) ? m : n;
    if (max_mn < 1) max_mn = 1;

    /* Decode and Test input parameters */
    notrna = (trana[0] == 'N' || trana[0] == 'n');
    notrnb = (tranb[0] == 'N' || tranb[0] == 'n');

    /* Use the same block size for all matrices.
     * From ilaenv.f lines 477-485: for CTRSYL (complex precision),
     * NB = MIN(MAX(24, INT(MIN(N1,N2)*8/100)), 80)
     * with a minimum of 8 from ctrsyl3.f line 221.
     */
    {
        int min_mn_local = (m < n) ? m : n;
        int nb_calc = (min_mn_local * 8) / 100;
        if (nb_calc < 24) nb_calc = 24;
        if (nb_calc > 80) nb_calc = 80;
        nb = (nb_calc > 8) ? nb_calc : 8;
    }

    /* Compute number of blocks in A and B */
    nba = (m + nb - 1) / nb;
    if (nba < 1) nba = 1;
    nbb = (n + nb - 1) / nb;
    if (nbb < 1) nbb = 1;

    /* Compute workspace */
    *info = 0;
    lquery = (ldswork == -1);
    if (lquery) {
        swork[0] = ((nba > nbb) ? nba : nbb) + max_mn;
        swork[1] = 2 * nbb + nba;
    }

    /* Test the input arguments */
    if (!notrna && !(trana[0] == 'C' || trana[0] == 'c')) {
        *info = -1;
    } else if (!notrnb && !(tranb[0] == 'C' || tranb[0] == 'c')) {
        *info = -2;
    } else if (isgn != 1 && isgn != -1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("CTRSYL3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *scale = ONE;
    if (m == 0 || n == 0)
        return;

    /* Use unblocked code for small problems or if insufficient
     * workspace is provided
     */
    {
        int min_nba_nbb = (nba < nbb) ? nba : nbb;
        int max_nba_nbb = (nba > nbb) ? nba : nbb;
        if (min_nba_nbb == 1 || ldswork < max_nba_nbb + max_mn) {
            ctrsyl(trana, tranb, isgn, m, n, A, lda, B, ldb,
                   C, ldc, scale, info);
            return;
        }
    }

    /* Use the tail of column 0 in swork as clange workspace */
    int rows = (nba > nbb) ? nba : nbb;
    f32* wnrm = &swork[rows];

    /* Set constants to control overflow */
    smlnum = slamch("S");
    bignum = ONE / smlnum;

    /* Set local scaling factors. */
    for (l = 0; l < nbb; l++) {
        for (k = 0; k < nba; k++) {
            swork[k + l * ldswork] = ONE;
        }
    }

    /* Fallback scaling factor to prevent flushing of SWORK(K,L) to zero.
     * This scaling is to ensure compatibility with TRSYL and may get flushed.
     */
    buf = ONE;

    /* Compute upper bounds of blocks of A and B */
    awrk = nbb;
    for (k = 0; k < nba; k++) {
        k1 = k * nb;
        k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
        for (l = k; l < nba; l++) {
            l1 = l * nb;
            l2 = ((l + 1) * nb < m) ? (l + 1) * nb : m;
            if (notrna) {
                swork[k + (awrk + l) * ldswork] = clange("I", k2 - k1, l2 - l1,
                                                          &A[k1 + l1 * lda], lda, wnrm);
            } else {
                swork[l + (awrk + k) * ldswork] = clange("1", k2 - k1, l2 - l1,
                                                          &A[k1 + l1 * lda], lda, wnrm);
            }
        }
    }
    bwrk = nbb + nba;
    for (k = 0; k < nbb; k++) {
        k1 = k * nb;
        k2 = ((k + 1) * nb < n) ? (k + 1) * nb : n;
        for (l = k; l < nbb; l++) {
            l1 = l * nb;
            l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;
            if (notrnb) {
                swork[k + (bwrk + l) * ldswork] = clange("I", k2 - k1, l2 - l1,
                                                          &B[k1 + l1 * ldb], ldb, wnrm);
            } else {
                swork[l + (bwrk + k) * ldswork] = clange("1", k2 - k1, l2 - l1,
                                                          &B[k1 + l1 * ldb], ldb, wnrm);
            }
        }
    }

    sgn = (f32)isgn;
    csgn = CMPLXF(sgn, ZERO);

    if (notrna && notrnb) {

        /* Solve    A*X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-left corner column by column by
         *
         *  A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *           M                         L-1
         * R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)].
         *         I=K+1                       J=1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = nba - 1; k >= 0; k--) {

            k1 = k * nb;
            k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
            for (l = 0; l < nbb; l++) {

                l1 = l * nb;
                l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;

                ctrsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        buf = ZERO;
                    } else {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = clange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, wnrm);

                for (i = k - 1; i >= 0; i--) {

                    /* C(I,L) := C(I,L) - A(I,K) * C(K,L) */

                    i1 = i * nb;
                    i2 = ((i + 1) * nb < m) ? (i + 1) * nb : m;

                    cnrm = clange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = l1; jj < l2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_cone = CMPLXF(-1.0f, 0.0f);
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i2 - i1, l2 - l1, k2 - k1, &neg_cone,
                                    &A[i1 + k1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                    &CONE, &C[i1 + l1 * ldc], ldc);
                    }

                }

                for (j = l + 1; j < nbb; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(L,J) */

                    j1 = j * nb;
                    j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

                    cnrm = clange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_csgn = -csgn;
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k2 - k1, j2 - j1, l2 - l1, &neg_csgn,
                                    &C[k1 + l1 * ldc], ldc, &B[l1 + j1 * ldb], ldb,
                                    &CONE, &C[k1 + j1 * ldc], ldc);
                    }
                }
            }
        }

    } else if (!notrna && notrnb) {

        /* Solve    A**H *X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * upper-left corner column by column by
         *
         *   A(K,K)**H*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *            K-1                        L-1
         *   R(K,L) = SUM [A(I,K)**H*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]
         *            I=1                        J=1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = 0; k < nba; k++) {

            k1 = k * nb;
            k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
            for (l = 0; l < nbb; l++) {

                l1 = l * nb;
                l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;

                ctrsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        buf = ZERO;
                    } else {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = clange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, wnrm);

                for (i = k + 1; i < nba; i++) {

                    /* C(I,L) := C(I,L) - A(K,I)**H * C(K,L) */

                    i1 = i * nb;
                    i2 = ((i + 1) * nb < m) ? (i + 1) * nb : m;

                    cnrm = clange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_cone = CMPLXF(-1.0f, 0.0f);
                        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    i2 - i1, l2 - l1, k2 - k1, &neg_cone,
                                    &A[k1 + i1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                    &CONE, &C[i1 + l1 * ldc], ldc);
                    }

                }

                for (j = l + 1; j < nbb; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(L,J) */

                    j1 = j * nb;
                    j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

                    cnrm = clange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_csgn = -csgn;
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k2 - k1, j2 - j1, l2 - l1, &neg_csgn,
                                    &C[k1 + l1 * ldc], ldc, &B[l1 + j1 * ldb], ldb,
                                    &CONE, &C[k1 + j1 * ldc], ldc);
                    }
                }
            }
        }

    } else if (!notrna && !notrnb) {

        /* Solve    A**H *X + ISGN*X*B**H = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * top-right corner column by column by
         *
         *    A(K,K)**H*X(K,L) + ISGN*X(K,L)*B(L,L)**H = C(K,L) - R(K,L)
         *
         * Where
         *              K-1                          N
         *     R(K,L) = SUM [A(I,K)**H*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**H].
         *              I=1                        J=L+1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = 0; k < nba; k++) {

            k1 = k * nb;
            k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
            for (l = nbb - 1; l >= 0; l--) {

                l1 = l * nb;
                l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;

                ctrsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        buf = ZERO;
                    } else {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = clange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, wnrm);

                for (i = k + 1; i < nba; i++) {

                    /* C(I,L) := C(I,L) - A(K,I)**H * C(K,L) */

                    i1 = i * nb;
                    i2 = ((i + 1) * nb < m) ? (i + 1) * nb : m;

                    cnrm = clange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_cone = CMPLXF(-1.0f, 0.0f);
                        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    i2 - i1, l2 - l1, k2 - k1, &neg_cone,
                                    &A[k1 + i1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                    &CONE, &C[i1 + l1 * ldc], ldc);
                    }

                }

                for (j = 0; j < l; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(J,L)**H */

                    j1 = j * nb;
                    j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

                    cnrm = clange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_csgn = -csgn;
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    k2 - k1, j2 - j1, l2 - l1, &neg_csgn,
                                    &C[k1 + l1 * ldc], ldc, &B[j1 + l1 * ldb], ldb,
                                    &CONE, &C[k1 + j1 * ldc], ldc);
                    }
                }
            }
        }

    } else if (notrna && !notrnb) {

        /* Solve    A*X + ISGN*X*B**H = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-right corner column by column by
         *
         *     A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)**H = C(K,L) - R(K,L)
         *
         * Where
         *               M                          N
         *     R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**H].
         *             I=K+1                      J=L+1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = nba - 1; k >= 0; k--) {

            k1 = k * nb;
            k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
            for (l = nbb - 1; l >= 0; l--) {

                l1 = l * nb;
                l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;

                ctrsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        buf = ZERO;
                    } else {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = clange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, wnrm);

                for (i = 0; i < k; i++) {

                    /* C(I,L) := C(I,L) - A(I,K) * C(K,L) */

                    i1 = i * nb;
                    i2 = ((i + 1) * nb < m) ? (i + 1) * nb : m;

                    cnrm = clange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = l1; jj < l2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_csscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_cone = CMPLXF(-1.0f, 0.0f);
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    i2 - i1, l2 - l1, k2 - k1, &neg_cone,
                                    &A[i1 + k1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                    &CONE, &C[i1 + l1 * ldc], ldc);
                    }

                }

                for (j = 0; j < l; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(J,L)**H */

                    j1 = j * nb;
                    j2 = ((j + 1) * nb < n) ? (j + 1) * nb : n;

                    cnrm = clange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, wnrm);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                f32 tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = l1; jj < l2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_csscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    {
                        const c64 neg_csgn = -csgn;
                        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    k2 - k1, j2 - j1, l2 - l1, &neg_csgn,
                                    &C[k1 + l1 * ldc], ldc, &B[j1 + l1 * ldb], ldb,
                                    &CONE, &C[k1 + j1 * ldc], ldc);
                    }
                }
            }
        }
    }


    /* Reduce local scaling factors */

    *scale = swork[0];
    for (k = 0; k < nba; k++) {
        for (l = 0; l < nbb; l++) {
            if (*scale > swork[k + l * ldswork])
                *scale = swork[k + l * ldswork];
        }
    }
    if (*scale == ZERO) {

        swork[0] = (f32)((nba > nbb) ? nba : nbb);
        swork[1] = (f32)(2 * nbb + nba);
        return;
    }

    /* Realize consistent scaling */

    for (k = 0; k < nba; k++) {
        k1 = k * nb;
        k2 = ((k + 1) * nb < m) ? (k + 1) * nb : m;
        for (l = 0; l < nbb; l++) {
            l1 = l * nb;
            l2 = ((l + 1) * nb < n) ? (l + 1) * nb : n;
            scal = *scale / swork[k + l * ldswork];
            if (scal != ONE) {
                for (ll = l1; ll < l2; ll++) {
                    cblas_csscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                }
            }
        }
    }

    if (buf != ONE && buf > ZERO) {

        scaloc = (*scale / smlnum < ONE / buf) ? *scale / smlnum : ONE / buf;
        buf = buf * scaloc;
        *scale = *scale / scaloc;
    }

    if (buf != ONE && buf > ZERO) {

        scal = fabsf(crealf(C[0]));
        {
            f32 tmp = fabsf(cimagf(C[0]));
            if (tmp > scal) scal = tmp;
        }
        for (k = 0; k < m; k++) {
            for (l = 0; l < n; l++) {
                f32 tmp = fabsf(crealf(C[k + l * ldc]));
                if (tmp > scal) scal = tmp;
                tmp = fabsf(cimagf(C[k + l * ldc]));
                if (tmp > scal) scal = tmp;
            }
        }

        scaloc = (bignum / scal < ONE / buf) ? bignum / scal : ONE / buf;
        buf = buf * scaloc;
        clascl("G", -1, -1, ONE, scaloc, m, n, C, ldc, &iinfo);
    }

    *scale = *scale * buf;

    /* Restore workspace dimensions */

    swork[0] = (f32)((nba > nbb) ? nba : nbb);
    swork[1] = (f32)(2 * nbb + nba);
}
