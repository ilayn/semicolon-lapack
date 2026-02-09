/**
 * @file strsyl3.c
 * @brief STRSYL3 solves the real Sylvester matrix equation (blocked version).
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include <cblas.h>

/**
 * STRSYL3 solves the real Sylvester matrix equation:
 *
 *    op(A)*X + X*op(B) = scale*C or
 *    op(A)*X - X*op(B) = scale*C,
 *
 * where op(A) = A or A**T, and A and B are both upper quasi-
 * triangular. A is M-by-M and B is N-by-N; the right hand side C and
 * the solution X are M-by-N; and scale is an output scale factor, set
 * <= 1 to avoid overflow in X.
 *
 * A and B must be in Schur canonical form (as returned by SHSEQR), that
 * is, block upper triangular with 1-by-1 and 2-by-2 diagonal blocks;
 * each 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.
 *
 * This is the block version of the algorithm.
 *
 * @param[in] trana  Specifies the option op(A):
 *                   = 'N': op(A) = A    (No transpose)
 *                   = 'T': op(A) = A**T (Transpose)
 *                   = 'C': op(A) = A**H (Conjugate transpose = Transpose)
 * @param[in] tranb  Specifies the option op(B):
 *                   = 'N': op(B) = B    (No transpose)
 *                   = 'T': op(B) = B**T (Transpose)
 *                   = 'C': op(B) = B**H (Conjugate transpose = Transpose)
 * @param[in] isgn   Specifies the sign in the equation:
 *                   = +1: solve op(A)*X + X*op(B) = scale*C
 *                   = -1: solve op(A)*X - X*op(B) = scale*C
 * @param[in] m      The order of the matrix A, and the number of rows in the
 *                   matrices X and C. m >= 0.
 * @param[in] n      The order of the matrix B, and the number of columns in the
 *                   matrices X and C. n >= 0.
 * @param[in] A      Double precision array, dimension (lda, m).
 *                   The upper quasi-triangular matrix A, in Schur canonical form.
 * @param[in] lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[in] B      Double precision array, dimension (ldb, n).
 *                   The upper quasi-triangular matrix B, in Schur canonical form.
 * @param[in] ldb    The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] C  Double precision array, dimension (ldc, n).
 *                   On entry, the M-by-N right hand side matrix C.
 *                   On exit, C is overwritten by the solution matrix X.
 * @param[in] ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out] scale The scale factor, scale, set <= 1 to avoid overflow in X.
 * @param[out] iwork Integer array, dimension (MAX(1, liwork)).
 *                   On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in] liwork The dimension of the array iwork.
 *                   liwork >= ((m + nb - 1) / nb + 1) + ((n + nb - 1) / nb + 1),
 *                   where nb is the optimal block size.
 *                   If liwork = -1, then a workspace query is assumed.
 * @param[out] swork Double precision array, dimension (MAX(2, rows), MAX(1, cols)).
 *                   On exit, if info = 0, swork[0] returns the optimal rows
 *                   and swork[1] returns the optimal cols.
 * @param[in] ldswork The leading dimension of the array swork.
 *                    ldswork >= MAX(2, rows), where rows = ((m + nb - 1) / nb + 1).
 *                    If ldswork = -1, then a workspace query is assumed.
 * @param[out] info  = 0: successful exit
 *                   < 0: if info = -i, the i-th argument had an illegal value
 *                   = 1: A and B have common or very close eigenvalues; perturbed
 *                        values were used to solve the equation (but the matrices
 *                        A and B are unchanged).
 */
void strsyl3(const char* trana, const char* tranb, const int isgn,
             const int m, const int n,
             const float* A, const int lda,
             const float* B, const int ldb,
             float* C, const int ldc,
             float* scale,
             int* iwork, const int liwork,
             float* swork, const int ldswork,
             int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int notrna, notrnb, lquery, skip;
    int awrk, bwrk, i, i1, i2, iinfo, j, j1, j2, jj;
    int k, k1, k2, l, l1, l2, ll, nba, nb, nbb, pc;
    float anrm, bignum, bnrm, cnrm, scal, scaloc;
    float scamin, sgn, xnrm, buf, smlnum;

    /* Decode and Test input parameters */
    notrna = (trana[0] == 'N' || trana[0] == 'n');
    notrnb = (tranb[0] == 'N' || tranb[0] == 'n');

    /* Use the same block size for all matrices.
     * From ilaenv.f lines 477-485: for STRSYL (real precision),
     * NB = MIN(MAX(48, INT(MIN(N1,N2)*16/100)), 240)
     * with a minimum of 8 from line 250.
     */
    {
        int min_mn_local = (m < n) ? m : n;
        int nb_calc = (min_mn_local * 16) / 100;
        if (nb_calc < 48) nb_calc = 48;
        if (nb_calc > 240) nb_calc = 240;
        nb = (nb_calc > 8) ? nb_calc : 8;
    }

    /* Compute number of blocks in A and B */
    nba = (m + nb - 1) / nb;
    if (nba < 1) nba = 1;
    nbb = (n + nb - 1) / nb;
    if (nbb < 1) nbb = 1;

    /* Compute workspace */
    *info = 0;
    lquery = (liwork == -1 || ldswork == -1);
    iwork[0] = nba + nbb + 2;
    if (lquery) {
        swork[0] = (nba > nbb) ? nba : nbb;
        swork[1] = 2 * nbb + nba;
    }

    /* Test the input arguments */
    if (!notrna && !(trana[0] == 'T' || trana[0] == 't') &&
        !(trana[0] == 'C' || trana[0] == 'c')) {
        *info = -1;
    } else if (!notrnb && !(tranb[0] == 'T' || tranb[0] == 't') &&
               !(tranb[0] == 'C' || tranb[0] == 'c')) {
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
        xerbla("STRSYL3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *scale = ONE;
    if (m == 0 || n == 0)
        return;

    /* Use unblocked code for small problems or if insufficient
     * workspaces are provided
     */
    {
        int min_nba_nbb = (nba < nbb) ? nba : nbb;
        int max_nba_nbb = (nba > nbb) ? nba : nbb;
        if (min_nba_nbb == 1 || ldswork < max_nba_nbb ||
            liwork < iwork[0]) {
            strsyl(trana, tranb, isgn, m, n, A, lda, B, ldb,
                   C, ldc, scale, info);
            return;
        }
    }

    /* Set constants to control overflow */
    smlnum = slamch("S");
    bignum = ONE / smlnum;

    /* Partition A such that 2-by-2 blocks on the diagonal are not split */
    skip = 0;
    for (i = 0; i < nba; i++) {
        iwork[i] = i * nb;
    }
    iwork[nba] = m;
    for (k = 0; k < nba; k++) {
        l1 = iwork[k];
        l2 = iwork[k + 1];
        for (l = l1; l < l2; l++) {
            if (skip) {
                skip = 0;
                continue;
            }
            if (l >= m - 1) {
                /* A(m-1, m-1) is a 1-by-1 block */
                continue;
            }
            if (A[l + (l + 1) * lda] != ZERO && A[(l + 1) + l * lda] != ZERO) {
                /* Check if 2-by-2 block is split */
                if (l + 1 == iwork[k + 1]) {
                    iwork[k + 1] = iwork[k + 1] + 1;
                    continue;
                }
                skip = 1;
            }
        }
    }
    iwork[nba] = m;
    if (iwork[nba - 1] >= iwork[nba]) {
        iwork[nba - 1] = iwork[nba];
        nba = nba - 1;
    }

    /* Partition B such that 2-by-2 blocks on the diagonal are not split */
    pc = nba + 1;
    skip = 0;
    for (i = 0; i < nbb; i++) {
        iwork[pc + i] = i * nb;
    }
    iwork[pc + nbb] = n;
    for (k = 0; k < nbb; k++) {
        l1 = iwork[pc + k];
        l2 = iwork[pc + k + 1];
        for (l = l1; l < l2; l++) {
            if (skip) {
                skip = 0;
                continue;
            }
            if (l >= n - 1) {
                /* B(n-1, n-1) is a 1-by-1 block */
                continue;
            }
            if (B[l + (l + 1) * ldb] != ZERO && B[(l + 1) + l * ldb] != ZERO) {
                /* Check if 2-by-2 block is split */
                if (l + 1 == iwork[pc + k + 1]) {
                    iwork[pc + k + 1] = iwork[pc + k + 1] + 1;
                    continue;
                }
                skip = 1;
            }
        }
    }
    iwork[pc + nbb] = n;
    if (iwork[pc + nbb - 1] >= iwork[pc + nbb]) {
        iwork[pc + nbb - 1] = iwork[pc + nbb];
        nbb = nbb - 1;
    }

    /* Set local scaling factors - must never attain zero. */
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
        k1 = iwork[k];
        k2 = iwork[k + 1];
        for (l = k; l < nba; l++) {
            l1 = iwork[l];
            l2 = iwork[l + 1];
            if (notrna) {
                swork[k + (awrk + l) * ldswork] = slange("I", k2 - k1, l2 - l1,
                                                          &A[k1 + l1 * lda], lda, NULL);
            } else {
                swork[l + (awrk + k) * ldswork] = slange("1", k2 - k1, l2 - l1,
                                                          &A[k1 + l1 * lda], lda, NULL);
            }
        }
    }
    bwrk = nbb + nba;
    for (k = 0; k < nbb; k++) {
        k1 = iwork[pc + k];
        k2 = iwork[pc + k + 1];
        for (l = k; l < nbb; l++) {
            l1 = iwork[pc + l];
            l2 = iwork[pc + l + 1];
            if (notrnb) {
                swork[k + (bwrk + l) * ldswork] = slange("I", k2 - k1, l2 - l1,
                                                          &B[k1 + l1 * ldb], ldb, NULL);
            } else {
                swork[l + (bwrk + k) * ldswork] = slange("1", k2 - k1, l2 - l1,
                                                          &B[k1 + l1 * ldb], ldb, NULL);
            }
        }
    }

    sgn = (float)isgn;

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

            /* K1: row index of the first row in X(K,L)
             * K2: row index of the first row in X(K+1,L)
             * so the K2 - K1 is the column count of the block X(K,L)
             */
            k1 = iwork[k];
            k2 = iwork[k + 1];
            for (l = 0; l < nbb; l++) {

                /* L1: column index of the first column in X(K,L)
                 * L2: column index of the first column in X(K,L+1)
                 * so that L2 - L1 is the row count of the block X(K,L)
                 */
                l1 = iwork[pc + l];
                l2 = iwork[pc + l + 1];

                strsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        /* The magnitude of the largest entry of X(K1:K2-1, L1:L2-1)
                         * is larger than the product of BIGNUM**2 and cannot be
                         * represented in the form (1/SCALE)*X(K1:K2-1, L1:L2-1).
                         * Mark the computation as pointless.
                         */
                        buf = ZERO;
                    } else {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            /* Bound by BIGNUM to not introduce Inf. The value
                             * is irrelevant; corresponding entries of the
                             * solution will be flushed in consistency scaling.
                             */
                            float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = slange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, NULL);

                for (i = k - 1; i >= 0; i--) {

                    /* C(I,L) := C(I,L) - A(I,K) * C(K,L) */

                    i1 = iwork[i];
                    i2 = iwork[i + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(I,L) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = l1; jj < l2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                i2 - i1, l2 - l1, k2 - k1, -ONE,
                                &A[i1 + k1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                ONE, &C[i1 + l1 * ldc], ldc);

                }

                for (j = l + 1; j < nbb; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(L,J) */

                    j1 = iwork[pc + j];
                    j2 = iwork[pc + j + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(K,J) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                k2 - k1, j2 - j1, l2 - l1, -sgn,
                                &C[k1 + l1 * ldc], ldc, &B[l1 + j1 * ldb], ldb,
                                ONE, &C[k1 + j1 * ldc], ldc);
                }
            }
        }
    } else if (!notrna && notrnb) {

        /* Solve    A**T*X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * upper-left corner column by column by
         *
         *   A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *            K-1                        L-1
         *   R(K,L) = SUM [A(I,K)**T*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]
         *            I=1                        J=1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = 0; k < nba; k++) {

            /* K1: row index of the first row in X(K,L)
             * K2: row index of the first row in X(K+1,L)
             * so the K2 - K1 is the column count of the block X(K,L)
             */
            k1 = iwork[k];
            k2 = iwork[k + 1];
            for (l = 0; l < nbb; l++) {

                /* L1: column index of the first column in X(K,L)
                 * L2: column index of the first column in X(K,L+1)
                 * so that L2 - L1 is the row count of the block X(K,L)
                 */
                l1 = iwork[pc + l];
                l2 = iwork[pc + l + 1];

                strsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        /* The magnitude of the largest entry of X(K1:K2-1, L1:L2-1)
                         * is larger than the product of BIGNUM**2 and cannot be
                         * represented in the form (1/SCALE)*X(K1:K2-1, L1:L2-1).
                         * Mark the computation as pointless.
                         */
                        buf = ZERO;
                    } else {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            /* Bound by BIGNUM to not introduce Inf. The value
                             * is irrelevant; corresponding entries of the
                             * solution will be flushed in consistency scaling.
                             */
                            float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = slange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, NULL);

                for (i = k + 1; i < nba; i++) {

                    /* C(I,L) := C(I,L) - A(K,I)**T * C(K,L) */

                    i1 = iwork[i];
                    i2 = iwork[i + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to to C(I,L) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                i2 - i1, l2 - l1, k2 - k1, -ONE,
                                &A[k1 + i1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                ONE, &C[i1 + l1 * ldc], ldc);

                }

                for (j = l + 1; j < nbb; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(L,J) */

                    j1 = iwork[pc + j];
                    j2 = iwork[pc + j + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to to C(K,J) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                k2 - k1, j2 - j1, l2 - l1, -sgn,
                                &C[k1 + l1 * ldc], ldc, &B[l1 + j1 * ldb], ldb,
                                ONE, &C[k1 + j1 * ldc], ldc);
                }
            }
        }
    } else if (!notrna && !notrnb) {

        /* Solve    A**T*X + ISGN*X*B**T = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * top-right corner column by column by
         *
         *    A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
         *
         * Where
         *              K-1                          N
         *     R(K,L) = SUM [A(I,K)**T*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
         *              I=1                        J=L+1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = 0; k < nba; k++) {

            /* K1: row index of the first row in X(K,L)
             * K2: row index of the first row in X(K+1,L)
             * so the K2 - K1 is the column count of the block X(K,L)
             */
            k1 = iwork[k];
            k2 = iwork[k + 1];
            for (l = nbb - 1; l >= 0; l--) {

                /* L1: column index of the first column in X(K,L)
                 * L2: column index of the first column in X(K,L+1)
                 * so that L2 - L1 is the row count of the block X(K,L)
                 */
                l1 = iwork[pc + l];
                l2 = iwork[pc + l + 1];

                strsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        /* The magnitude of the largest entry of X(K1:K2-1, L1:L2-1)
                         * is larger than the product of BIGNUM**2 and cannot be
                         * represented in the form (1/SCALE)*X(K1:K2-1, L1:L2-1).
                         * Mark the computation as pointless.
                         */
                        buf = ZERO;
                    } else {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            /* Bound by BIGNUM to not introduce Inf. The value
                             * is irrelevant; corresponding entries of the
                             * solution will be flushed in consistency scaling.
                             */
                            float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                xnrm = slange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, NULL);

                for (i = k + 1; i < nba; i++) {

                    /* C(I,L) := C(I,L) - A(K,I)**T * C(K,L) */

                    i1 = iwork[i];
                    i2 = iwork[i + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(I,L) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                i2 - i1, l2 - l1, k2 - k1, -ONE,
                                &A[k1 + i1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                ONE, &C[i1 + l1 * ldc], ldc);

                }

                for (j = 0; j < l; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(J,L)**T */

                    j1 = iwork[pc + j];
                    j2 = iwork[pc + j + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(K,J) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                k2 - k1, j2 - j1, l2 - l1, -sgn,
                                &C[k1 + l1 * ldc], ldc, &B[j1 + l1 * ldb], ldb,
                                ONE, &C[k1 + j1 * ldc], ldc);
                }
            }
        }
    } else if (notrna && !notrnb) {

        /* Solve    A*X + ISGN*X*B**T = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-right corner column by column by
         *
         *     A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
         *
         * Where
         *               M                          N
         *     R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
         *             I=K+1                      J=L+1
         *
         * Start loop over block rows (index = K) and block columns (index = L)
         */
        for (k = nba - 1; k >= 0; k--) {

            /* K1: row index of the first row in X(K,L)
             * K2: row index of the first row in X(K+1,L)
             * so the K2 - K1 is the column count of the block X(K,L)
             */
            k1 = iwork[k];
            k2 = iwork[k + 1];
            for (l = nbb - 1; l >= 0; l--) {

                /* L1: column index of the first column in X(K,L)
                 * L2: column index of the first column in X(K,L+1)
                 * so that L2 - L1 is the row count of the block X(K,L)
                 */
                l1 = iwork[pc + l];
                l2 = iwork[pc + l + 1];

                strsyl(trana, tranb, isgn, k2 - k1, l2 - l1,
                       &A[k1 + k1 * lda], lda,
                       &B[l1 + l1 * ldb], ldb,
                       &C[k1 + l1 * ldc], ldc, &scaloc, &iinfo);
                *info = (*info > iinfo) ? *info : iinfo;

                if (scaloc * swork[k + l * ldswork] == ZERO) {
                    if (scaloc == ZERO) {
                        /* The magnitude of the largest entry of X(K1:K2-1, L1:L2-1)
                         * is larger than the product of BIGNUM**2 and cannot be
                         * represented in the form (1/SCALE)*X(K1:K2-1, L1:L2-1).
                         * Mark the computation as pointless.
                         */
                        buf = ZERO;
                    } else {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                    }
                    for (jj = 0; jj < nbb; jj++) {
                        for (ll = 0; ll < nba; ll++) {
                            /* Bound by BIGNUM to not introduce Inf. The value
                             * is irrelevant; corresponding entries of the
                             * solution will be flushed in consistency scaling.
                             */
                            float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                            swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                        }
                    }
                }
                swork[k + l * ldswork] = scaloc * swork[k + l * ldswork];
                xnrm = slange("I", k2 - k1, l2 - l1, &C[k1 + l1 * ldc], ldc, NULL);

                for (i = 0; i < k; i++) {

                    /* C(I,L) := C(I,L) - A(I,K) * C(K,L) */

                    i1 = iwork[i];
                    i2 = iwork[i + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", i2 - i1, l2 - l1, &C[i1 + l1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[i + l * ldswork] < swork[k + l * ldswork])
                             ? swork[i + l * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[i + l * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    anrm = swork[i + (awrk + k) * ldswork];
                    scaloc = slarmm(anrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(I,L) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[i + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (ll = l1; ll < l2; ll++) {
                            cblas_sscal(i2 - i1, scal, &C[i1 + ll * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[i + l * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                i2 - i1, l2 - l1, k2 - k1, -ONE,
                                &A[i1 + k1 * lda], lda, &C[k1 + l1 * ldc], ldc,
                                ONE, &C[i1 + l1 * ldc], ldc);

                }

                for (j = 0; j < l; j++) {

                    /* C(K,J) := C(K,J) - SGN * C(K,L) * B(J,L)**T */

                    j1 = iwork[pc + j];
                    j2 = iwork[pc + j + 1];

                    /* Compute scaling factor to survive the linear update
                     * simulating consistent scaling.
                     */
                    cnrm = slange("I", k2 - k1, j2 - j1, &C[k1 + j1 * ldc],
                                  ldc, NULL);
                    scamin = (swork[k + j * ldswork] < swork[k + l * ldswork])
                             ? swork[k + j * ldswork] : swork[k + l * ldswork];
                    cnrm = cnrm * (scamin / swork[k + j * ldswork]);
                    xnrm = xnrm * (scamin / swork[k + l * ldswork]);
                    bnrm = swork[l + (bwrk + j) * ldswork];
                    scaloc = slarmm(bnrm, xnrm, cnrm);
                    if (scaloc * scamin == ZERO) {
                        /* Use second scaling factor to prevent flushing to zero. */
                        buf = buf * ldexpf(1.0f, ilogb(scaloc));
                        for (jj = 0; jj < nbb; jj++) {
                            for (ll = 0; ll < nba; ll++) {
                                float tmp = swork[ll + jj * ldswork] / ldexpf(1.0f, ilogb(scaloc));
                                swork[ll + jj * ldswork] = (tmp < bignum) ? tmp : bignum;
                            }
                        }
                        scamin = scamin / ldexpf(1.0f, ilogb(scaloc));
                        scaloc = scaloc / ldexpf(1.0f, ilogb(scaloc));
                    }
                    cnrm = cnrm * scaloc;
                    xnrm = xnrm * scaloc;

                    /* Simultaneously apply the robust update factor and the
                     * consistency scaling factor to C(K,J) and C(K,L).
                     */
                    scal = (scamin / swork[k + l * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = l1; jj < l2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    scal = (scamin / swork[k + j * ldswork]) * scaloc;
                    if (scal != ONE) {
                        for (jj = j1; jj < j2; jj++) {
                            cblas_sscal(k2 - k1, scal, &C[k1 + jj * ldc], 1);
                        }
                    }

                    /* Record current scaling factor */
                    swork[k + l * ldswork] = scamin * scaloc;
                    swork[k + j * ldswork] = scamin * scaloc;

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                k2 - k1, j2 - j1, l2 - l1, -sgn,
                                &C[k1 + l1 * ldc], ldc, &B[j1 + l1 * ldb], ldb,
                                ONE, &C[k1 + j1 * ldc], ldc);
                }
            }
        }

    }

    /* Reduce local scaling factors */
    *scale = swork[0];
    for (k = 0; k < nba; k++) {
        for (l = 0; l < nbb; l++) {
            float sw = swork[k + l * ldswork];
            if (sw < *scale) *scale = sw;
        }
    }

    if (*scale == ZERO) {

        /* The magnitude of the largest entry of the solution is larger
         * than the product of BIGNUM**2 and cannot be represented in the
         * form (1/SCALE)*X if SCALE is DOUBLE PRECISION. Set SCALE to
         * zero and give up.
         */
        iwork[0] = nba + nbb + 2;
        swork[0] = (nba > nbb) ? nba : nbb;
        swork[1] = 2 * nbb + nba;
        return;
    }

    /* Realize consistent scaling */
    for (k = 0; k < nba; k++) {
        k1 = iwork[k];
        k2 = iwork[k + 1];
        for (l = 0; l < nbb; l++) {
            l1 = iwork[pc + l];
            l2 = iwork[pc + l + 1];
            scal = *scale / swork[k + l * ldswork];
            if (scal != ONE) {
                for (ll = l1; ll < l2; ll++) {
                    cblas_sscal(k2 - k1, scal, &C[k1 + ll * ldc], 1);
                }
            }
        }
    }

    if (buf != ONE && buf > ZERO) {

        /* Decrease SCALE as much as possible. */
        scaloc = (*scale / smlnum < ONE / buf) ? *scale / smlnum : ONE / buf;
        buf = buf * scaloc;
        *scale = *scale / scaloc;
    }

    if (buf != ONE && buf > ZERO) {

        /* In case of overly aggressive scaling during the computation,
         * flushing of the global scale factor may be prevented by
         * undoing some of the scaling. This step is to ensure that
         * this routine flushes only scale factors that TRSYL also
         * flushes and be usable as a drop-in replacement.
         *
         * How much can the normwise largest entry be upscaled?
         */
        scal = C[0];
        for (k = 0; k < m; k++) {
            for (l = 0; l < n; l++) {
                float absval = fabsf(C[k + l * ldc]);
                if (absval > scal) scal = absval;
            }
        }

        /* Increase BUF as close to 1 as possible and apply scaling. */
        scaloc = (bignum / scal < ONE / buf) ? bignum / scal : ONE / buf;
        buf = buf * scaloc;
        slascl("G", -1, -1, ONE, scaloc, m, n, C, ldc, &iinfo);
    }

    /* Combine with buffer scaling factor. SCALE will be flushed if
     * BUF is less than one here.
     */
    *scale = *scale * buf;

    /* Restore workspace dimensions */
    iwork[0] = nba + nbb + 2;
    swork[0] = (nba > nbb) ? nba : nbb;
    swork[1] = 2 * nbb + nba;

    return;
}
