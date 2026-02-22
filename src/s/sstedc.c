/**
 * @file sstedc.c
 * @brief SSTEDC computes all eigenvalues and, optionally, eigenvectors of a
 *        symmetric tridiagonal matrix using the divide and conquer method.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSTEDC computes all eigenvalues and, optionally, eigenvectors of a
 * symmetric tridiagonal matrix using the divide and conquer method.
 * The eigenvectors of a full or band real symmetric matrix can also be
 * found if SSYTRD or SSPTRD or SSBTRD has been used to reduce this
 * matrix to tridiagonal form.
 *
 * @param[in]     compz = 'N': Compute eigenvalues only.
 *                        = 'I': Compute eigenvectors of tridiagonal matrix also.
 *                        = 'V': Compute eigenvectors of original dense symmetric
 *                               matrix also. On entry, Z contains the orthogonal
 *                               matrix used to reduce the original matrix to
 *                               tridiagonal form.
 * @param[in]     n     The dimension of the symmetric tridiagonal matrix. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the diagonal elements of the tridiagonal matrix.
 *                      On exit, if info = 0, the eigenvalues in ascending order.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the subdiagonal elements of the tridiagonal matrix.
 *                      On exit, E has been destroyed.
 * @param[in,out] Z     Double precision array, dimension (ldz, n).
 *                      On entry, if compz = 'V', then Z contains the orthogonal
 *                      matrix used in the reduction to tridiagonal form.
 *                      On exit, if info = 0, then if compz = 'V', Z contains the
 *                      orthonormal eigenvectors of the original symmetric matrix,
 *                      and if compz = 'I', Z contains the orthonormal eigenvectors
 *                      of the symmetric tridiagonal matrix.
 *                      If compz = 'N', then Z is not referenced.
 * @param[in]     ldz   The leading dimension of the array Z. ldz >= 1.
 *                      If eigenvectors are desired, then ldz >= max(1,n).
 * @param[out]    work  Double precision array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work.
 *                      If compz = 'N' or n <= 1 then lwork must be at least 1.
 *                      If compz = 'V' and n > 1 then lwork must be at least
 *                                 ( 1 + 3*n + 2*n*lgn + 4*n^2 ),
 *                                 where lgn = smallest integer k such that 2^k >= n.
 *                      If compz = 'I' and n > 1 then lwork must be at least
 *                                 ( 1 + 4*n + n^2 ).
 *                      Note that for compz = 'I' or 'V', then if n is less than or
 *                      equal to the minimum divide size, usually 25, then lwork need
 *                      only be max(1, 2*(n-1)).
 *                      If lwork = -1, then a workspace query is assumed.
 * @param[out]    iwork Integer array, dimension (max(1,liwork)).
 *                      On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork The dimension of the array iwork.
 *                      If compz = 'N' or n <= 1 then liwork must be at least 1.
 *                      If compz = 'V' and n > 1 then liwork must be at least
 *                                 ( 6 + 6*n + 5*n*lgn ).
 *                      If compz = 'I' and n > 1 then liwork must be at least
 *                                 ( 3 + 5*n ).
 *                      Note that for compz = 'I' or 'V', then if n is less than or
 *                      equal to the minimum divide size, usually 25, then liwork
 *                      need only be 1.
 *                      If liwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: The algorithm failed to compute an eigenvalue while
 *                           working on the submatrix lying in rows and columns
 *                           info/(n+1) through mod(info,n+1).
 */
void sstedc(const char* compz, const INT n,
            f32* D, f32* E,
            f32* Z, const INT ldz,
            f32* work, const INT lwork,
            INT* iwork, const INT liwork, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    INT lquery;
    INT i, icompz, ii, j, k, lgn, liwmin = 1, lwmin = 1, m, start, finish, storez, strtrw;
    f32 eps, orgnrm, p, tiny;

    /* SMLSIZ from ILAENV(9, 'SSTEDC', ...) - hardcoded per project convention */
    const INT SMLSIZ = 25;

    /*
     * Test the input parameters.
     */
    *info = 0;
    lquery = (lwork == -1 || liwork == -1);

    if (compz[0] == 'N' || compz[0] == 'n') {
        icompz = 0;
    } else if (compz[0] == 'V' || compz[0] == 'v') {
        icompz = 1;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        icompz = 2;
    } else {
        icompz = -1;
    }
    if (icompz < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if ((ldz < 1) ||
               (icompz > 0 && ldz < (1 > n ? 1 : n))) {
        *info = -6;
    }

    if (*info == 0) {

        /*
         * Compute the workspace requirements
         */
        if (n <= 1 || icompz == 0) {
            liwmin = 1;
            lwmin = 1;
        } else if (n <= SMLSIZ) {
            liwmin = 1;
            lwmin = 2 * (n - 1);
        } else {
            lgn = (INT)(logf((f32)n) / logf(TWO));
            if ((1 << lgn) < n)
                lgn = lgn + 1;
            if ((1 << lgn) < n)
                lgn = lgn + 1;
            if (icompz == 1) {
                lwmin = 1 + 3*n + 2*n*lgn + 4*n*n;
                liwmin = 6 + 6*n + 5*n*lgn;
            } else if (icompz == 2) {
                lwmin = 1 + 4*n + n*n;
                liwmin = 3 + 5*n;
            }
        }
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (liwork < liwmin && !lquery) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("SSTEDC", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /*
     * Quick return if possible
     */
    if (n == 0)
        return;
    if (n == 1) {
        if (icompz != 0)
            Z[0] = ONE;
        return;
    }

    /*
     * If the following conditional clause is removed, then the routine
     * will use the Divide and Conquer routine to compute only the
     * eigenvalues, which requires (3N + 3N**2) real workspace and
     * (2 + 5N + 2N lg(N)) integer workspace.
     * Since on many architectures SSTERF is much faster than any other
     * algorithm for finding eigenvalues only, it is used here
     * as the default. If the conditional clause is removed, then
     * information on the size of workspace needs to be changed.
     *
     * If COMPZ = 'N', use SSTERF to compute the eigenvalues.
     */
    if (icompz == 0) {
        ssterf(n, D, E, info);
        goto L50;
    }

    /*
     * If N is smaller than the minimum divide size (SMLSIZ+1), then
     * solve the problem with another solver.
     */
    if (n <= SMLSIZ) {

        ssteqr(compz, n, D, E, Z, ldz, work, info);

    } else {

        /*
         * If COMPZ = 'V', the Z matrix must be stored elsewhere for later
         * use.
         */
        if (icompz == 1) {
            storez = n * n;  /* 0-based offset into work */
        } else {
            storez = 0;
        }

        if (icompz == 2) {
            slaset("Full", n, n, ZERO, ONE, Z, ldz);
        }

        /*
         * Scale.
         */
        orgnrm = slanst("M", n, D, E);
        if (orgnrm == ZERO)
            goto L50;

        eps = slamch("Epsilon");

        start = 0;  /* 0-based */

        /*
         * while ( start < n )
         */
    L10:
        if (start < n) {

            /*
             * Let finish be the position of the next subdiagonal entry
             * such that E(finish) <= tiny or finish = n-1 if no such
             * subdiagonal exists.  The matrix identified by the elements
             * between start and finish constitutes an independent
             * sub-problem.
             */
            finish = start;
        L20:
            if (finish < n - 1) {
                tiny = eps * sqrtf(fabsf(D[finish])) *
                           sqrtf(fabsf(D[finish + 1]));
                if (fabsf(E[finish]) > tiny) {
                    finish = finish + 1;
                    goto L20;
                }
            }

            /*
             * (Sub) Problem determined.  Compute its size and solve it.
             */
            m = finish - start + 1;
            if (m == 1) {
                start = finish + 1;
                goto L10;
            }
            if (m > SMLSIZ) {

                /*
                 * Scale.
                 */
                orgnrm = slanst("M", m, &D[start], &E[start]);
                slascl("G", 0, 0, orgnrm, ONE, m, 1, &D[start],
                        m, info);
                slascl("G", 0, 0, orgnrm, ONE, m - 1, 1,
                        &E[start], m - 1, info);

                if (icompz == 1) {
                    strtrw = 0;  /* 0-based: row 0 of Z */
                } else {
                    strtrw = start;  /* 0-based */
                }
                slaed0(icompz, n, m, &D[start], &E[start],
                        &Z[strtrw + start * ldz], ldz, &work[0], n,
                        &work[storez], iwork, info);
                if (*info != 0) {
                    *info = (*info / (m + 1) + start) * (n + 1) +
                            (*info % (m + 1)) + start;
                    goto L50;
                }

                /*
                 * Scale back.
                 */
                slascl("G", 0, 0, ONE, orgnrm, m, 1, &D[start],
                        m, info);

            } else {
                if (icompz == 1) {

                    /*
                     * Since QR won't update a Z matrix which is larger than
                     * the length of D, we must solve the sub-problem in a
                     * workspace and then multiply back into Z.
                     */
                    ssteqr("I", m, &D[start], &E[start], work,
                            m, &work[m * m], info);
                    slacpy("A", n, m, &Z[start * ldz], ldz,
                            &work[storez], n);
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n, m, m, ONE,
                                &work[storez], n, work, m, ZERO,
                                &Z[start * ldz], ldz);
                } else if (icompz == 2) {
                    ssteqr("I", m, &D[start], &E[start],
                            &Z[start + start * ldz], ldz, work, info);
                } else {
                    ssterf(m, &D[start], &E[start], info);
                }
                if (*info != 0) {
                    *info = start * (n + 1) + finish;
                    goto L50;
                }
            }

            start = finish + 1;
            goto L10;
        }

        /*
         * endwhile
         */
        if (icompz == 0) {

            /*
             * Use Quick Sort
             */
            slasrt("I", n, D, info);

        } else {

            /*
             * Use Selection Sort to minimize swaps of eigenvectors
             */
            for (ii = 1; ii < n; ii++) {
                i = ii - 1;
                k = i;
                p = D[i];
                for (j = ii; j < n; j++) {
                    if (D[j] < p) {
                        k = j;
                        p = D[j];
                    }
                }
                if (k != i) {
                    D[k] = D[i];
                    D[i] = p;
                    cblas_sswap(n, &Z[i * ldz], 1, &Z[k * ldz], 1);
                }
            }
        }
    }

L50:
    work[0] = (f32)lwmin;
    iwork[0] = liwmin;

    return;
}
