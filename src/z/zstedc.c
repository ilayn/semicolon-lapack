/**
 * @file zstedc.c
 * @brief ZSTEDC computes all eigenvalues and, optionally, eigenvectors of a
 *        symmetric tridiagonal matrix using the divide and conquer method.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSTEDC computes all eigenvalues and, optionally, eigenvectors of a
 * symmetric tridiagonal matrix using the divide and conquer method.
 * The eigenvectors of a full or band complex Hermitian matrix can also
 * be found if ZHETRD or ZHPTRD or ZHBTRD has been used to reduce this
 * matrix to tridiagonal form.
 *
 * @param[in]     compz  = 'N': Compute eigenvalues only.
 *                         = 'I': Compute eigenvectors of tridiagonal matrix also.
 *                         = 'V': Compute eigenvectors of original Hermitian matrix
 *                                also. On entry, Z contains the unitary matrix used
 *                                to reduce the original matrix to tridiagonal form.
 * @param[in]     n      The dimension of the symmetric tridiagonal matrix. n >= 0.
 * @param[in,out] D      Double precision array, dimension (n).
 *                        On entry, the diagonal elements of the tridiagonal matrix.
 *                        On exit, if info = 0, the eigenvalues in ascending order.
 * @param[in,out] E      Double precision array, dimension (n-1).
 *                        On entry, the subdiagonal elements of the tridiagonal matrix.
 *                        On exit, E has been destroyed.
 * @param[in,out] Z      Complex*16 array, dimension (ldz, n).
 *                        On entry, if compz = 'V', then Z contains the unitary
 *                        matrix used in the reduction to tridiagonal form.
 *                        On exit, if info = 0, then if compz = 'V', Z contains the
 *                        orthonormal eigenvectors of the original Hermitian matrix,
 *                        and if compz = 'I', Z contains the orthonormal eigenvectors
 *                        of the symmetric tridiagonal matrix.
 *                        If compz = 'N', then Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1.
 *                        If eigenvectors are desired, then ldz >= max(1,n).
 * @param[out]    work   Complex*16 array, dimension (max(1,lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                        If compz = 'N' or 'I', or n <= 1, lwork must be at least 1.
 *                        If compz = 'V' and n > 1, lwork must be at least n*n.
 *                        If lwork = -1, then a workspace query is assumed.
 * @param[out]    rwork  Double precision array, dimension (max(1,lrwork)).
 *                        On exit, if info = 0, rwork[0] returns the optimal lrwork.
 * @param[in]     lrwork The dimension of the array rwork.
 *                        If compz = 'N' or n <= 1, lrwork must be at least 1.
 *                        If compz = 'V' and n > 1, lrwork must be at least
 *                                   1 + 3*n + 2*n*lgn + 4*n^2,
 *                                   where lgn = smallest integer k such that 2^k >= n.
 *                        If compz = 'I' and n > 1, lrwork must be at least
 *                                   1 + 4*n + 2*n^2.
 *                        If lrwork = -1, then a workspace query is assumed.
 * @param[out]    iwork  Integer array, dimension (max(1,liwork)).
 *                        On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork The dimension of the array iwork.
 *                        If compz = 'N' or n <= 1, liwork must be at least 1.
 *                        If compz = 'V' and n > 1, liwork must be at least
 *                                   6 + 6*n + 5*n*lgn.
 *                        If compz = 'I' and n > 1, liwork must be at least
 *                                   3 + 5*n.
 *                        If liwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: The algorithm failed to compute an eigenvalue while
 *                           working on the submatrix lying in rows and columns
 *                           info/(n+1) through mod(info,n+1).
 */
void zstedc(const char* compz, const int n,
            double* D, double* E,
            double complex* Z, const int ldz,
            double complex* work, const int lwork,
            double* rwork, const int lrwork,
            int* iwork, const int liwork, int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;

    int lquery;
    int finish, i, icompz, ii, j, k, lgn, ll;
    int liwmin = 1, lrwmin = 1, lwmin = 1, m, start;
    double eps, orgnrm, p, tiny;

    /* SMLSIZ from ILAENV(9, 'ZSTEDC', ...) - hardcoded per project convention */
    const int SMLSIZ = 25;

    /*
     * Test the input parameters.
     */
    *info = 0;
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

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
            lwmin = 1;
            liwmin = 1;
            lrwmin = 1;
        } else if (n <= SMLSIZ) {
            lwmin = 1;
            liwmin = 1;
            lrwmin = 2 * (n - 1);
        } else if (icompz == 1) {
            lgn = (int)(log((double)n) / log(TWO));
            if ((1 << lgn) < n)
                lgn = lgn + 1;
            if ((1 << lgn) < n)
                lgn = lgn + 1;
            lwmin = n * n;
            lrwmin = 1 + 3*n + 2*n*lgn + 4*n*n;
            liwmin = 6 + 6*n + 5*n*lgn;
        } else if (icompz == 2) {
            lwmin = 1;
            lrwmin = 1 + 4*n + 2*n*n;
            liwmin = 3 + 5*n;
        }
        work[0] = CMPLX((double)lwmin, 0.0);
        rwork[0] = (double)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -10;
        } else if (liwork < liwmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZSTEDC", -(*info));
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
            Z[0] = CMPLX(ONE, 0.0);
        return;
    }

    /*
     * If the following conditional clause is removed, then the routine
     * will use the Divide and Conquer routine to compute only the
     * eigenvalues, which requires (3N + 3N**2) real workspace and
     * (2 + 5N + 2N lg(N)) integer workspace.
     * Since on many architectures DSTERF is much faster than any other
     * algorithm for finding eigenvalues only, it is used here
     * as the default. If the conditional clause is removed, then
     * information on the size of workspace needs to be changed.
     *
     * If COMPZ = 'N', use DSTERF to compute the eigenvalues.
     */
    if (icompz == 0) {
        dsterf(n, D, E, info);
        goto L70;
    }

    /*
     * If N is smaller than the minimum divide size (SMLSIZ+1), then
     * solve the problem with another solver.
     */
    if (n <= SMLSIZ) {

        zsteqr(compz, n, D, E, Z, ldz, rwork, info);

    } else {

        /*
         * If COMPZ = 'I', we simply call DSTEDC instead.
         */
        if (icompz == 2) {
            dlaset("Full", n, n, ZERO, ONE, rwork, n);
            ll = n * n;
            dstedc("I", n, D, E, rwork, n,
                   &rwork[ll], lrwork - ll, iwork, liwork, info);
            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    Z[i + j * ldz] = CMPLX(rwork[j * n + i], 0.0);
                }
            }
            goto L70;
        }

        /*
         * From now on, only option left to be handled is COMPZ = 'V',
         * i.e. ICOMPZ = 1.
         *
         * Scale.
         */
        orgnrm = dlanst("M", n, D, E);
        if (orgnrm == ZERO)
            goto L70;

        eps = dlamch("Epsilon");

        start = 0;  /* 0-based */

        /*
         * while ( start < n )
         */
    L30:
        if (start < n) {

            /*
             * Let FINISH be the position of the next subdiagonal entry
             * such that E( FINISH ) <= TINY or FINISH = N if no such
             * subdiagonal exists.  The matrix identified by the elements
             * between START and FINISH constitutes an independent
             * sub-problem.
             */
            finish = start;
        L40:
            if (finish < n - 1) {
                tiny = eps * sqrt(fabs(D[finish])) *
                           sqrt(fabs(D[finish + 1]));
                if (fabs(E[finish]) > tiny) {
                    finish = finish + 1;
                    goto L40;
                }
            }

            /*
             * (Sub) Problem determined.  Compute its size and solve it.
             */
            m = finish - start + 1;
            if (m > SMLSIZ) {

                /*
                 * Scale.
                 */
                orgnrm = dlanst("M", m, &D[start], &E[start]);
                dlascl("G", 0, 0, orgnrm, ONE, m, 1, &D[start],
                        m, info);
                dlascl("G", 0, 0, orgnrm, ONE, m - 1, 1,
                        &E[start], m - 1, info);

                zlaed0(n, m, &D[start], &E[start], &Z[start * ldz],
                        ldz, work, n, rwork, iwork, info);
                if (*info > 0) {
                    *info = (*info / (m + 1) + start) * (n + 1) +
                            (*info % (m + 1)) + start;
                    goto L70;
                }

                /*
                 * Scale back.
                 */
                dlascl("G", 0, 0, ONE, orgnrm, m, 1, &D[start],
                        m, info);

            } else {
                dsteqr("I", m, &D[start], &E[start], rwork, m,
                        &rwork[m * m], info);
                zlacrm(n, m, &Z[start * ldz], ldz, rwork, m, work,
                        n, &rwork[m * m]);
                zlacpy("A", n, m, work, n, &Z[start * ldz], ldz);
                if (*info > 0) {
                    *info = start * (n + 1) + finish;
                    goto L70;
                }
            }

            start = finish + 1;
            goto L30;
        }

        /*
         * endwhile
         *
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
                cblas_zswap(n, &Z[i * ldz], 1, &Z[k * ldz], 1);
            }
        }
    }

L70:
    work[0] = CMPLX((double)lwmin, 0.0);
    rwork[0] = (double)lrwmin;
    iwork[0] = liwmin;

    return;
}
