/**
 * @file stgsyl.c
 * @brief STGSYL solves the generalized Sylvester equation.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * STGSYL solves the generalized Sylvester equation:
 *
 *             A * R - L * B = scale * C                 (1)
 *             D * R - L * E = scale * F
 *
 * where R and L are unknown m-by-n matrices, (A, D), (B, E) and
 * (C, F) are given matrix pairs of size m-by-m, n-by-n and m-by-n,
 * respectively, with real entries. (A, D) and (B, E) must be in
 * generalized (real) Schur canonical form, i.e. A, B are upper quasi
 * triangular and D, E are upper triangular.
 *
 * The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output
 * scaling factor chosen to avoid overflow.
 *
 * In matrix notation (1) is equivalent to solve  Zx = scale b, where
 * Z is defined as
 *
 *            Z = [ kron(In, A)  -kron(B**T, Im) ]         (2)
 *                [ kron(In, D)  -kron(E**T, Im) ].
 *
 * Here Ik is the identity matrix of size k and X**T is the transpose of
 * X. kron(X, Y) is the Kronecker product between the matrices X and Y.
 *
 * If TRANS = 'T', STGSYL solves the transposed system Z**T*y = scale*b,
 * which is equivalent to solve for R and L in
 *
 *             A**T * R + D**T * L = scale * C           (3)
 *             R * B**T + L * E**T = scale * -F
 *
 * This case (TRANS = 'T') is used to compute an one-norm-based estimate
 * of Dif[(A,D), (B,E)], the separation between the matrix pairs (A,D)
 * and (B,E), using SLACON.
 *
 * If IJOB >= 1, STGSYL computes a Frobenius norm-based estimate
 * of Dif[(A,D),(B,E)]. That is, the reciprocal of a lower bound on the
 * reciprocal of the smallest singular value of Z. See [1-2] for more
 * information.
 *
 * This is a level 3 BLAS algorithm.
 *
 * @param[in]     trans   'N': solve the generalized Sylvester equation (1).
 *                        'T': solve the 'transposed' system (3).
 * @param[in]     ijob    Specifies what kind of functionality to be performed.
 *                        = 0: solve (1) only.
 *                        = 1: The functionality of 0 and 3.
 *                        = 2: The functionality of 0 and 4.
 *                        = 3: Only an estimate of Dif[(A,D), (B,E)] is computed.
 *                             (look ahead strategy IJOB  = 1 is used).
 *                        = 4: Only an estimate of Dif[(A,D), (B,E)] is computed.
 *                             ( SGECON on sub-systems is used ).
 *                        Not referenced if TRANS = 'T'.
 * @param[in]     m       The order of the matrices A and D, and the row dimension of
 *                        the matrices C, F, R and L.
 * @param[in]     n       The order of the matrices B and E, and the column dimension
 *                        of the matrices C, F, R and L.
 * @param[in]     A       The upper quasi triangular matrix A.
 *                        Array of dimension (lda, m).
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1, m).
 * @param[in]     B       The upper quasi triangular matrix B.
 *                        Array of dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] C       On entry, C contains the right-hand-side of the first matrix
 *                        equation in (1) or (3).
 *                        On exit, if IJOB = 0, 1 or 2, C has been overwritten by
 *                        the solution R. If IJOB = 3 or 4 and TRANS = 'N', C holds R,
 *                        the solution achieved during the computation of the
 *                        Dif-estimate.
 *                        Array of dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of the array C. ldc >= max(1, m).
 * @param[in]     D       The upper triangular matrix D.
 *                        Array of dimension (ldd, m).
 * @param[in]     ldd     The leading dimension of the array D. ldd >= max(1, m).
 * @param[in]     E       The upper triangular matrix E.
 *                        Array of dimension (lde, n).
 * @param[in]     lde     The leading dimension of the array E. lde >= max(1, n).
 * @param[in,out] F       On entry, F contains the right-hand-side of the second matrix
 *                        equation in (1) or (3).
 *                        On exit, if IJOB = 0, 1 or 2, F has been overwritten by
 *                        the solution L. If IJOB = 3 or 4 and TRANS = 'N', F holds L,
 *                        the solution achieved during the computation of the
 *                        Dif-estimate.
 *                        Array of dimension (ldf, n).
 * @param[in]     ldf     The leading dimension of the array F. ldf >= max(1, m).
 * @param[out]    scale   On exit SCALE is the scaling factor in (1) or (3).
 *                        If 0 < SCALE < 1, C and F hold the solutions R and L, resp.,
 *                        to a slightly perturbed system but the input matrices A, B, D
 *                        and E have not been changed. If SCALE = 0, C and F hold the
 *                        solutions R and L, respectively, to the homogeneous system
 *                        with C = F = 0. Normally, SCALE = 1.
 * @param[out]    dif     On exit DIF is the reciprocal of a lower bound of the
 *                        reciprocal of the Dif-function, i.e. DIF is an upper bound of
 *                        Dif[(A,D), (B,E)] = sigma_min(Z), where Z as in (2).
 *                        IF IJOB = 0 or TRANS = 'T', DIF is not touched.
 * @param[out]    work    Array of dimension (max(1, lwork)).
 *                        On exit, if INFO = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of the array work. lwork >= 1.
 *                        If IJOB = 1 or 2 and TRANS = 'N', lwork >= max(1,2*m*n).
 *                        If lwork = -1, then a workspace query is assumed; the routine
 *                        only calculates the optimal size of the work array, returns
 *                        this value as the first entry of the work array, and no error
 *                        message related to lwork is issued by XERBLA.
 * @param[out]    iwork   Integer array of dimension (m+n+6).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: If info = -i, the i-th argument had an illegal value.
 *                         - > 0: (A, D) and (B, E) have common or close eigenvalues.
 */
void stgsyl(
    const char* trans,
    const INT ijob,
    const INT m,
    const INT n,
    const f32* restrict A,
    const INT lda,
    const f32* restrict B,
    const INT ldb,
    f32* restrict C,
    const INT ldc,
    const f32* restrict D,
    const INT ldd,
    const f32* restrict E,
    const INT lde,
    f32* restrict F,
    const INT ldf,
    f32* scale,
    f32* dif,
    f32* restrict work,
    const INT lwork,
    INT* restrict iwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lquery, notran;
    INT i, ie, ifunc, iround, is, isolve, j, je, js, k;
    INT linfo, lwmin, mb, nb, p, ppqq, pq, q;
    f32 dscale, dsum, scale2, scaloc;

    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');
    lquery = (lwork == -1);

    if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -1;
    } else if (notran) {
        if ((ijob < 0) || (ijob > 4)) {
            *info = -2;
        }
    }
    if (*info == 0) {
        if (m <= 0) {
            *info = -3;
        } else if (n <= 0) {
            *info = -4;
        } else if (lda < (1 > m ? 1 : m)) {
            *info = -6;
        } else if (ldb < (1 > n ? 1 : n)) {
            *info = -8;
        } else if (ldc < (1 > m ? 1 : m)) {
            *info = -10;
        } else if (ldd < (1 > m ? 1 : m)) {
            *info = -12;
        } else if (lde < (1 > n ? 1 : n)) {
            *info = -14;
        } else if (ldf < (1 > m ? 1 : m)) {
            *info = -16;
        }
    }

    if (*info == 0) {
        if (notran) {
            if (ijob == 1 || ijob == 2) {
                lwmin = (1 > 2 * m * n) ? 1 : 2 * m * n;
            } else {
                lwmin = 1;
            }
        } else {
            lwmin = 1;
        }
        work[0] = (f32)lwmin;

        if (lwork < lwmin && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        xerbla("STGSYL", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (m == 0 || n == 0) {
        *scale = 1;
        if (notran) {
            if (ijob != 0) {
                *dif = 0;
            }
        }
        return;
    }

    mb = 2;
    nb = 2;

    isolve = 1;
    ifunc = 0;
    if (notran) {
        if (ijob >= 3) {
            ifunc = ijob - 2;
            slaset("F", m, n, ZERO, ZERO, C, ldc);
            slaset("F", m, n, ZERO, ZERO, F, ldf);
        } else if (ijob >= 1) {
            isolve = 2;
        }
    }

    if ((mb <= 1 && nb <= 1) || (mb >= m && nb >= n)) {

        for (iround = 1; iround <= isolve; iround++) {

            dscale = ZERO;
            dsum = ONE;
            pq = 0;
            stgsy2(trans, ifunc, m, n, A, lda, B, ldb, C, ldc, D,
                   ldd, E, lde, F, ldf, scale, &dsum, &dscale,
                   iwork, &pq, info);
            if (dscale != ZERO) {
                if (ijob == 1 || ijob == 3) {
                    *dif = sqrtf((f32)(2 * m * n)) / (dscale * sqrtf(dsum));
                } else {
                    *dif = sqrtf((f32)pq) / (dscale * sqrtf(dsum));
                }
            }

            if (isolve == 2 && iround == 1) {
                if (notran) {
                    ifunc = ijob;
                }
                scale2 = *scale;
                slacpy("F", m, n, C, ldc, work, m);
                slacpy("F", m, n, F, ldf, &work[m * n], m);
                slaset("F", m, n, ZERO, ZERO, C, ldc);
                slaset("F", m, n, ZERO, ZERO, F, ldf);
            } else if (isolve == 2 && iround == 2) {
                slacpy("F", m, n, work, m, C, ldc);
                slacpy("F", m, n, &work[m * n], m, F, ldf);
                *scale = scale2;
            }
        }

        return;
    }

    p = 0;
    i = 1;
    while (i <= m) {
        p = p + 1;
        iwork[p - 1] = i;
        i = i + mb;
        if (i >= m) {
            break;
        }
        if (A[i - 1 + (i - 2) * lda] != ZERO) {
            i = i + 1;
        }
    }

    iwork[p] = m + 1;
    if (iwork[p - 1] == iwork[p]) {
        p = p - 1;
    }

    q = p + 1;
    j = 1;
    while (j <= n) {
        q = q + 1;
        iwork[q - 1] = j;
        j = j + nb;
        if (j >= n) {
            break;
        }
        if (B[j - 1 + (j - 2) * ldb] != ZERO) {
            j = j + 1;
        }
    }

    iwork[q] = n + 1;
    if (iwork[q - 1] == iwork[q]) {
        q = q - 1;
    }

    if (notran) {

        for (iround = 1; iround <= isolve; iround++) {

            dscale = ZERO;
            dsum = ONE;
            pq = 0;
            *scale = ONE;
            for (j = p + 2; j <= q; j++) {
                js = iwork[j - 1];
                je = iwork[j] - 1;
                nb = je - js + 1;
                for (i = p; i >= 1; i--) {
                    is = iwork[i - 1];
                    ie = iwork[i] - 1;
                    mb = ie - is + 1;
                    ppqq = 0;
                    stgsy2(trans, ifunc, mb, nb, &A[(is - 1) + (is - 1) * lda],
                           lda,
                           &B[(js - 1) + (js - 1) * ldb], ldb, &C[(is - 1) + (js - 1) * ldc], ldc,
                           &D[(is - 1) + (is - 1) * ldd], ldd, &E[(js - 1) + (js - 1) * lde], lde,
                           &F[(is - 1) + (js - 1) * ldf], ldf, &scaloc, &dsum, &dscale,
                           &iwork[q + 1], &ppqq, &linfo);
                    if (linfo > 0) {
                        *info = linfo;
                    }

                    pq = pq + ppqq;
                    if (scaloc != ONE) {
                        for (k = 1; k <= js - 1; k++) {
                            cblas_sscal(m, scaloc, &C[0 + (k - 1) * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + (k - 1) * ldf], 1);
                        }
                        for (k = js; k <= je; k++) {
                            cblas_sscal(is - 1, scaloc, &C[0 + (k - 1) * ldc], 1);
                            cblas_sscal(is - 1, scaloc, &F[0 + (k - 1) * ldf], 1);
                        }
                        for (k = js; k <= je; k++) {
                            cblas_sscal(m - ie, scaloc, &C[ie + (k - 1) * ldc], 1);
                            cblas_sscal(m - ie, scaloc, &F[ie + (k - 1) * ldf], 1);
                        }
                        for (k = je + 1; k <= n; k++) {
                            cblas_sscal(m, scaloc, &C[0 + (k - 1) * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + (k - 1) * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }

                    if (i > 1) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    is - 1, nb, mb, -ONE,
                                    &A[0 + (is - 1) * lda], lda, &C[(is - 1) + (js - 1) * ldc], ldc,
                                    ONE, &C[0 + (js - 1) * ldc], ldc);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    is - 1, nb, mb, -ONE,
                                    &D[0 + (is - 1) * ldd], ldd, &C[(is - 1) + (js - 1) * ldc], ldc,
                                    ONE, &F[0 + (js - 1) * ldf], ldf);
                    }
                    if (j < q) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    mb, n - je, nb, ONE,
                                    &F[(is - 1) + (js - 1) * ldf], ldf, &B[(js - 1) + je * ldb], ldb,
                                    ONE, &C[(is - 1) + je * ldc], ldc);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    mb, n - je, nb, ONE,
                                    &F[(is - 1) + (js - 1) * ldf], ldf, &E[(js - 1) + je * lde], lde,
                                    ONE, &F[(is - 1) + je * ldf], ldf);
                    }
                }
            }
            if (dscale != ZERO) {
                if (ijob == 1 || ijob == 3) {
                    *dif = sqrtf((f32)(2 * m * n)) / (dscale * sqrtf(dsum));
                } else {
                    *dif = sqrtf((f32)pq) / (dscale * sqrtf(dsum));
                }
            }
            if (isolve == 2 && iround == 1) {
                if (notran) {
                    ifunc = ijob;
                }
                scale2 = *scale;
                slacpy("F", m, n, C, ldc, work, m);
                slacpy("F", m, n, F, ldf, &work[m * n], m);
                slaset("F", m, n, ZERO, ZERO, C, ldc);
                slaset("F", m, n, ZERO, ZERO, F, ldf);
            } else if (isolve == 2 && iround == 2) {
                slacpy("F", m, n, work, m, C, ldc);
                slacpy("F", m, n, &work[m * n], m, F, ldf);
                *scale = scale2;
            }
        }

    } else {

        *scale = ONE;
        for (i = 1; i <= p; i++) {
            is = iwork[i - 1];
            ie = iwork[i] - 1;
            mb = ie - is + 1;
            for (j = q; j >= p + 2; j--) {
                js = iwork[j - 1];
                je = iwork[j] - 1;
                nb = je - js + 1;
                stgsy2(trans, ifunc, mb, nb, &A[(is - 1) + (is - 1) * lda], lda,
                       &B[(js - 1) + (js - 1) * ldb], ldb, &C[(is - 1) + (js - 1) * ldc], ldc,
                       &D[(is - 1) + (is - 1) * ldd], ldd, &E[(js - 1) + (js - 1) * lde], lde,
                       &F[(is - 1) + (js - 1) * ldf], ldf, &scaloc, &dsum, &dscale,
                       &iwork[q + 1], &ppqq, &linfo);
                if (linfo > 0) {
                    *info = linfo;
                }
                if (scaloc != ONE) {
                    for (k = 1; k <= js - 1; k++) {
                        cblas_sscal(m, scaloc, &C[0 + (k - 1) * ldc], 1);
                        cblas_sscal(m, scaloc, &F[0 + (k - 1) * ldf], 1);
                    }
                    for (k = js; k <= je; k++) {
                        cblas_sscal(is - 1, scaloc, &C[0 + (k - 1) * ldc], 1);
                        cblas_sscal(is - 1, scaloc, &F[0 + (k - 1) * ldf], 1);
                    }
                    for (k = js; k <= je; k++) {
                        cblas_sscal(m - ie, scaloc, &C[ie + (k - 1) * ldc], 1);
                        cblas_sscal(m - ie, scaloc, &F[ie + (k - 1) * ldf], 1);
                    }
                    for (k = je + 1; k <= n; k++) {
                        cblas_sscal(m, scaloc, &C[0 + (k - 1) * ldc], 1);
                        cblas_sscal(m, scaloc, &F[0 + (k - 1) * ldf], 1);
                    }
                    *scale = (*scale) * scaloc;
                }

                if (j > p + 2) {
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                mb, js - 1, nb, ONE, &C[(is - 1) + (js - 1) * ldc],
                                ldc, &B[0 + (js - 1) * ldb], ldb, ONE, &F[(is - 1) + 0 * ldf],
                                ldf);
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                mb, js - 1, nb, ONE, &F[(is - 1) + (js - 1) * ldf],
                                ldf, &E[0 + (js - 1) * lde], lde, ONE, &F[(is - 1) + 0 * ldf],
                                ldf);
                }
                if (i < p) {
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                m - ie, nb, mb, -ONE,
                                &A[(is - 1) + ie * lda], lda, &C[(is - 1) + (js - 1) * ldc], ldc,
                                ONE, &C[ie + (js - 1) * ldc], ldc);
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                m - ie, nb, mb, -ONE,
                                &D[(is - 1) + ie * ldd], ldd, &F[(is - 1) + (js - 1) * ldf], ldf,
                                ONE, &C[ie + (js - 1) * ldc], ldc);
                }
            }
        }

    }

    work[0] = (f32)lwmin;
}
