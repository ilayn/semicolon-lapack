/**
 * @file stgsy2.c
 * @brief STGSY2 solves the generalized Sylvester equation (unblocked algorithm).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

#define LDZ 8

/**
 * STGSY2 solves the generalized Sylvester equation:
 *
 *             A * R - L * B = scale * C                (1)
 *             D * R - L * E = scale * F,
 *
 * using Level 1 and 2 BLAS. where R and L are unknown M-by-N matrices,
 * (A, D), (B, E) and (C, F) are given matrix pairs of size M-by-M,
 * N-by-N and M-by-N, respectively, with real entries. (A, D) and (B, E)
 * must be in generalized Schur canonical form, i.e. A, B are upper
 * quasi triangular and D, E are upper triangular. The solution (R, L)
 * overwrites (C, F). 0 <= SCALE <= 1 is an output scaling factor
 * chosen to avoid overflow.
 *
 * @param[in]     trans   'N': solve the generalized Sylvester equation (1).
 *                        'T': solve the 'transposed' system.
 * @param[in]     ijob    Specifies what kind of functionality to be performed.
 *                        = 0: solve (1) only.
 *                        = 1: A contribution from this subsystem to a Frobenius
 *                             norm-based estimate (look ahead strategy).
 *                        = 2: A contribution using SGECON on sub-systems.
 * @param[in]     m       The order of A and D, and the row dimension of C, F, R and L.
 * @param[in]     n       The order of B and E, and the column dimension of C, F, R and L.
 * @param[in]     A       Array of dimension (lda, m). Upper quasi triangular matrix.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, m).
 * @param[in]     B       Array of dimension (ldb, n). Upper quasi triangular matrix.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] C       Array of dimension (ldc, n). On entry, the right-hand-side.
 *                        On exit, if ijob = 0, overwritten by the solution R.
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, m).
 * @param[in]     D       Array of dimension (ldd, m). Upper triangular matrix.
 * @param[in]     ldd     The leading dimension of D. ldd >= max(1, m).
 * @param[in]     E       Array of dimension (lde, n). Upper triangular matrix.
 * @param[in]     lde     The leading dimension of E. lde >= max(1, n).
 * @param[in,out] F       Array of dimension (ldf, n). On entry, the right-hand-side.
 *                        On exit, if ijob = 0, overwritten by the solution L.
 * @param[in]     ldf     The leading dimension of F. ldf >= max(1, m).
 * @param[out]    scale   On exit, 0 <= scale <= 1. Output scaling factor.
 * @param[in,out] rdsum   On entry/exit, sum of squares for Dif-estimate.
 * @param[in,out] rdscal  On entry/exit, scaling factor for rdsum.
 * @param[out]    iwork   Integer array of dimension (m+n+2).
 * @param[out]    pq      Number of subsystems solved.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: The matrix pairs (A, D) and (B, E) have common or very
 *                           close eigenvalues.
 */
void stgsy2(
    const char* trans,
    const int ijob,
    const int m,
    const int n,
    const float* const restrict A,
    const int lda,
    const float* const restrict B,
    const int ldb,
    float* const restrict C,
    const int ldc,
    const float* const restrict D,
    const int ldd,
    const float* const restrict E,
    const int lde,
    float* const restrict F,
    const int ldf,
    float* scale,
    float* rdsum,
    float* rdscal,
    int* const restrict iwork,
    int* pq,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int notran;
    int i, ie, ierr, ii, is, isp1, j, je, jj, js, jsp1;
    int k, mb, nb, p, q, zdim;
    float alpha, scaloc;

    int ipiv[LDZ], jpiv[LDZ];
    float rhs[LDZ], z[LDZ * LDZ];

    *info = 0;
    ierr = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -1;
    } else if (notran) {
        if ((ijob < 0) || (ijob > 2)) {
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
    if (*info != 0) {
        xerbla("STGSY2", -(*info));
        return;
    }

    /* Determine block structure of A */
    *pq = 0;
    p = 0;
    i = 0;
    while (i < m) {
        p = p + 1;
        iwork[p - 1] = i;
        if (i == m - 1) {
            break;
        }
        if (A[(i + 1) + i * lda] != ZERO) {
            i = i + 2;
        } else {
            i = i + 1;
        }
    }
    iwork[p] = m;

    /* Determine block structure of B */
    q = p + 1;
    j = 0;
    while (j < n) {
        q = q + 1;
        iwork[q - 1] = j;
        if (j == n - 1) {
            break;
        }
        if (B[(j + 1) + j * ldb] != ZERO) {
            j = j + 2;
        } else {
            j = j + 1;
        }
    }
    iwork[q] = n;
    *pq = p * (q - p - 1);

    if (notran) {

        /* Solve (I, J) - subsystem
           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
           for I = P, P - 1, ..., 1; J = 1, 2, ..., Q */

        *scale = ONE;
        scaloc = ONE;

        for (j = p + 1; j < q; j++) {
            js = iwork[j];
            jsp1 = js + 1;
            je = iwork[j + 1] - 1;
            nb = je - js + 1;

            for (i = p - 1; i >= 0; i--) {

                is = iwork[i];
                isp1 = is + 1;
                ie = iwork[i + 1] - 1;
                mb = ie - is + 1;
                zdim = mb * nb * 2;

                if ((mb == 1) && (nb == 1)) {

                    /* Build a 2-by-2 system Z * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = D[is + is * ldd];
                    z[0 + 1 * LDZ] = -B[js + js * ldb];
                    z[1 + 1 * LDZ] = -E[js + js * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = F[is + js * ldf];

                    /* Solve Z * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }

                    if (ijob == 0) {
                        sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                        if (scaloc != ONE) {
                            for (k = 0; k < n; k++) {
                                cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                                cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                            }
                            *scale = (*scale) * scaloc;
                        }
                    } else {
                        slatdf(ijob, zdim, z, LDZ, rhs, rdsum, rdscal, ipiv, jpiv);
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    F[is + js * ldf] = rhs[1];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (i > 0) {
                        alpha = -rhs[0];
                        cblas_saxpy(is, alpha, &A[0 + is * lda], 1, &C[0 + js * ldc], 1);
                        cblas_saxpy(is, alpha, &D[0 + is * ldd], 1, &F[0 + js * ldf], 1);
                    }
                    if (j < q - 1) {
                        cblas_saxpy(n - je - 1, rhs[1], &B[js + (je + 1) * ldb], ldb,
                                    &C[is + (je + 1) * ldc], ldc);
                        cblas_saxpy(n - je - 1, rhs[1], &E[js + (je + 1) * lde], lde,
                                    &F[is + (je + 1) * ldf], ldf);
                    }

                } else if ((mb == 1) && (nb == 2)) {

                    /* Build a 4-by-4 system Z * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = ZERO;
                    z[2 + 0 * LDZ] = D[is + is * ldd];
                    z[3 + 0 * LDZ] = ZERO;

                    z[0 + 1 * LDZ] = ZERO;
                    z[1 + 1 * LDZ] = A[is + is * lda];
                    z[2 + 1 * LDZ] = ZERO;
                    z[3 + 1 * LDZ] = D[is + is * ldd];

                    z[0 + 2 * LDZ] = -B[js + js * ldb];
                    z[1 + 2 * LDZ] = -B[js + jsp1 * ldb];
                    z[2 + 2 * LDZ] = -E[js + js * lde];
                    z[3 + 2 * LDZ] = -E[js + jsp1 * lde];

                    z[0 + 3 * LDZ] = -B[jsp1 + js * ldb];
                    z[1 + 3 * LDZ] = -B[jsp1 + jsp1 * ldb];
                    z[2 + 3 * LDZ] = ZERO;
                    z[3 + 3 * LDZ] = -E[jsp1 + jsp1 * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = C[is + jsp1 * ldc];
                    rhs[2] = F[is + js * ldf];
                    rhs[3] = F[is + jsp1 * ldf];

                    /* Solve Z * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }

                    if (ijob == 0) {
                        sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                        if (scaloc != ONE) {
                            for (k = 0; k < n; k++) {
                                cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                                cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                            }
                            *scale = (*scale) * scaloc;
                        }
                    } else {
                        slatdf(ijob, zdim, z, LDZ, rhs, rdsum, rdscal, ipiv, jpiv);
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    C[is + jsp1 * ldc] = rhs[1];
                    F[is + js * ldf] = rhs[2];
                    F[is + jsp1 * ldf] = rhs[3];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (i > 0) {
                        cblas_sger(CblasColMajor, is, nb, -ONE, &A[0 + is * lda], 1,
                                   &rhs[0], 1, &C[0 + js * ldc], ldc);
                        cblas_sger(CblasColMajor, is, nb, -ONE, &D[0 + is * ldd], 1,
                                   &rhs[0], 1, &F[0 + js * ldf], ldf);
                    }
                    if (j < q - 1) {
                        cblas_saxpy(n - je - 1, rhs[2], &B[js + (je + 1) * ldb], ldb,
                                    &C[is + (je + 1) * ldc], ldc);
                        cblas_saxpy(n - je - 1, rhs[2], &E[js + (je + 1) * lde], lde,
                                    &F[is + (je + 1) * ldf], ldf);
                        cblas_saxpy(n - je - 1, rhs[3], &B[jsp1 + (je + 1) * ldb], ldb,
                                    &C[is + (je + 1) * ldc], ldc);
                        cblas_saxpy(n - je - 1, rhs[3], &E[jsp1 + (je + 1) * lde], lde,
                                    &F[is + (je + 1) * ldf], ldf);
                    }

                } else if ((mb == 2) && (nb == 1)) {

                    /* Build a 4-by-4 system Z * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = A[isp1 + is * lda];
                    z[2 + 0 * LDZ] = D[is + is * ldd];
                    z[3 + 0 * LDZ] = ZERO;

                    z[0 + 1 * LDZ] = A[is + isp1 * lda];
                    z[1 + 1 * LDZ] = A[isp1 + isp1 * lda];
                    z[2 + 1 * LDZ] = D[is + isp1 * ldd];
                    z[3 + 1 * LDZ] = D[isp1 + isp1 * ldd];

                    z[0 + 2 * LDZ] = -B[js + js * ldb];
                    z[1 + 2 * LDZ] = ZERO;
                    z[2 + 2 * LDZ] = -E[js + js * lde];
                    z[3 + 2 * LDZ] = ZERO;

                    z[0 + 3 * LDZ] = ZERO;
                    z[1 + 3 * LDZ] = -B[js + js * ldb];
                    z[2 + 3 * LDZ] = ZERO;
                    z[3 + 3 * LDZ] = -E[js + js * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = C[isp1 + js * ldc];
                    rhs[2] = F[is + js * ldf];
                    rhs[3] = F[isp1 + js * ldf];

                    /* Solve Z * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }
                    if (ijob == 0) {
                        sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                        if (scaloc != ONE) {
                            for (k = 0; k < n; k++) {
                                cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                                cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                            }
                            *scale = (*scale) * scaloc;
                        }
                    } else {
                        slatdf(ijob, zdim, z, LDZ, rhs, rdsum, rdscal, ipiv, jpiv);
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    C[isp1 + js * ldc] = rhs[1];
                    F[is + js * ldf] = rhs[2];
                    F[isp1 + js * ldf] = rhs[3];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (i > 0) {
                        cblas_sgemv(CblasColMajor, CblasNoTrans, is, mb, -ONE,
                                    &A[0 + is * lda], lda, &rhs[0], 1, ONE,
                                    &C[0 + js * ldc], 1);
                        cblas_sgemv(CblasColMajor, CblasNoTrans, is, mb, -ONE,
                                    &D[0 + is * ldd], ldd, &rhs[0], 1, ONE,
                                    &F[0 + js * ldf], 1);
                    }
                    if (j < q - 1) {
                        cblas_sger(CblasColMajor, mb, n - je - 1, ONE, &rhs[2], 1,
                                   &B[js + (je + 1) * ldb], ldb, &C[is + (je + 1) * ldc], ldc);
                        cblas_sger(CblasColMajor, mb, n - je - 1, ONE, &rhs[2], 1,
                                   &E[js + (je + 1) * lde], lde, &F[is + (je + 1) * ldf], ldf);
                    }

                } else if ((mb == 2) && (nb == 2)) {

                    /* Build an 8-by-8 system Z * x = RHS */
                    slaset("F", LDZ, LDZ, ZERO, ZERO, z, LDZ);

                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = A[isp1 + is * lda];
                    z[4 + 0 * LDZ] = D[is + is * ldd];

                    z[0 + 1 * LDZ] = A[is + isp1 * lda];
                    z[1 + 1 * LDZ] = A[isp1 + isp1 * lda];
                    z[4 + 1 * LDZ] = D[is + isp1 * ldd];
                    z[5 + 1 * LDZ] = D[isp1 + isp1 * ldd];

                    z[2 + 2 * LDZ] = A[is + is * lda];
                    z[3 + 2 * LDZ] = A[isp1 + is * lda];
                    z[6 + 2 * LDZ] = D[is + is * ldd];

                    z[2 + 3 * LDZ] = A[is + isp1 * lda];
                    z[3 + 3 * LDZ] = A[isp1 + isp1 * lda];
                    z[6 + 3 * LDZ] = D[is + isp1 * ldd];
                    z[7 + 3 * LDZ] = D[isp1 + isp1 * ldd];

                    z[0 + 4 * LDZ] = -B[js + js * ldb];
                    z[2 + 4 * LDZ] = -B[js + jsp1 * ldb];
                    z[4 + 4 * LDZ] = -E[js + js * lde];
                    z[6 + 4 * LDZ] = -E[js + jsp1 * lde];

                    z[1 + 5 * LDZ] = -B[js + js * ldb];
                    z[3 + 5 * LDZ] = -B[js + jsp1 * ldb];
                    z[5 + 5 * LDZ] = -E[js + js * lde];
                    z[7 + 5 * LDZ] = -E[js + jsp1 * lde];

                    z[0 + 6 * LDZ] = -B[jsp1 + js * ldb];
                    z[2 + 6 * LDZ] = -B[jsp1 + jsp1 * ldb];
                    z[6 + 6 * LDZ] = -E[jsp1 + jsp1 * lde];

                    z[1 + 7 * LDZ] = -B[jsp1 + js * ldb];
                    z[3 + 7 * LDZ] = -B[jsp1 + jsp1 * ldb];
                    z[7 + 7 * LDZ] = -E[jsp1 + jsp1 * lde];

                    /* Set up right hand side(s) */
                    k = 0;
                    ii = mb * nb;
                    for (jj = 0; jj < nb; jj++) {
                        cblas_scopy(mb, &C[is + (js + jj) * ldc], 1, &rhs[k], 1);
                        cblas_scopy(mb, &F[is + (js + jj) * ldf], 1, &rhs[ii], 1);
                        k = k + mb;
                        ii = ii + mb;
                    }

                    /* Solve Z * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }
                    if (ijob == 0) {
                        sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                        if (scaloc != ONE) {
                            for (k = 0; k < n; k++) {
                                cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                                cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                            }
                            *scale = (*scale) * scaloc;
                        }
                    } else {
                        slatdf(ijob, zdim, z, LDZ, rhs, rdsum, rdscal, ipiv, jpiv);
                    }

                    /* Unpack solution vector(s) */
                    k = 0;
                    ii = mb * nb;
                    for (jj = 0; jj < nb; jj++) {
                        cblas_scopy(mb, &rhs[k], 1, &C[is + (js + jj) * ldc], 1);
                        cblas_scopy(mb, &rhs[ii], 1, &F[is + (js + jj) * ldf], 1);
                        k = k + mb;
                        ii = ii + mb;
                    }

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (i > 0) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    is, nb, mb, -ONE, &A[0 + is * lda], lda,
                                    &rhs[0], mb, ONE, &C[0 + js * ldc], ldc);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    is, nb, mb, -ONE, &D[0 + is * ldd], ldd,
                                    &rhs[0], mb, ONE, &F[0 + js * ldf], ldf);
                    }
                    if (j < q - 1) {
                        k = mb * nb;
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    mb, n - je - 1, nb, ONE, &rhs[k], mb,
                                    &B[js + (je + 1) * ldb], ldb, ONE,
                                    &C[is + (je + 1) * ldc], ldc);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    mb, n - je - 1, nb, ONE, &rhs[k], mb,
                                    &E[js + (je + 1) * lde], lde, ONE,
                                    &F[is + (je + 1) * ldf], ldf);
                    }

                }

            }
        }

    } else {

        /* Solve (I, J) - subsystem
             A(I, I)**T * R(I, J) + D(I, I)**T * L(J, J)  =  C(I, J)
             R(I, I)  * B(J, J) + L(I, J)  * E(J, J)  = -F(I, J)
           for I = 1, 2, ..., P, J = Q, Q - 1, ..., 1 */

        *scale = ONE;
        scaloc = ONE;

        for (i = 0; i < p; i++) {

            is = iwork[i];
            isp1 = is + 1;
            ie = iwork[i + 1] - 1;
            mb = ie - is + 1;

            for (j = q - 1; j >= p + 1; j--) {

                js = iwork[j];
                jsp1 = js + 1;
                je = iwork[j + 1] - 1;
                nb = je - js + 1;
                zdim = mb * nb * 2;

                if ((mb == 1) && (nb == 1)) {

                    /* Build a 2-by-2 system Z**T * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = -B[js + js * ldb];
                    z[0 + 1 * LDZ] = D[is + is * ldd];
                    z[1 + 1 * LDZ] = -E[js + js * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = F[is + js * ldf];

                    /* Solve Z**T * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }

                    sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                    if (scaloc != ONE) {
                        for (k = 0; k < n; k++) {
                            cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    F[is + js * ldf] = rhs[1];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (j > p + 1) {
                        alpha = rhs[0];
                        cblas_saxpy(js, alpha, &B[0 + js * ldb], 1, &F[is + 0 * ldf], ldf);
                        alpha = rhs[1];
                        cblas_saxpy(js, alpha, &E[0 + js * lde], 1, &F[is + 0 * ldf], ldf);
                    }
                    if (i < p - 1) {
                        alpha = -rhs[0];
                        cblas_saxpy(m - ie - 1, alpha, &A[is + (ie + 1) * lda], lda,
                                    &C[(ie + 1) + js * ldc], 1);
                        alpha = -rhs[1];
                        cblas_saxpy(m - ie - 1, alpha, &D[is + (ie + 1) * ldd], ldd,
                                    &C[(ie + 1) + js * ldc], 1);
                    }

                } else if ((mb == 1) && (nb == 2)) {

                    /* Build a 4-by-4 system Z**T * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = ZERO;
                    z[2 + 0 * LDZ] = -B[js + js * ldb];
                    z[3 + 0 * LDZ] = -B[jsp1 + js * ldb];

                    z[0 + 1 * LDZ] = ZERO;
                    z[1 + 1 * LDZ] = A[is + is * lda];
                    z[2 + 1 * LDZ] = -B[js + jsp1 * ldb];
                    z[3 + 1 * LDZ] = -B[jsp1 + jsp1 * ldb];

                    z[0 + 2 * LDZ] = D[is + is * ldd];
                    z[1 + 2 * LDZ] = ZERO;
                    z[2 + 2 * LDZ] = -E[js + js * lde];
                    z[3 + 2 * LDZ] = ZERO;

                    z[0 + 3 * LDZ] = ZERO;
                    z[1 + 3 * LDZ] = D[is + is * ldd];
                    z[2 + 3 * LDZ] = -E[js + jsp1 * lde];
                    z[3 + 3 * LDZ] = -E[jsp1 + jsp1 * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = C[is + jsp1 * ldc];
                    rhs[2] = F[is + js * ldf];
                    rhs[3] = F[is + jsp1 * ldf];

                    /* Solve Z**T * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }
                    sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                    if (scaloc != ONE) {
                        for (k = 0; k < n; k++) {
                            cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    C[is + jsp1 * ldc] = rhs[1];
                    F[is + js * ldf] = rhs[2];
                    F[is + jsp1 * ldf] = rhs[3];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (j > p + 1) {
                        cblas_saxpy(js, rhs[0], &B[0 + js * ldb], 1, &F[is + 0 * ldf], ldf);
                        cblas_saxpy(js, rhs[1], &B[0 + jsp1 * ldb], 1, &F[is + 0 * ldf], ldf);
                        cblas_saxpy(js, rhs[2], &E[0 + js * lde], 1, &F[is + 0 * ldf], ldf);
                        cblas_saxpy(js, rhs[3], &E[0 + jsp1 * lde], 1, &F[is + 0 * ldf], ldf);
                    }
                    if (i < p - 1) {
                        cblas_sger(CblasColMajor, m - ie - 1, nb, -ONE, &A[is + (ie + 1) * lda], lda,
                                   &rhs[0], 1, &C[(ie + 1) + js * ldc], ldc);
                        cblas_sger(CblasColMajor, m - ie - 1, nb, -ONE, &D[is + (ie + 1) * ldd], ldd,
                                   &rhs[2], 1, &C[(ie + 1) + js * ldc], ldc);
                    }

                } else if ((mb == 2) && (nb == 1)) {

                    /* Build a 4-by-4 system Z**T * x = RHS */
                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = A[is + isp1 * lda];
                    z[2 + 0 * LDZ] = -B[js + js * ldb];
                    z[3 + 0 * LDZ] = ZERO;

                    z[0 + 1 * LDZ] = A[isp1 + is * lda];
                    z[1 + 1 * LDZ] = A[isp1 + isp1 * lda];
                    z[2 + 1 * LDZ] = ZERO;
                    z[3 + 1 * LDZ] = -B[js + js * ldb];

                    z[0 + 2 * LDZ] = D[is + is * ldd];
                    z[1 + 2 * LDZ] = D[is + isp1 * ldd];
                    z[2 + 2 * LDZ] = -E[js + js * lde];
                    z[3 + 2 * LDZ] = ZERO;

                    z[0 + 3 * LDZ] = ZERO;
                    z[1 + 3 * LDZ] = D[isp1 + isp1 * ldd];
                    z[2 + 3 * LDZ] = ZERO;
                    z[3 + 3 * LDZ] = -E[js + js * lde];

                    /* Set up right hand side(s) */
                    rhs[0] = C[is + js * ldc];
                    rhs[1] = C[isp1 + js * ldc];
                    rhs[2] = F[is + js * ldf];
                    rhs[3] = F[isp1 + js * ldf];

                    /* Solve Z**T * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }

                    sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                    if (scaloc != ONE) {
                        for (k = 0; k < n; k++) {
                            cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }

                    /* Unpack solution vector(s) */
                    C[is + js * ldc] = rhs[0];
                    C[isp1 + js * ldc] = rhs[1];
                    F[is + js * ldf] = rhs[2];
                    F[isp1 + js * ldf] = rhs[3];

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (j > p + 1) {
                        cblas_sger(CblasColMajor, mb, js, ONE, &rhs[0], 1,
                                   &B[0 + js * ldb], 1, &F[is + 0 * ldf], ldf);
                        cblas_sger(CblasColMajor, mb, js, ONE, &rhs[2], 1,
                                   &E[0 + js * lde], 1, &F[is + 0 * ldf], ldf);
                    }
                    if (i < p - 1) {
                        cblas_sgemv(CblasColMajor, CblasTrans, mb, m - ie - 1, -ONE,
                                    &A[is + (ie + 1) * lda], lda, &rhs[0], 1, ONE,
                                    &C[(ie + 1) + js * ldc], 1);
                        cblas_sgemv(CblasColMajor, CblasTrans, mb, m - ie - 1, -ONE,
                                    &D[is + (ie + 1) * ldd], ldd, &rhs[2], 1, ONE,
                                    &C[(ie + 1) + js * ldc], 1);
                    }

                } else if ((mb == 2) && (nb == 2)) {

                    /* Build an 8-by-8 system Z**T * x = RHS */
                    slaset("F", LDZ, LDZ, ZERO, ZERO, z, LDZ);

                    z[0 + 0 * LDZ] = A[is + is * lda];
                    z[1 + 0 * LDZ] = A[is + isp1 * lda];
                    z[4 + 0 * LDZ] = -B[js + js * ldb];
                    z[6 + 0 * LDZ] = -B[jsp1 + js * ldb];

                    z[0 + 1 * LDZ] = A[isp1 + is * lda];
                    z[1 + 1 * LDZ] = A[isp1 + isp1 * lda];
                    z[5 + 1 * LDZ] = -B[js + js * ldb];
                    z[7 + 1 * LDZ] = -B[jsp1 + js * ldb];

                    z[2 + 2 * LDZ] = A[is + is * lda];
                    z[3 + 2 * LDZ] = A[is + isp1 * lda];
                    z[4 + 2 * LDZ] = -B[js + jsp1 * ldb];
                    z[6 + 2 * LDZ] = -B[jsp1 + jsp1 * ldb];

                    z[2 + 3 * LDZ] = A[isp1 + is * lda];
                    z[3 + 3 * LDZ] = A[isp1 + isp1 * lda];
                    z[5 + 3 * LDZ] = -B[js + jsp1 * ldb];
                    z[7 + 3 * LDZ] = -B[jsp1 + jsp1 * ldb];

                    z[0 + 4 * LDZ] = D[is + is * ldd];
                    z[1 + 4 * LDZ] = D[is + isp1 * ldd];
                    z[4 + 4 * LDZ] = -E[js + js * lde];

                    z[1 + 5 * LDZ] = D[isp1 + isp1 * ldd];
                    z[5 + 5 * LDZ] = -E[js + js * lde];

                    z[2 + 6 * LDZ] = D[is + is * ldd];
                    z[3 + 6 * LDZ] = D[is + isp1 * ldd];
                    z[4 + 6 * LDZ] = -E[js + jsp1 * lde];
                    z[6 + 6 * LDZ] = -E[jsp1 + jsp1 * lde];

                    z[3 + 7 * LDZ] = D[isp1 + isp1 * ldd];
                    z[5 + 7 * LDZ] = -E[js + jsp1 * lde];
                    z[7 + 7 * LDZ] = -E[jsp1 + jsp1 * lde];

                    /* Set up right hand side(s) */
                    k = 0;
                    ii = mb * nb;
                    for (jj = 0; jj < nb; jj++) {
                        cblas_scopy(mb, &C[is + (js + jj) * ldc], 1, &rhs[k], 1);
                        cblas_scopy(mb, &F[is + (js + jj) * ldf], 1, &rhs[ii], 1);
                        k = k + mb;
                        ii = ii + mb;
                    }

                    /* Solve Z**T * x = RHS */
                    sgetc2(zdim, z, LDZ, ipiv, jpiv, &ierr);
                    if (ierr > 0) {
                        *info = ierr;
                    }

                    sgesc2(zdim, z, LDZ, rhs, ipiv, jpiv, &scaloc);
                    if (scaloc != ONE) {
                        for (k = 0; k < n; k++) {
                            cblas_sscal(m, scaloc, &C[0 + k * ldc], 1);
                            cblas_sscal(m, scaloc, &F[0 + k * ldf], 1);
                        }
                        *scale = (*scale) * scaloc;
                    }

                    /* Unpack solution vector(s) */
                    k = 0;
                    ii = mb * nb;
                    for (jj = 0; jj < nb; jj++) {
                        cblas_scopy(mb, &rhs[k], 1, &C[is + (js + jj) * ldc], 1);
                        cblas_scopy(mb, &rhs[ii], 1, &F[is + (js + jj) * ldf], 1);
                        k = k + mb;
                        ii = ii + mb;
                    }

                    /* Substitute R(I, J) and L(I, J) into remaining equation. */
                    if (j > p + 1) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    mb, js, nb, ONE, &C[is + js * ldc], ldc,
                                    &B[0 + js * ldb], ldb, ONE, &F[is + 0 * ldf], ldf);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    mb, js, nb, ONE, &F[is + js * ldf], ldf,
                                    &E[0 + js * lde], lde, ONE, &F[is + 0 * ldf], ldf);
                    }
                    if (i < p - 1) {
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    m - ie - 1, nb, mb, -ONE, &A[is + (ie + 1) * lda], lda,
                                    &C[is + js * ldc], ldc, ONE, &C[(ie + 1) + js * ldc], ldc);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    m - ie - 1, nb, mb, -ONE, &D[is + (ie + 1) * ldd], ldd,
                                    &F[is + js * ldf], ldf, ONE, &C[(ie + 1) + js * ldc], ldc);
                    }

                }

            }
        }

    }
}

#undef LDZ
