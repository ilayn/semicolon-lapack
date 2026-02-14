/**
 * @file dtgex2.c
 * @brief DTGEX2 swaps adjacent diagonal blocks in an upper (quasi) triangular
 *        matrix pair by an orthogonal equivalence transformation.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

#define LDST 4
#define WANDS 1

/**
 * DTGEX2 swaps adjacent diagonal blocks (A11, B11) and (A22, B22)
 * of size 1-by-1 or 2-by-2 in an upper (quasi) triangular matrix pair
 * (A, B) by an orthogonal equivalence transformation.
 *
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Array of dimension (lda, n). On entry, the matrix A.
 *                        On exit, the updated matrix A.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Array of dimension (ldb, n). On entry, the matrix B.
 *                        On exit, the updated matrix B.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q       Array of dimension (ldq, n). If wantq, the orthogonal matrix Q.
 *                        On exit, the updated matrix Q. Not referenced if !wantq.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Array of dimension (ldz, n). If wantz, the orthogonal matrix Z.
 *                        On exit, the updated matrix Z. Not referenced if !wantz.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[in]     j1      The index to the first block (A11, B11). 0 <= j1 < n (0-based).
 * @param[in]     n1      The order of the first block. n1 = 0, 1, or 2.
 * @param[in]     n2      The order of the second block. n2 = 0, 1, or 2.
 * @param[out]    work    Array of dimension (lwork).
 * @param[in]     lwork   The dimension of work. lwork >= max(1, n*(n2+n1), (n2+n1)^2 * 2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - = 1: the swap was rejected; blocks not swapped
 *                         - = -16: lwork too small
 */
void dtgex2(
    const int wantq,
    const int wantz,
    const int n,
    f64* restrict A,
    const int lda,
    f64* restrict B,
    const int ldb,
    f64* restrict Q,
    const int ldq,
    f64* restrict Z,
    const int ldz,
    const int j1,
    const int n1,
    const int n2,
    f64* restrict work,
    const int lwork,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWENTY = 20.0;

    int weak, strong;
    int i, idum, linfo, m;
    f64 bqra21, brqa21, ddum, dnorma, dnormb, dscale;
    f64 dsum, eps, f, g, sa, sb, scale, smlnum;
    f64 thresha, threshb;

    int iwork[LDST + 2];
    f64 ai[2], ar[2], be[2];
    f64 ir[LDST * LDST], ircop[LDST * LDST];
    f64 li[LDST * LDST], licop[LDST * LDST];
    f64 s[LDST * LDST], scpy[LDST * LDST];
    f64 t[LDST * LDST], tcpy[LDST * LDST];
    f64 taul[LDST], taur[LDST];

    *info = 0;

    /* Quick return if possible */
    if (n <= 1 || n1 <= 0 || n2 <= 0) {
        return;
    }
    if (n1 > n || (j1 + n1) > n) {
        return;
    }
    m = n1 + n2;

    int max_nm = n * m;
    int max_m2 = m * m * 2;
    int max_work = 1 > max_nm ? 1 : max_nm;
    max_work = max_work > max_m2 ? max_work : max_m2;

    if (lwork < max_work) {
        *info = -16;
        work[0] = (f64)max_work;
        return;
    }

    weak = 0;
    strong = 0;

    /* Make a local copy of selected block */
    dlaset("Full", LDST, LDST, ZERO, ZERO, li, LDST);
    dlaset("Full", LDST, LDST, ZERO, ZERO, ir, LDST);
    dlacpy("Full", m, m, &A[j1 + j1 * lda], lda, s, LDST);
    dlacpy("Full", m, m, &B[j1 + j1 * ldb], ldb, t, LDST);

    /* Compute threshold for testing acceptance of swapping. */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    dscale = ZERO;
    dsum = ONE;
    dlacpy("Full", m, m, s, LDST, work, m);
    dlassq(m * m, work, 1, &dscale, &dsum);
    dnorma = dscale * sqrt(dsum);
    dscale = ZERO;
    dsum = ONE;
    dlacpy("Full", m, m, t, LDST, work, m);
    dlassq(m * m, work, 1, &dscale, &dsum);
    dnormb = dscale * sqrt(dsum);

    thresha = TWENTY * eps * dnorma > smlnum ? TWENTY * eps * dnorma : smlnum;
    threshb = TWENTY * eps * dnormb > smlnum ? TWENTY * eps * dnormb : smlnum;

    if (m == 2) {

        /* CASE 1: Swap 1-by-1 and 1-by-1 blocks.
           Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks
           using Givens rotations and perform the swap tentatively. */

        f = s[1 + 1 * LDST] * t[0 + 0 * LDST] - t[1 + 1 * LDST] * s[0 + 0 * LDST];
        g = s[1 + 1 * LDST] * t[0 + 1 * LDST] - t[1 + 1 * LDST] * s[0 + 1 * LDST];
        sa = fabs(s[1 + 1 * LDST]) * fabs(t[0 + 0 * LDST]);
        sb = fabs(s[0 + 0 * LDST]) * fabs(t[1 + 1 * LDST]);
        dlartg(f, g, &ir[0 + 1 * LDST], &ir[0 + 0 * LDST], &ddum);
        ir[1 + 0 * LDST] = -ir[0 + 1 * LDST];
        ir[1 + 1 * LDST] = ir[0 + 0 * LDST];
        cblas_drot(2, &s[0 + 0 * LDST], 1, &s[0 + 1 * LDST], 1, ir[0 + 0 * LDST], ir[1 + 0 * LDST]);
        cblas_drot(2, &t[0 + 0 * LDST], 1, &t[0 + 1 * LDST], 1, ir[0 + 0 * LDST], ir[1 + 0 * LDST]);
        if (sa >= sb) {
            dlartg(s[0 + 0 * LDST], s[1 + 0 * LDST], &li[0 + 0 * LDST], &li[1 + 0 * LDST], &ddum);
        } else {
            dlartg(t[0 + 0 * LDST], t[1 + 0 * LDST], &li[0 + 0 * LDST], &li[1 + 0 * LDST], &ddum);
        }
        cblas_drot(2, &s[0 + 0 * LDST], LDST, &s[1 + 0 * LDST], LDST, li[0 + 0 * LDST], li[1 + 0 * LDST]);
        cblas_drot(2, &t[0 + 0 * LDST], LDST, &t[1 + 0 * LDST], LDST, li[0 + 0 * LDST], li[1 + 0 * LDST]);
        li[1 + 1 * LDST] = li[0 + 0 * LDST];
        li[0 + 1 * LDST] = -li[1 + 0 * LDST];

        /* Weak stability test: |S21| <= O(EPS F-norm((A)))
                              and |T21| <= O(EPS F-norm((B))) */
        weak = (fabs(s[1 + 0 * LDST]) <= thresha) && (fabs(t[1 + 0 * LDST]) <= threshb);
        if (!weak) {
            goto L70;
        }

        if (WANDS) {

            /* Strong stability test:
                  F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
               and
                  F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B))) */

            dlacpy("Full", m, m, &A[j1 + j1 * lda], lda, &work[m * m], m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, ONE,
                        li, LDST, s, LDST, ZERO, work, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, -ONE,
                        work, m, ir, LDST, ONE, &work[m * m], m);
            dscale = ZERO;
            dsum = ONE;
            dlassq(m * m, &work[m * m], 1, &dscale, &dsum);
            sa = dscale * sqrt(dsum);

            dlacpy("Full", m, m, &B[j1 + j1 * ldb], ldb, &work[m * m], m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, ONE,
                        li, LDST, t, LDST, ZERO, work, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, -ONE,
                        work, m, ir, LDST, ONE, &work[m * m], m);
            dscale = ZERO;
            dsum = ONE;
            dlassq(m * m, &work[m * m], 1, &dscale, &dsum);
            sb = dscale * sqrt(dsum);
            strong = (sa <= thresha) && (sb <= threshb);
            if (!strong) {
                goto L70;
            }
        }

        /* Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
                  (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)). */

        cblas_drot(j1 + 2, &A[0 + j1 * lda], 1, &A[0 + (j1 + 1) * lda], 1,
                   ir[0 + 0 * LDST], ir[1 + 0 * LDST]);
        cblas_drot(j1 + 2, &B[0 + j1 * ldb], 1, &B[0 + (j1 + 1) * ldb], 1,
                   ir[0 + 0 * LDST], ir[1 + 0 * LDST]);
        cblas_drot(n - j1, &A[j1 + j1 * lda], lda, &A[(j1 + 1) + j1 * lda], lda,
                   li[0 + 0 * LDST], li[1 + 0 * LDST]);
        cblas_drot(n - j1, &B[j1 + j1 * ldb], ldb, &B[(j1 + 1) + j1 * ldb], ldb,
                   li[0 + 0 * LDST], li[1 + 0 * LDST]);

        /* Set N1-by-N2 (2,1) - blocks to ZERO. */
        A[(j1 + 1) + j1 * lda] = ZERO;
        B[(j1 + 1) + j1 * ldb] = ZERO;

        /* Accumulate transformations into Q and Z if requested. */
        if (wantz) {
            cblas_drot(n, &Z[0 + j1 * ldz], 1, &Z[0 + (j1 + 1) * ldz], 1,
                       ir[0 + 0 * LDST], ir[1 + 0 * LDST]);
        }
        if (wantq) {
            cblas_drot(n, &Q[0 + j1 * ldq], 1, &Q[0 + (j1 + 1) * ldq], 1,
                       li[0 + 0 * LDST], li[1 + 0 * LDST]);
        }

        /* Exit with INFO = 0 if swap was successfully performed. */
        return;

    } else {

        /* CASE 2: Swap 1-by-1 and 2-by-2 blocks, or 2-by-2 and 2-by-2 blocks.

           Solve the generalized Sylvester equation
                S11 * R - L * S22 = SCALE * S12
                T11 * R - L * T22 = SCALE * T12
           for R and L. Solutions in LI and IR. */

        dlacpy("Full", n1, n2, &t[0 + n1 * LDST], LDST, li, LDST);
        dlacpy("Full", n1, n2, &s[0 + n1 * LDST], LDST, &ir[n2 + (n1) * LDST], LDST);
        dtgsy2("N", 0, n1, n2, s, LDST, &s[n1 + n1 * LDST], LDST,
               &ir[n2 + n1 * LDST], LDST, t, LDST, &t[n1 + n1 * LDST],
               LDST, li, LDST, &scale, &dsum, &dscale, iwork, &idum, &linfo);
        if (linfo != 0) {
            goto L70;
        }

        /* Compute orthogonal matrix QL:
                   QL**T * LI = [ TL ]
                                [ 0  ]
           where
                   LI =  [      -L              ]
                         [ SCALE * identity(N2) ] */

        for (i = 0; i < n2; i++) {
            cblas_dscal(n1, -ONE, &li[0 + i * LDST], 1);
            li[n1 + i + i * LDST] = scale;
        }
        dgeqr2(m, n2, li, LDST, taul, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }
        dorg2r(m, m, n2, li, LDST, taul, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }

        /* Compute orthogonal matrix RQ:
                   IR * RQ**T =   [ 0  TR],
           where IR = [ SCALE * identity(N1), R ] */

        for (i = 0; i < n1; i++) {
            ir[n2 + i + i * LDST] = scale;
        }
        dgerq2(n1, m, &ir[n2 + 0 * LDST], LDST, taur, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }
        dorgr2(m, m, n1, ir, LDST, taur, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }

        /* Perform the swapping tentatively: */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, m, ONE,
                    li, LDST, s, LDST, ZERO, work, m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, ONE,
                    work, m, ir, LDST, ZERO, s, LDST);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, m, ONE,
                    li, LDST, t, LDST, ZERO, work, m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, ONE,
                    work, m, ir, LDST, ZERO, t, LDST);
        dlacpy("F", m, m, s, LDST, scpy, LDST);
        dlacpy("F", m, m, t, LDST, tcpy, LDST);
        dlacpy("F", m, m, ir, LDST, ircop, LDST);
        dlacpy("F", m, m, li, LDST, licop, LDST);

        /* Triangularize the B-part by an RQ factorization.
           Apply transformation (from left) to A-part, giving S. */

        dgerq2(m, m, t, LDST, taur, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }
        dormr2("R", "T", m, m, m, t, LDST, taur, s, LDST, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }
        dormr2("L", "N", m, m, m, t, LDST, taur, ir, LDST, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }

        /* Compute F-norm(S21) in BRQA21. (T21 is 0.) */
        dscale = ZERO;
        dsum = ONE;
        for (i = 0; i < n2; i++) {
            dlassq(n1, &s[n2 + i * LDST], 1, &dscale, &dsum);
        }
        brqa21 = dscale * sqrt(dsum);

        /* Triangularize the B-part by a QR factorization.
           Apply transformation (from right) to A-part, giving S. */

        dgeqr2(m, m, tcpy, LDST, taul, work, &linfo);
        if (linfo != 0) {
            goto L70;
        }
        dorm2r("L", "T", m, m, m, tcpy, LDST, taul, scpy, LDST, work, info);
        dorm2r("R", "N", m, m, m, tcpy, LDST, taul, licop, LDST, work, info);
        if (linfo != 0) {
            goto L70;
        }

        /* Compute F-norm(S21) in BQRA21. (T21 is 0.) */
        dscale = ZERO;
        dsum = ONE;
        for (i = 0; i < n2; i++) {
            dlassq(n1, &scpy[n2 + i * LDST], 1, &dscale, &dsum);
        }
        bqra21 = dscale * sqrt(dsum);

        /* Decide which method to use.
             Weak stability test:
                F-norm(S21) <= O(EPS * F-norm((S))) */

        if (bqra21 <= brqa21 && bqra21 <= thresha) {
            dlacpy("F", m, m, scpy, LDST, s, LDST);
            dlacpy("F", m, m, tcpy, LDST, t, LDST);
            dlacpy("F", m, m, ircop, LDST, ir, LDST);
            dlacpy("F", m, m, licop, LDST, li, LDST);
        } else if (brqa21 >= thresha) {
            goto L70;
        }

        /* Set lower triangle of B-part to zero */
        dlaset("Lower", m - 1, m - 1, ZERO, ZERO, &t[1 + 0 * LDST], LDST);

        if (WANDS) {

            /* Strong stability test:
                  F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
               and
                  F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B))) */

            dlacpy("Full", m, m, &A[j1 + j1 * lda], lda, &work[m * m], m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, ONE,
                        li, LDST, s, LDST, ZERO, work, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, -ONE,
                        work, m, ir, LDST, ONE, &work[m * m], m);
            dscale = ZERO;
            dsum = ONE;
            dlassq(m * m, &work[m * m], 1, &dscale, &dsum);
            sa = dscale * sqrt(dsum);

            dlacpy("Full", m, m, &B[j1 + j1 * ldb], ldb, &work[m * m], m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, ONE,
                        li, LDST, t, LDST, ZERO, work, m);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, -ONE,
                        work, m, ir, LDST, ONE, &work[m * m], m);
            dscale = ZERO;
            dsum = ONE;
            dlassq(m * m, &work[m * m], 1, &dscale, &dsum);
            sb = dscale * sqrt(dsum);
            strong = (sa <= thresha) && (sb <= threshb);
            if (!strong) {
                goto L70;
            }

        }

        /* If the swap is accepted ("weakly" and "strongly"), apply the
           transformations and set N1-by-N2 (2,1)-block to zero. */

        dlaset("Full", n1, n2, ZERO, ZERO, &s[n2 + 0 * LDST], LDST);

        /* copy back M-by-M diagonal block starting at index J1 of (A, B) */
        dlacpy("F", m, m, s, LDST, &A[j1 + j1 * lda], lda);
        dlacpy("F", m, m, t, LDST, &B[j1 + j1 * ldb], ldb);
        dlaset("Full", LDST, LDST, ZERO, ZERO, t, LDST);

        /* Standardize existing 2-by-2 blocks. */
        dlaset("Full", m, m, ZERO, ZERO, work, m);
        work[0] = ONE;
        t[0 + 0 * LDST] = ONE;
        idum = lwork - m * m - 2;
        if (n2 > 1) {
            dlagv2(&A[j1 + j1 * lda], lda, &B[j1 + j1 * ldb], ldb, ar, ai, be,
                   &work[0], &work[1], &t[0 + 0 * LDST], &t[1 + 0 * LDST]);
            work[m] = -work[1];
            work[m + 1] = work[0];
            t[n2 - 1 + (n2 - 1) * LDST] = t[0 + 0 * LDST];
            t[0 + 1 * LDST] = -t[1 + 0 * LDST];
        }
        work[m * m - 1] = ONE;
        t[(m - 1) + (m - 1) * LDST] = ONE;

        if (n1 > 1) {
            dlagv2(&A[(j1 + n2) + (j1 + n2) * lda], lda, &B[(j1 + n2) + (j1 + n2) * ldb], ldb,
                   taur, taul, &work[m * m], &work[n2 * m + n2], &work[n2 * m + n2 + 1],
                   &t[n2 + n2 * LDST], &t[(m - 1) + (m - 2) * LDST]);
            work[m * m - 1] = work[n2 * m + n2];
            work[m * m - 2] = -work[n2 * m + n2 + 1];
            t[(m - 1) + (m - 1) * LDST] = t[n2 + n2 * LDST];
            t[(m - 2) + (m - 1) * LDST] = -t[(m - 1) + (m - 2) * LDST];
        }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n2, n1, n2, ONE,
                    work, m, &A[j1 + (j1 + n2) * lda], lda, ZERO, &work[m * m], n2);
        dlacpy("Full", n2, n1, &work[m * m], n2, &A[j1 + (j1 + n2) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n2, n1, n2, ONE,
                    work, m, &B[j1 + (j1 + n2) * ldb], ldb, ZERO, &work[m * m], n2);
        dlacpy("Full", n2, n1, &work[m * m], n2, &B[j1 + (j1 + n2) * ldb], ldb);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, ONE,
                    li, LDST, work, m, ZERO, &work[m * m], m);
        dlacpy("Full", m, m, &work[m * m], m, li, LDST);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n1, n1, ONE,
                    &A[j1 + (j1 + n2) * lda], lda, &t[n2 + n2 * LDST], LDST, ZERO, work, n2);
        dlacpy("Full", n2, n1, work, n2, &A[j1 + (j1 + n2) * lda], lda);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n2, n1, n1, ONE,
                    &B[j1 + (j1 + n2) * ldb], ldb, &t[n2 + n2 * LDST], LDST, ZERO, work, n2);
        dlacpy("Full", n2, n1, work, n2, &B[j1 + (j1 + n2) * ldb], ldb);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, m, ONE,
                    ir, LDST, t, LDST, ZERO, work, m);
        dlacpy("Full", m, m, work, m, ir, LDST);

        /* Accumulate transformations into Q and Z if requested. */
        if (wantq) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, ONE,
                        &Q[0 + j1 * ldq], ldq, li, LDST, ZERO, work, n);
            dlacpy("Full", n, m, work, n, &Q[0 + j1 * ldq], ldq);
        }

        if (wantz) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m, ONE,
                        &Z[0 + j1 * ldz], ldz, ir, LDST, ZERO, work, n);
            dlacpy("Full", n, m, work, n, &Z[0 + j1 * ldz], ldz);
        }

        /* Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
                   (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)). */

        i = j1 + m;
        if (i < n) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n - i, m, ONE,
                        li, LDST, &A[j1 + i * lda], lda, ZERO, work, m);
            dlacpy("Full", m, n - i, work, m, &A[j1 + i * lda], lda);
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n - i, m, ONE,
                        li, LDST, &B[j1 + i * ldb], ldb, ZERO, work, m);
            dlacpy("Full", m, n - i, work, m, &B[j1 + i * ldb], ldb);
        }
        i = j1;
        if (i > 0) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, i, m, m, ONE,
                        &A[0 + j1 * lda], lda, ir, LDST, ZERO, work, i);
            dlacpy("Full", i, m, work, i, &A[0 + j1 * lda], lda);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, i, m, m, ONE,
                        &B[0 + j1 * ldb], ldb, ir, LDST, ZERO, work, i);
            dlacpy("Full", i, m, work, i, &B[0 + j1 * ldb], ldb);
        }

        /* Exit with INFO = 0 if swap was successfully performed. */
        return;

    }

    /* Exit with INFO = 1 if swap was rejected. */
L70:
    *info = 1;
    return;
}

#undef LDST
#undef WANDS
