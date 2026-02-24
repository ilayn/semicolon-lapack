/**
 * @file sgsvts3.c
 * @brief SGSVTS3 tests SGGSVD3 for the GSVD of two matrices.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SGSVTS3 tests SGGSVD3, which computes the GSVD of an M-by-N matrix A
 * and a P-by-N matrix B:
 *              U'*A*Q = D1*R and V'*B*Q = D2*R.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     p       The number of rows of the matrix B. p >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, overwritten by SGGSVD3.
 * @param[in]     lda     Leading dimension of A and AF. lda >= max(1,m).
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, overwritten by SGGSVD3.
 * @param[in]     ldb     Leading dimension of B and BF. ldb >= max(1,p).
 * @param[out]    U       The M-by-M orthogonal matrix U.
 * @param[in]     ldu     Leading dimension of U. ldu >= max(1,m).
 * @param[out]    V       The P-by-P orthogonal matrix V.
 * @param[in]     ldv     Leading dimension of V. ldv >= max(1,p).
 * @param[out]    Q       The N-by-N orthogonal matrix Q.
 * @param[in]     ldq     Leading dimension of Q. ldq >= max(1,n).
 * @param[out]    alpha   Generalized singular values (dimension n).
 * @param[out]    beta    Generalized singular values (dimension n).
 * @param[out]    R       The upper triangular matrix R (dimension ldr,n).
 * @param[in]     ldr     Leading dimension of R. ldr >= max(1,n).
 * @param[out]    iwork   Integer workspace of dimension n.
 * @param[out]    work    Workspace of dimension lwork.
 * @param[in]     lwork   Dimension of work. lwork >= max(m,p,n)^2.
 * @param[out]    rwork   Real workspace of dimension max(m,p,n).
 * @param[out]    result  Test ratios (dimension 6):
 *                        result[0] = norm(U'*A*Q - D1*R) / (max(m,n)*norm(A)*ulp)
 *                        result[1] = norm(V'*B*Q - D2*R) / (max(p,n)*norm(B)*ulp)
 *                        result[2] = norm(I - U'*U) / (m*ulp)
 *                        result[3] = norm(I - V'*V) / (p*ulp)
 *                        result[4] = norm(I - Q'*Q) / (n*ulp)
 *                        result[5] = 0 if alpha is in decreasing order, ulpinv otherwise
 */
void sgsvts3(const INT m, const INT p, const INT n,
             const f32* A, f32* AF, const INT lda,
             const f32* B, f32* BF, const INT ldb,
             f32* U, const INT ldu,
             f32* V, const INT ldv,
             f32* Q, const INT ldq,
             f32* alpha, f32* beta,
             f32* R, const INT ldr,
             INT* iwork,
             f32* work, const INT lwork,
             f32* rwork,
             f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j, k, l, info;
    INT minval;
    f32 anorm, bnorm, resid, temp, ulp, ulpinv, unfl;

    ulp = slamch("P");
    ulpinv = ONE / ulp;
    unfl = slamch("S");

    slacpy("F", m, n, A, lda, AF, lda);
    slacpy("F", p, n, B, ldb, BF, ldb);

    anorm = slange("1", m, n, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = slange("1", p, n, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    sggsvd3("U", "V", "Q", m, n, p, &k, &l, AF, lda, BF, ldb,
            alpha, beta, U, ldu, V, ldv, Q, ldq, work, lwork,
            iwork, &info);

    minval = k + l;
    if (m < minval) minval = m;
    for (i = 0; i < minval; i++) {
        for (j = i; j < k + l; j++) {
            R[i + j * ldr] = AF[i + (n - k - l + j) * lda];
        }
    }

    if (m - k - l < 0) {
        for (i = m; i < k + l; i++) {
            for (j = i; j < k + l; j++) {
                R[i + j * ldr] = BF[(i - k) + (n - k - l + j) * ldb];
            }
        }
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, ONE, A, lda, Q, ldq, ZERO, work, lda);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, ONE, U, ldu, work, lda, ZERO, AF, lda);

    for (i = 0; i < k; i++) {
        for (j = i; j < k + l; j++) {
            AF[i + (n - k - l + j) * lda] -= R[i + j * ldr];
        }
    }

    minval = k + l;
    if (m < minval) minval = m;
    for (i = k; i < minval; i++) {
        for (j = i; j < k + l; j++) {
            AF[i + (n - k - l + j) * lda] -= alpha[i] * R[i + j * ldr];
        }
    }

    resid = slange("1", m, n, AF, lda, rwork);
    if (anorm > ZERO) {
        INT maxmn = (m > n) ? m : n;
        if (maxmn < 1) maxmn = 1;
        result[0] = ((resid / (f32)maxmn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                p, n, n, ONE, B, ldb, Q, ldq, ZERO, work, ldb);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                p, n, p, ONE, V, ldv, work, ldb, ZERO, BF, ldb);

    for (i = 0; i < l; i++) {
        for (j = i; j < l; j++) {
            BF[i + (n - l + j) * ldb] -= beta[k + i] * R[(k + i) + (k + j) * ldr];
        }
    }

    resid = slange("1", p, n, BF, ldb, rwork);
    if (bnorm > ZERO) {
        INT maxpn = (p > n) ? p : n;
        if (maxpn < 1) maxpn = 1;
        result[1] = ((resid / (f32)maxpn) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    slaset("F", m, m, ZERO, ONE, work, ldu);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -ONE, U, ldu, ONE, work, ldu);

    resid = slansy("1", "U", m, work, ldu, rwork);
    result[2] = (resid / (f32)(m > 1 ? m : 1)) / ulp;

    slaset("F", p, p, ZERO, ONE, work, ldv);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -ONE, V, ldv, ONE, work, ldv);

    resid = slansy("1", "U", p, work, ldv, rwork);
    result[3] = (resid / (f32)(p > 1 ? p : 1)) / ulp;

    slaset("F", n, n, ZERO, ONE, work, ldq);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, n, -ONE, Q, ldq, ONE, work, ldq);

    resid = slansy("1", "U", n, work, ldq, rwork);
    result[4] = (resid / (f32)(n > 1 ? n : 1)) / ulp;

    cblas_scopy(n, alpha, 1, work, 1);
    minval = k + l;
    if (m < minval) minval = m;
    for (i = k; i < minval; i++) {
        j = iwork[i];
        if (i != j) {
            temp = work[i];
            work[i] = work[j];
            work[j] = temp;
        }
    }

    result[5] = ZERO;
    for (i = k; i < minval - 1; i++) {
        if (work[i] < work[i + 1]) {
            result[5] = ulpinv;
        }
    }
}
