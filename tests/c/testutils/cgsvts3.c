/**
 * @file cgsvts3.c
 * @brief CGSVTS3 tests CGGSVD3 for the GSVD of two matrices.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGSVTS3 tests CGGSVD3, which computes the GSVD of an M-by-N matrix A
 * and a P-by-N matrix B:
 *              U'*A*Q = D1*R and V'*B*Q = D2*R.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     p       The number of rows of the matrix B. p >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, overwritten by CGGSVD3.
 * @param[in]     lda     Leading dimension of A and AF. lda >= max(1,m).
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, overwritten by CGGSVD3.
 * @param[in]     ldb     Leading dimension of B and BF. ldb >= max(1,p).
 * @param[out]    U       The M-by-M unitary matrix U.
 * @param[in]     ldu     Leading dimension of U. ldu >= max(1,m).
 * @param[out]    V       The P-by-P unitary matrix V.
 * @param[in]     ldv     Leading dimension of V. ldv >= max(1,p).
 * @param[out]    Q       The N-by-N unitary matrix Q.
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
void cgsvts3(const INT m, const INT p, const INT n,
             const c64* A, c64* AF, const INT lda,
             const c64* B, c64* BF, const INT ldb,
             c64* U, const INT ldu,
             c64* V, const INT ldv,
             c64* Q, const INT ldq,
             f32* alpha, f32* beta,
             c64* R, const INT ldr,
             INT* iwork,
             c64* work, const INT lwork,
             f32* rwork,
             f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, k, l, info;
    INT minval;
    f32 anorm, bnorm, resid, temp, ulp, ulpinv, unfl;

    ulp = slamch("P");
    ulpinv = ONE / ulp;
    unfl = slamch("S");

    clacpy("F", m, n, A, lda, AF, lda);
    clacpy("F", p, n, B, ldb, BF, ldb);

    anorm = clange("1", m, n, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = clange("1", p, n, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    cggsvd3("U", "V", "Q", m, n, p, &k, &l, AF, lda, BF, ldb,
            alpha, beta, U, ldu, V, ldv, Q, ldq, work, lwork,
            rwork, iwork, &info);

    /* Copy R */

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

    /* Compute A:= U'*A*Q - D1*R */

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, &CONE, A, lda, Q, ldq, &CZERO, work, lda);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CONE, U, ldu, work, lda, &CZERO, AF, lda);

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

    /* Compute norm( U'*A*Q - D1*R ) / ( MAX(1,M,N)*norm(A)*ULP ) . */

    resid = clange("1", m, n, AF, lda, rwork);
    if (anorm > ZERO) {
        INT maxmn = (m > n) ? m : n;
        if (maxmn < 1) maxmn = 1;
        result[0] = ((resid / (f32)maxmn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute B := V'*B*Q - D2*R */

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                p, n, n, &CONE, B, ldb, Q, ldq, &CZERO, work, ldb);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                p, n, p, &CONE, V, ldv, work, ldb, &CZERO, BF, ldb);

    for (i = 0; i < l; i++) {
        for (j = i; j < l; j++) {
            BF[i + (n - l + j) * ldb] -= beta[k + i] * R[(k + i) + (k + j) * ldr];
        }
    }

    /* Compute norm( V'*B*Q - D2*R ) / ( MAX(P,N)*norm(B)*ULP ) . */

    resid = clange("1", p, n, BF, ldb, rwork);
    if (bnorm > ZERO) {
        INT maxpn = (p > n) ? p : n;
        if (maxpn < 1) maxpn = 1;
        result[1] = ((resid / (f32)maxpn) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - U'*U */

    claset("F", m, m, CZERO, CONE, work, ldu);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -ONE, U, ldu, ONE, work, ldu);

    /* Compute norm( I - U'*U ) / ( M * ULP ) . */

    resid = clanhe("1", "U", m, work, ldu, rwork);
    result[2] = (resid / (f32)(m > 1 ? m : 1)) / ulp;

    /* Compute I - V'*V */

    claset("F", p, p, CZERO, CONE, work, ldv);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                p, p, -ONE, V, ldv, ONE, work, ldv);

    /* Compute norm( I - V'*V ) / ( P * ULP ) . */

    resid = clanhe("1", "U", p, work, ldv, rwork);
    result[3] = (resid / (f32)(p > 1 ? p : 1)) / ulp;

    /* Compute I - Q'*Q */

    claset("F", n, n, CZERO, CONE, work, ldq);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, -ONE, Q, ldq, ONE, work, ldq);

    /* Compute norm( I - Q'*Q ) / ( N * ULP ) . */

    resid = clanhe("1", "U", n, work, ldq, rwork);
    result[4] = (resid / (f32)(n > 1 ? n : 1)) / ulp;

    /* Check sorting */

    cblas_scopy(n, alpha, 1, rwork, 1);
    minval = k + l;
    if (m < minval) minval = m;
    for (i = k; i < minval; i++) {
        j = iwork[i];
        if (i != j) {
            temp = rwork[i];
            rwork[i] = rwork[j];
            rwork[j] = temp;
        }
    }

    result[5] = ZERO;
    for (i = k; i < minval - 1; i++) {
        if (rwork[i] < rwork[i + 1]) {
            result[5] = ulpinv;
        }
    }
}
