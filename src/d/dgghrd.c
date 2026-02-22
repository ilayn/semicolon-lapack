/**
 * @file dgghrd.c
 * @brief DGGHRD reduces a pair of real matrices (A,B) to generalized upper Hessenberg form.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DGGHRD reduces a pair of real matrices (A,B) to generalized upper
 * Hessenberg form using orthogonal transformations, where A is a
 * general matrix and B is upper triangular. The form of the
 * generalized eigenvalue problem is
 *    A*x = lambda*B*x,
 * and B is typically made upper triangular by computing its QR
 * factorization and moving the orthogonal matrix Q to the left side
 * of the equation.
 *
 * This subroutine simultaneously reduces A to a Hessenberg matrix H:
 *    Q**T*A*Z = H
 * and transforms B to another upper triangular matrix T:
 *    Q**T*B*Z = T
 * in order to reduce the problem to its standard form
 *    H*y = lambda*T*y
 * where y = Z**T*x.
 *
 * @param[in]     compq   = 'N': do not compute Q;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'V': Q must contain an orthogonal matrix Q1 on entry.
 * @param[in]     compz   = 'N': do not compute Z;
 *                        = 'I': Z is initialized to the unit matrix;
 *                        = 'V': Z must contain an orthogonal matrix Z1 on entry.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in]     ilo     See ihi.
 * @param[in]     ihi     ILO and IHI mark the rows and columns of A which are to be
 *                        reduced. It is assumed that A is already upper triangular in
 *                        rows and columns 0:ilo-1 and ihi+1:n-1. ILO and IHI are normally
 *                        set by a previous call to DGGBAL; otherwise they should be set
 *                        to 0 and n-1 respectively.
 *                        0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 * @param[in,out] A       Array of dimension (lda, n). On entry, the N-by-N general
 *                        matrix. On exit, the upper Hessenberg matrix H.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Array of dimension (ldb, n). On entry, the upper triangular
 *                        matrix B. On exit, the upper triangular matrix T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[in,out] Q       Array of dimension (ldq, n). The orthogonal matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= N if COMPQ='V' or 'I'.
 * @param[in,out] Z       Array of dimension (ldz, n). The orthogonal matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= N if COMPZ='V' or 'I'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgghrd(
    const char* compq,
    const char* compz,
    const INT n,
    const INT ilo,
    const INT ihi,
    f64* restrict A,
    const INT lda,
    f64* restrict B,
    const INT ldb,
    f64* restrict Q,
    const INT ldq,
    f64* restrict Z,
    const INT ldz,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT ilq, ilz;
    INT icompq, icompz, jcol, jrow;
    f64 c, s, temp;

    if (compq[0] == 'N' || compq[0] == 'n') {
        ilq = 0;
        icompq = 1;
    } else if (compq[0] == 'V' || compq[0] == 'v') {
        ilq = 1;
        icompq = 2;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        ilq = 1;
        icompq = 3;
    } else {
        icompq = 0;
    }

    if (compz[0] == 'N' || compz[0] == 'n') {
        ilz = 0;
        icompz = 1;
    } else if (compz[0] == 'V' || compz[0] == 'v') {
        ilz = 1;
        icompz = 2;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        ilz = 1;
        icompz = 3;
    } else {
        icompz = 0;
    }

    *info = 0;
    if (icompq <= 0) {
        *info = -1;
    } else if (icompz <= 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 0 || (n > 0 && ilo > n - 1)) {
        *info = -4;
    } else if (ihi > n - 1 || ihi < ilo - 1) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if ((ilq && ldq < n) || ldq < 1) {
        *info = -11;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -13;
    }
    if (*info != 0) {
        xerbla("DGGHRD", -(*info));
        return;
    }

    if (icompq == 3)
        dlaset("F", n, n, ZERO, ONE, Q, ldq);
    if (icompz == 3)
        dlaset("F", n, n, ZERO, ONE, Z, ldz);

    if (n <= 1)
        return;

    for (jcol = 0; jcol < n - 1; jcol++) {
        for (jrow = jcol + 1; jrow < n; jrow++) {
            B[jrow + jcol * ldb] = ZERO;
        }
    }

    for (jcol = ilo; jcol <= ihi - 2; jcol++) {
        for (jrow = ihi; jrow >= jcol + 2; jrow--) {
            temp = A[jrow - 1 + jcol * lda];
            dlartg(temp, A[jrow + jcol * lda], &c, &s, &A[jrow - 1 + jcol * lda]);
            A[jrow + jcol * lda] = ZERO;
            cblas_drot(n - jcol - 1, &A[jrow - 1 + (jcol + 1) * lda], lda,
                       &A[jrow + (jcol + 1) * lda], lda, c, s);
            cblas_drot(n + 2 - jrow - 1, &B[jrow - 1 + (jrow - 1) * ldb], ldb,
                       &B[jrow + (jrow - 1) * ldb], ldb, c, s);
            if (ilq)
                cblas_drot(n, &Q[(jrow - 1) * ldq], 1, &Q[jrow * ldq], 1, c, s);

            temp = B[jrow + jrow * ldb];
            dlartg(temp, B[jrow + (jrow - 1) * ldb], &c, &s, &B[jrow + jrow * ldb]);
            B[jrow + (jrow - 1) * ldb] = ZERO;
            cblas_drot(ihi + 1, &A[jrow * lda], 1, &A[(jrow - 1) * lda], 1, c, s);
            cblas_drot(jrow, &B[jrow * ldb], 1, &B[(jrow - 1) * ldb], 1, c, s);
            if (ilz)
                cblas_drot(n, &Z[jrow * ldz], 1, &Z[(jrow - 1) * ldz], 1, c, s);
        }
    }
}
