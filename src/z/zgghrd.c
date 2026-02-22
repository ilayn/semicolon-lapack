/**
 * @file zgghrd.c
 * @brief ZGGHRD reduces a pair of complex matrices (A,B) to generalized upper Hessenberg form.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZGGHRD reduces a pair of complex matrices (A,B) to generalized upper
 * Hessenberg form using unitary transformations, where A is a
 * general matrix and B is upper triangular. The form of the
 * generalized eigenvalue problem is
 *    A*x = lambda*B*x,
 * and B is typically made upper triangular by computing its QR
 * factorization and moving the unitary matrix Q to the left side
 * of the equation.
 *
 * This subroutine simultaneously reduces A to a Hessenberg matrix H:
 *    Q**H*A*Z = H
 * and transforms B to another upper triangular matrix T:
 *    Q**H*B*Z = T
 * in order to reduce the problem to its standard form
 *    H*y = lambda*T*y
 * where y = Z**H*x.
 *
 * The unitary matrices Q and Z are determined as products of Givens
 * rotations. They may either be formed explicitly, or they may be
 * postmultiplied into input matrices Q1 and Z1, so that
 *      Q1 * A * Z1**H = (Q1*Q) * H * (Z1*Z)**H
 *      Q1 * B * Z1**H = (Q1*Q) * T * (Z1*Z)**H
 *
 * @param[in]     compq   = 'N': do not compute Q;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'V': Q must contain a unitary matrix Q1 on entry.
 * @param[in]     compz   = 'N': do not compute Z;
 *                        = 'I': Z is initialized to the unit matrix;
 *                        = 'V': Z must contain a unitary matrix Z1 on entry.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in]     ilo     See ihi.
 * @param[in]     ihi     ILO and IHI mark the rows and columns of A which are to be
 *                        reduced. It is assumed that A is already upper triangular in
 *                        rows and columns 0:ilo-1 and ihi+1:n-1. ILO and IHI are normally
 *                        set by a previous call to ZGGBAL; otherwise they should be set
 *                        to 0 and n-1 respectively.
 *                        0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 * @param[in,out] A       Array of dimension (lda, n). On entry, the N-by-N general
 *                        matrix. On exit, the upper Hessenberg matrix H.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B       Array of dimension (ldb, n). On entry, the upper triangular
 *                        matrix B. On exit, the upper triangular matrix T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[in,out] Q       Array of dimension (ldq, n). The unitary matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= N if COMPQ='V' or 'I'.
 * @param[in,out] Z       Array of dimension (ldz, n). The unitary matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= N if COMPZ='V' or 'I'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zgghrd(
    const char* compq,
    const char* compz,
    const INT n,
    const INT ilo,
    const INT ihi,
    c128* restrict A,
    const INT lda,
    c128* restrict B,
    const INT ldb,
    c128* restrict Q,
    const INT ldq,
    c128* restrict Z,
    const INT ldz,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT ilq, ilz;
    INT icompq, icompz, jcol, jrow;
    f64 c;
    c128 ctemp, s;

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
        xerbla("ZGGHRD", -(*info));
        return;
    }

    if (icompq == 3)
        zlaset("F", n, n, CZERO, CONE, Q, ldq);
    if (icompz == 3)
        zlaset("F", n, n, CZERO, CONE, Z, ldz);

    if (n <= 1)
        return;

    for (jcol = 0; jcol < n - 1; jcol++) {
        for (jrow = jcol + 1; jrow < n; jrow++) {
            B[jrow + jcol * ldb] = CZERO;
        }
    }

    for (jcol = ilo; jcol <= ihi - 2; jcol++) {
        for (jrow = ihi; jrow >= jcol + 2; jrow--) {

            ctemp = A[jrow - 1 + jcol * lda];
            zlartg(ctemp, A[jrow + jcol * lda], &c, &s,
                   &A[jrow - 1 + jcol * lda]);
            A[jrow + jcol * lda] = CZERO;
            zrot(n - jcol - 1, &A[jrow - 1 + (jcol + 1) * lda], lda,
                 &A[jrow + (jcol + 1) * lda], lda, c, s);
            zrot(n + 2 - jrow - 1, &B[jrow - 1 + (jrow - 1) * ldb], ldb,
                 &B[jrow + (jrow - 1) * ldb], ldb, c, s);
            if (ilq)
                zrot(n, &Q[(jrow - 1) * ldq], 1, &Q[jrow * ldq], 1, c,
                     conj(s));

            ctemp = B[jrow + jrow * ldb];
            zlartg(ctemp, B[jrow + (jrow - 1) * ldb], &c, &s,
                   &B[jrow + jrow * ldb]);
            B[jrow + (jrow - 1) * ldb] = CZERO;
            zrot(ihi + 1, &A[jrow * lda], 1, &A[(jrow - 1) * lda], 1, c, s);
            zrot(jrow, &B[jrow * ldb], 1, &B[(jrow - 1) * ldb], 1, c, s);
            if (ilz)
                zrot(n, &Z[jrow * ldz], 1, &Z[(jrow - 1) * ldz], 1, c, s);
        }
    }
}
