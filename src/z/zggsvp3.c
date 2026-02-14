/**
 * @file zggsvp3.c
 * @brief ZGGSVP3 computes unitary matrices U, V, Q for GSVD preprocessing.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGGSVP3 computes unitary matrices U, V and Q such that
 *
 *                    N-K-L  K    L
 *  U**H*A*Q =     K ( 0    A12  A13 )  if M-K-L >= 0;
 *                 L ( 0     0   A23 )
 *             M-K-L ( 0     0    0  )
 *
 *                  N-K-L  K    L
 *         =     K ( 0    A12  A13 )  if M-K-L < 0;
 *             M-K ( 0     0   A23 )
 *
 *                  N-K-L  K    L
 *  V**H*B*Q =   L ( 0     0   B13 )
 *             P-L ( 0     0    0  )
 *
 * where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
 * upper triangular. K+L = the effective numerical rank of (A**H,B**H)**H.
 *
 * @param[in]     jobu    = 'U': Unitary matrix U is computed;
 *                        = 'N': U is not computed.
 * @param[in]     jobv    = 'V': Unitary matrix V is computed;
 *                        = 'N': V is not computed.
 * @param[in]     jobq    = 'Q': Unitary matrix Q is computed;
 *                        = 'N': Q is not computed.
 * @param[in]     m       The number of rows of matrix A. m >= 0.
 * @param[in]     p       The number of rows of matrix B. p >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in,out] A       On entry, the M-by-N matrix A.
 *                        On exit, triangular (or trapezoidal) matrix.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] B       On entry, the P-by-N matrix B.
 *                        On exit, triangular matrix.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1,p).
 * @param[in]     tola    Threshold for rank determination of A.
 * @param[in]     tolb    Threshold for rank determination of B.
 * @param[out]    k       Subblock dimension.
 * @param[out]    l       Subblock dimension.
 * @param[out]    U       Unitary matrix U (dimension ldu,m).
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    V       Unitary matrix V (dimension ldv,p).
 * @param[in]     ldv     Leading dimension of V.
 * @param[out]    Q       Unitary matrix Q (dimension ldq,n).
 * @param[in]     ldq     Leading dimension of Q.
 * @param[out]    iwork   Integer workspace of dimension n.
 * @param[out]    rwork   Double precision workspace of dimension 2*n.
 * @param[out]    tau     Complex workspace of dimension n.
 * @param[out]    work    Complex workspace of dimension lwork.
 * @param[in]     lwork   Dimension of work. If lwork = -1, workspace query.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zggsvp3(const char* jobu, const char* jobv, const char* jobq,
             const int m, const int p, const int n,
             double complex* const restrict A, const int lda,
             double complex* const restrict B, const int ldb,
             const double tola, const double tolb,
             int* k, int* l,
             double complex* const restrict U, const int ldu,
             double complex* const restrict V, const int ldv,
             double complex* const restrict Q, const int ldq,
             int* const restrict iwork,
             double* const restrict rwork,
             double complex* const restrict tau,
             double complex* const restrict work, const int lwork,
             int* info)
{
    const double complex CZERO = CMPLX(0.0, 0.0);
    const double complex CONE = CMPLX(1.0, 0.0);

    int wantu, wantv, wantq, lquery;
    int forwrd = 1;
    int i, j, lwkopt;
    int minval, ierr;

    wantu = (jobu[0] == 'U' || jobu[0] == 'u');
    wantv = (jobv[0] == 'V' || jobv[0] == 'v');
    wantq = (jobq[0] == 'Q' || jobq[0] == 'q');
    lquery = (lwork == -1);
    lwkopt = 1;

    *info = 0;
    if (!wantu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!wantv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (!wantq && !(jobq[0] == 'N' || jobq[0] == 'n')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -8;
    } else if (ldb < (1 > p ? 1 : p)) {
        *info = -10;
    } else if (ldu < 1 || (wantu && ldu < m)) {
        *info = -16;
    } else if (ldv < 1 || (wantv && ldv < p)) {
        *info = -18;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -20;
    } else if (lwork < 1 && !lquery) {
        *info = -25;
    }

    if (*info == 0) {
        zgeqp3(p, n, B, ldb, iwork, tau, work, -1, rwork, &ierr);
        lwkopt = (int)creal(work[0]);
        if (wantv) {
            if (p > lwkopt) lwkopt = p;
        }
        minval = n < p ? n : p;
        if (minval > lwkopt) lwkopt = minval;
        if (m > lwkopt) lwkopt = m;
        if (wantq) {
            if (n > lwkopt) lwkopt = n;
        }
        zgeqp3(m, n, A, lda, iwork, tau, work, -1, rwork, &ierr);
        if ((int)creal(work[0]) > lwkopt) lwkopt = (int)creal(work[0]);
        if (lwkopt < 1) lwkopt = 1;
        work[0] = CMPLX((double)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZGGSVP3", -(*info));
        return;
    }
    if (lquery) {
        return;
    }

    for (i = 0; i < n; i++) {
        iwork[i] = 0;
    }
    zgeqp3(p, n, B, ldb, iwork, tau, work, lwork, rwork, info);

    zlapmt(forwrd, m, n, A, lda, iwork);

    *l = 0;
    minval = p < n ? p : n;
    for (i = 0; i < minval; i++) {
        if (cabs(B[i + i * ldb]) > tolb) {
            (*l)++;
        }
    }

    if (wantv) {
        zlaset("F", p, p, CZERO, CZERO, V, ldv);
        if (p > 1) {
            zlacpy("L", p - 1, n, &B[1], ldb, &V[1], ldv);
        }
        minval = p < n ? p : n;
        zung2r(p, p, minval, V, ldv, tau, work, &ierr);
    }

    for (j = 0; j < *l - 1; j++) {
        for (i = j + 1; i < *l; i++) {
            B[i + j * ldb] = CZERO;
        }
    }
    if (p > *l) {
        zlaset("F", p - *l, n, CZERO, CZERO, &B[*l], ldb);
    }

    if (wantq) {
        zlaset("F", n, n, CZERO, CONE, Q, ldq);
        zlapmt(forwrd, n, n, Q, ldq, iwork);
    }

    if (p >= *l && n != *l) {
        zgerq2(*l, n, B, ldb, tau, work, &ierr);

        zunmr2("R", "C", m, n, *l, B, ldb, tau, A, lda, work, &ierr);

        if (wantq) {
            zunmr2("R", "C", n, n, *l, B, ldb, tau, Q, ldq, work, &ierr);
        }

        zlaset("F", *l, n - *l, CZERO, CZERO, B, ldb);
        for (j = n - *l; j < n; j++) {
            for (i = j - n + *l + 1; i < *l; i++) {
                B[i + j * ldb] = CZERO;
            }
        }
    }

    for (i = 0; i < n - *l; i++) {
        iwork[i] = 0;
    }
    zgeqp3(m, n - *l, A, lda, iwork, tau, work, lwork, rwork, info);

    *k = 0;
    minval = m < n - *l ? m : n - *l;
    for (i = 0; i < minval; i++) {
        if (cabs(A[i + i * lda]) > tola) {
            (*k)++;
        }
    }

    minval = m < n - *l ? m : n - *l;
    zunm2r("L", "C", m, *l, minval, A, lda, tau, &A[(n - *l) * lda], lda, work, &ierr);

    if (wantu) {
        zlaset("F", m, m, CZERO, CZERO, U, ldu);
        if (m > 1) {
            zlacpy("L", m - 1, n - *l, &A[1], lda, &U[1], ldu);
        }
        minval = m < n - *l ? m : n - *l;
        zung2r(m, m, minval, U, ldu, tau, work, &ierr);
    }

    if (wantq) {
        zlapmt(forwrd, n, n - *l, Q, ldq, iwork);
    }

    for (j = 0; j < *k - 1; j++) {
        for (i = j + 1; i < *k; i++) {
            A[i + j * lda] = CZERO;
        }
    }
    if (m > *k) {
        zlaset("F", m - *k, n - *l, CZERO, CZERO, &A[*k], lda);
    }

    if (n - *l > *k) {
        zgerq2(*k, n - *l, A, lda, tau, work, &ierr);

        if (wantq) {
            zunmr2("R", "C", n, n - *l, *k, A, lda, tau, Q, ldq, work, &ierr);
        }

        zlaset("F", *k, n - *l - *k, CZERO, CZERO, A, lda);
        for (j = n - *l - *k; j < n - *l; j++) {
            for (i = j - n + *l + *k + 1; i < *k; i++) {
                A[i + j * lda] = CZERO;
            }
        }
    }

    if (m > *k) {
        zgeqr2(m - *k, *l, &A[*k + (n - *l) * lda], lda, tau, work, &ierr);

        if (wantu) {
            minval = m - *k < *l ? m - *k : *l;
            zunm2r("R", "N", m, m - *k, minval,
                   &A[*k + (n - *l) * lda], lda, tau, &U[*k * ldu], ldu, work, &ierr);
        }

        for (j = n - *l; j < n; j++) {
            for (i = j - n + *k + *l + 1; i < m; i++) {
                A[i + j * lda] = CZERO;
            }
        }
    }

    work[0] = CMPLX((double)lwkopt, 0.0);
}
