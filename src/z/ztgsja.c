/**
 * @file ztgsja.c
 * @brief ZTGSJA computes the GSVD of two upper triangular matrices.
 */

#include <complex.h>
#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

#define MAXIT 40

/**
 * ZTGSJA computes the generalized singular value decomposition (GSVD)
 * of two complex upper triangular (or trapezoidal) matrices A and B.
 *
 * On entry, it is assumed that matrices A and B have the following
 * forms, which may be obtained by the preprocessing subroutine ZGGSVP
 * from a general M-by-N matrix A and P-by-N matrix B:
 *
 *              N-K-L  K    L
 *    A =    K ( 0    A12  A13 ) if M-K-L >= 0;
 *           L ( 0     0   A23 )
 *       M-K-L ( 0     0    0  )
 *
 *            N-K-L  K    L
 *    A =  K ( 0    A12  A13 ) if M-K-L < 0;
 *       M-K ( 0     0   A23 )
 *
 *            N-K-L  K    L
 *    B =  L ( 0     0   B13 )
 *       P-L ( 0     0    0  )
 *
 * On exit,
 *
 *        U**H *A*Q = D1*( 0 R ),    V**H *B*Q = D2*( 0 R ),
 *
 * where U, V and Q are unitary matrices.
 *
 * @param[in]     jobu    = 'U': U must contain a unitary matrix U1 on entry;
 *                        = 'I': U is initialized to the unit matrix;
 *                        = 'N': U is not computed.
 * @param[in]     jobv    = 'V': V must contain a unitary matrix V1 on entry;
 *                        = 'I': V is initialized to the unit matrix;
 *                        = 'N': V is not computed.
 * @param[in]     jobq    = 'Q': Q must contain a unitary matrix Q1 on entry;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'N': Q is not computed.
 * @param[in]     m       The number of rows of matrix A. m >= 0.
 * @param[in]     p       The number of rows of matrix B. p >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in]     k       Subblock dimension from ZGGSVP3.
 * @param[in]     l       Subblock dimension from ZGGSVP3.
 * @param[in,out] A       On entry, the M-by-N matrix A.
 *                        On exit, contains the triangular matrix R.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] B       On entry, the P-by-N matrix B.
 *                        On exit, may contain part of R.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1,p).
 * @param[in]     tola    Convergence threshold for A.
 * @param[in]     tolb    Convergence threshold for B.
 * @param[out]    alpha   Generalized singular values (dimension n).
 * @param[out]    beta    Generalized singular values (dimension n).
 * @param[in,out] U       Unitary matrix U (dimension ldu,m).
 * @param[in]     ldu     Leading dimension of U.
 * @param[in,out] V       Unitary matrix V (dimension ldv,p).
 * @param[in]     ldv     Leading dimension of V.
 * @param[in,out] Q       Unitary matrix Q (dimension ldq,n).
 * @param[in]     ldq     Leading dimension of Q.
 * @param[out]    work    Complex workspace of dimension 2*n.
 * @param[out]    ncycle  Number of cycles for convergence.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: the procedure does not converge after MAXIT cycles.
 */
void ztgsja(const char* jobu, const char* jobv, const char* jobq,
            const INT m, const INT p, const INT n, const INT k, const INT l,
            c128* restrict A, const INT lda,
            c128* restrict B, const INT ldb,
            const f64 tola, const f64 tolb,
            f64* restrict alpha, f64* restrict beta,
            c128* restrict U, const INT ldu,
            c128* restrict V, const INT ldv,
            c128* restrict Q, const INT ldq,
            c128* restrict work, INT* ncycle, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HUGENUM = DBL_MAX;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT initu, wantu, initv, wantv, initq, wantq;
    INT upper;
    INT i, j, kcycle;
    f64 a1, a3, b1, b3;
    f64 csu, csv, csq;
    c128 a2, b2, snu, snv, snq;
    f64 error, gamma, rwk, ssmin;
    INT minval;

    initu = (jobu[0] == 'I' || jobu[0] == 'i');
    wantu = initu || (jobu[0] == 'U' || jobu[0] == 'u');

    initv = (jobv[0] == 'I' || jobv[0] == 'i');
    wantv = initv || (jobv[0] == 'V' || jobv[0] == 'v');

    initq = (jobq[0] == 'I' || jobq[0] == 'i');
    wantq = initq || (jobq[0] == 'Q' || jobq[0] == 'q');

    *info = 0;
    if (!initu && !wantu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!initv && !wantv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (!initq && !wantq && !(jobq[0] == 'N' || jobq[0] == 'n')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -10;
    } else if (ldb < (1 > p ? 1 : p)) {
        *info = -12;
    } else if (ldu < 1 || (wantu && ldu < m)) {
        *info = -18;
    } else if (ldv < 1 || (wantv && ldv < p)) {
        *info = -20;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -22;
    }
    if (*info != 0) {
        xerbla("ZTGSJA", -(*info));
        return;
    }

    if (initu) {
        zlaset("F", m, m, CZERO, CONE, U, ldu);
    }
    if (initv) {
        zlaset("F", p, p, CZERO, CONE, V, ldv);
    }
    if (initq) {
        zlaset("F", n, n, CZERO, CONE, Q, ldq);
    }

    upper = 0;
    for (kcycle = 0; kcycle < MAXIT; kcycle++) {
        upper = !upper;

        for (i = 0; i < l - 1; i++) {
            for (j = i + 1; j < l; j++) {
                a1 = ZERO;
                a2 = CZERO;
                a3 = ZERO;
                if (k + i < m) {
                    a1 = creal(A[(k + i) + (n - l + i) * lda]);
                }
                if (k + j < m) {
                    a3 = creal(A[(k + j) + (n - l + j) * lda]);
                }

                b1 = creal(B[i + (n - l + i) * ldb]);
                b3 = creal(B[j + (n - l + j) * ldb]);

                if (upper) {
                    if (k + i < m) {
                        a2 = A[(k + i) + (n - l + j) * lda];
                    }
                    b2 = B[i + (n - l + j) * ldb];
                } else {
                    if (k + j < m) {
                        a2 = A[(k + j) + (n - l + i) * lda];
                    }
                    b2 = B[j + (n - l + i) * ldb];
                }

                zlags2(upper, a1, a2, a3, b1, b2, b3,
                       &csu, &snu, &csv, &snv, &csq, &snq);

                if (k + j < m) {
                    zrot(l, &A[(k + j) + (n - l) * lda], lda,
                         &A[(k + i) + (n - l) * lda], lda, csu, conj(snu));
                }

                zrot(l, &B[j + (n - l) * ldb], ldb,
                     &B[i + (n - l) * ldb], ldb, csv, conj(snv));

                minval = k + l;
                if (m < minval) minval = m;
                zrot(minval, &A[(n - l + j) * lda], 1,
                     &A[(n - l + i) * lda], 1, csq, snq);

                zrot(l, &B[(n - l + j) * ldb], 1,
                     &B[(n - l + i) * ldb], 1, csq, snq);

                if (upper) {
                    if (k + i < m) {
                        A[(k + i) + (n - l + j) * lda] = CZERO;
                    }
                    B[i + (n - l + j) * ldb] = CZERO;
                } else {
                    if (k + j < m) {
                        A[(k + j) + (n - l + i) * lda] = CZERO;
                    }
                    B[j + (n - l + i) * ldb] = CZERO;
                }

                if (k + i < m) {
                    A[(k + i) + (n - l + i) * lda] = creal(A[(k + i) + (n - l + i) * lda]);
                }
                if (k + j < m) {
                    A[(k + j) + (n - l + j) * lda] = creal(A[(k + j) + (n - l + j) * lda]);
                }
                B[i + (n - l + i) * ldb] = creal(B[i + (n - l + i) * ldb]);
                B[j + (n - l + j) * ldb] = creal(B[j + (n - l + j) * ldb]);

                if (wantu && k + j < m) {
                    zrot(m, &U[(k + j) * ldu], 1,
                         &U[(k + i) * ldu], 1, csu, snu);
                }

                if (wantv) {
                    zrot(p, &V[j * ldv], 1,
                         &V[i * ldv], 1, csv, snv);
                }

                if (wantq) {
                    zrot(n, &Q[(n - l + j) * ldq], 1,
                         &Q[(n - l + i) * ldq], 1, csq, snq);
                }
            }
        }

        if (!upper) {
            error = ZERO;
            minval = l;
            if (m - k < minval) minval = m - k;
            for (i = 0; i < minval; i++) {
                cblas_zcopy(l - i, &A[(k + i) + (n - l + i) * lda], lda, work, 1);
                cblas_zcopy(l - i, &B[i + (n - l + i) * ldb], ldb, &work[l], 1);
                zlapll(l - i, work, 1, &work[l], 1, &ssmin);
                if (ssmin > error) error = ssmin;
            }

            if (fabs(error) <= (tola < tolb ? tola : tolb)) {
                goto converged;
            }
        }
    }

    *info = 1;
    goto finish;

converged:
    for (i = 0; i < k; i++) {
        alpha[i] = ONE;
        beta[i] = ZERO;
    }

    minval = l;
    if (m - k < minval) minval = m - k;
    for (i = 0; i < minval; i++) {
        a1 = creal(A[(k + i) + (n - l + i) * lda]);
        b1 = creal(B[i + (n - l + i) * ldb]);
        gamma = b1 / a1;

        if (gamma <= HUGENUM && gamma >= -HUGENUM) {
            if (gamma < ZERO) {
                cblas_zdscal(l - i, -ONE, &B[i + (n - l + i) * ldb], ldb);
                if (wantv) {
                    cblas_zdscal(p, -ONE, &V[i * ldv], 1);
                }
            }

            dlartg(fabs(gamma), ONE, &beta[k + i], &alpha[k + i], &rwk);

            if (alpha[k + i] >= beta[k + i]) {
                cblas_zdscal(l - i, ONE / alpha[k + i],
                             &A[(k + i) + (n - l + i) * lda], lda);
            } else {
                cblas_zdscal(l - i, ONE / beta[k + i],
                             &B[i + (n - l + i) * ldb], ldb);
                cblas_zcopy(l - i, &B[i + (n - l + i) * ldb], ldb,
                            &A[(k + i) + (n - l + i) * lda], lda);
            }
        } else {
            alpha[k + i] = ZERO;
            beta[k + i] = ONE;
            cblas_zcopy(l - i, &B[i + (n - l + i) * ldb], ldb,
                        &A[(k + i) + (n - l + i) * lda], lda);
        }
    }

    for (i = m; i < k + l; i++) {
        alpha[i] = ZERO;
        beta[i] = ONE;
    }

    if (k + l < n) {
        for (i = k + l; i < n; i++) {
            alpha[i] = ZERO;
            beta[i] = ZERO;
        }
    }

finish:
    *ncycle = kcycle + 1;
}
