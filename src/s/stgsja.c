/**
 * @file stgsja.c
 * @brief STGSJA computes the GSVD of two upper triangular matrices.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

#define MAXIT 40

/**
 * STGSJA computes the generalized singular value decomposition (GSVD)
 * of two real upper triangular (or trapezoidal) matrices A and B.
 *
 * On entry, it is assumed that matrices A and B have the following
 * forms, which may be obtained by the preprocessing subroutine DGGSVP
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
 *        U**T *A*Q = D1*( 0 R ),    V**T *B*Q = D2*( 0 R ),
 *
 * where U, V and Q are orthogonal matrices.
 *
 * @param[in]     jobu    = 'U': U must contain an orthogonal matrix U1 on entry;
 *                        = 'I': U is initialized to the unit matrix;
 *                        = 'N': U is not computed.
 * @param[in]     jobv    = 'V': V must contain an orthogonal matrix V1 on entry;
 *                        = 'I': V is initialized to the unit matrix;
 *                        = 'N': V is not computed.
 * @param[in]     jobq    = 'Q': Q must contain an orthogonal matrix Q1 on entry;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'N': Q is not computed.
 * @param[in]     m       The number of rows of matrix A. m >= 0.
 * @param[in]     p       The number of rows of matrix B. p >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in]     k       Subblock dimension from SGGSVP3.
 * @param[in]     l       Subblock dimension from SGGSVP3.
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
 * @param[in,out] U       Orthogonal matrix U (dimension ldu,m).
 * @param[in]     ldu     Leading dimension of U.
 * @param[in,out] V       Orthogonal matrix V (dimension ldv,p).
 * @param[in]     ldv     Leading dimension of V.
 * @param[in,out] Q       Orthogonal matrix Q (dimension ldq,n).
 * @param[in]     ldq     Leading dimension of Q.
 * @param[out]    work    Workspace of dimension 2*n.
 * @param[out]    ncycle  Number of cycles for convergence.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: the procedure does not converge after MAXIT cycles.
 */
void stgsja(const char* jobu, const char* jobv, const char* jobq,
            const int m, const int p, const int n, const int k, const int l,
            f32* restrict A, const int lda,
            f32* restrict B, const int ldb,
            const f32 tola, const f32 tolb,
            f32* restrict alpha, f32* restrict beta,
            f32* restrict U, const int ldu,
            f32* restrict V, const int ldv,
            f32* restrict Q, const int ldq,
            f32* restrict work, int* ncycle, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 HUGENUM = FLT_MAX;

    int initu, wantu, initv, wantv, initq, wantq;
    int upper;
    int i, j, kcycle;
    f32 a1, a2, a3, b1, b2, b3;
    f32 csu, snu, csv, snv, csq, snq;
    f32 error, gamma, rwk, ssmin;
    int minval;

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
        xerbla("STGSJA", -(*info));
        return;
    }

    if (initu) {
        slaset("F", m, m, ZERO, ONE, U, ldu);
    }
    if (initv) {
        slaset("F", p, p, ZERO, ONE, V, ldv);
    }
    if (initq) {
        slaset("F", n, n, ZERO, ONE, Q, ldq);
    }

    upper = 0;
    for (kcycle = 0; kcycle < MAXIT; kcycle++) {
        upper = !upper;

        for (i = 0; i < l - 1; i++) {
            for (j = i + 1; j < l; j++) {
                a1 = ZERO;
                a2 = ZERO;
                a3 = ZERO;
                if (k + i < m) {
                    a1 = A[(k + i) + (n - l + i) * lda];
                }
                if (k + j < m) {
                    a3 = A[(k + j) + (n - l + j) * lda];
                }

                b1 = B[i + (n - l + i) * ldb];
                b3 = B[j + (n - l + j) * ldb];

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

                slags2(upper, a1, a2, a3, b1, b2, b3,
                       &csu, &snu, &csv, &snv, &csq, &snq);

                if (k + j < m) {
                    cblas_srot(l, &A[(k + j) + (n - l) * lda], lda,
                               &A[(k + i) + (n - l) * lda], lda, csu, snu);
                }

                cblas_srot(l, &B[j + (n - l) * ldb], ldb,
                           &B[i + (n - l) * ldb], ldb, csv, snv);

                minval = k + l;
                if (m < minval) minval = m;
                cblas_srot(minval, &A[(n - l + j) * lda], 1,
                           &A[(n - l + i) * lda], 1, csq, snq);

                cblas_srot(l, &B[(n - l + j) * ldb], 1,
                           &B[(n - l + i) * ldb], 1, csq, snq);

                if (upper) {
                    if (k + i < m) {
                        A[(k + i) + (n - l + j) * lda] = ZERO;
                    }
                    B[i + (n - l + j) * ldb] = ZERO;
                } else {
                    if (k + j < m) {
                        A[(k + j) + (n - l + i) * lda] = ZERO;
                    }
                    B[j + (n - l + i) * ldb] = ZERO;
                }

                if (wantu && k + j < m) {
                    cblas_srot(m, &U[(k + j) * ldu], 1,
                               &U[(k + i) * ldu], 1, csu, snu);
                }

                if (wantv) {
                    cblas_srot(p, &V[j * ldv], 1,
                               &V[i * ldv], 1, csv, snv);
                }

                if (wantq) {
                    cblas_srot(n, &Q[(n - l + j) * ldq], 1,
                               &Q[(n - l + i) * ldq], 1, csq, snq);
                }
            }
        }

        if (!upper) {
            error = ZERO;
            minval = l;
            if (m - k < minval) minval = m - k;
            for (i = 0; i < minval; i++) {
                cblas_scopy(l - i, &A[(k + i) + (n - l + i) * lda], lda, work, 1);
                cblas_scopy(l - i, &B[i + (n - l + i) * ldb], ldb, &work[l], 1);
                slapll(l - i, work, 1, &work[l], 1, &ssmin);
                if (ssmin > error) error = ssmin;
            }

            if (fabsf(error) <= (tola < tolb ? tola : tolb)) {
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
        a1 = A[(k + i) + (n - l + i) * lda];
        b1 = B[i + (n - l + i) * ldb];
        gamma = b1 / a1;

        if (gamma <= HUGENUM && gamma >= -HUGENUM) {
            if (gamma < ZERO) {
                cblas_sscal(l - i, -ONE, &B[i + (n - l + i) * ldb], ldb);
                if (wantv) {
                    cblas_sscal(p, -ONE, &V[i * ldv], 1);
                }
            }

            slartg(fabsf(gamma), ONE, &beta[k + i], &alpha[k + i], &rwk);

            if (alpha[k + i] >= beta[k + i]) {
                cblas_sscal(l - i, ONE / alpha[k + i],
                            &A[(k + i) + (n - l + i) * lda], lda);
            } else {
                cblas_sscal(l - i, ONE / beta[k + i],
                            &B[i + (n - l + i) * ldb], ldb);
                cblas_scopy(l - i, &B[i + (n - l + i) * ldb], ldb,
                            &A[(k + i) + (n - l + i) * lda], lda);
            }
        } else {
            alpha[k + i] = ZERO;
            beta[k + i] = ONE;
            cblas_scopy(l - i, &B[i + (n - l + i) * ldb], ldb,
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
