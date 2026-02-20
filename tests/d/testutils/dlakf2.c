/**
 * @file dlakf2.c
 * @brief DLAKF2 forms a 2*M*N by 2*M*N matrix using Kronecker products.
 */

#include "verify.h"

/* Forward declarations */
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);

/**
 * Form the 2*M*N by 2*M*N matrix
 *
 *        Z = [ kron(In, A)  -kron(B', Im) ]
 *            [ kron(In, D)  -kron(E', Im) ],
 *
 * where In is the identity matrix of size n and X' is the transpose
 * of X. kron(X, Y) is the Kronecker product between the matrices X
 * and Y.
 *
 * @param[in] m
 *     Size of matrix, must be >= 1.
 *
 * @param[in] n
 *     Size of matrix, must be >= 1.
 *
 * @param[in] A
 *     Double precision array, dimension (lda, m).
 *     The matrix A in the output matrix Z.
 *
 * @param[in] lda
 *     The leading dimension of A, B, D, and E. (lda >= m+n)
 *
 * @param[in] B
 *     Double precision array, dimension (lda, n).
 *
 * @param[in] D
 *     Double precision array, dimension (lda, m).
 *
 * @param[in] E
 *     Double precision array, dimension (lda, n).
 *     The matrices used in forming the output matrix Z.
 *
 * @param[out] Z
 *     Double precision array, dimension (ldz, 2*m*n).
 *     The resultant Kronecker M*N*2 by M*N*2 matrix (see above.)
 *
 * @param[in] ldz
 *     The leading dimension of Z. (ldz >= 2*m*n)
 */
void dlakf2(const int m, const int n,
            const f64* A, const int lda,
            const f64* B, const f64* D, const f64* E,
            f64* Z, const int ldz)
{
    const f64 ZERO = 0.0;

    int i, ik, j, jk, l, mn, mn2;

    mn = m * n;
    mn2 = 2 * mn;
    dlaset("Full", mn2, mn2, ZERO, ZERO, Z, ldz);

    ik = 0;
    for (l = 0; l < n; l++) {

        /* form kron(In, A) */
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                Z[(ik + i) + (ik + j) * ldz] = A[i + j * lda];
            }
        }

        /* form kron(In, D) */
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                Z[(ik + mn + i) + (ik + j) * ldz] = D[i + j * lda];
            }
        }

        ik = ik + m;
    }

    ik = 0;
    for (l = 0; l < n; l++) {
        jk = mn;

        for (j = 0; j < n; j++) {

            /* form -kron(B', Im) */
            for (i = 0; i < m; i++) {
                Z[(ik + i) + (jk + i) * ldz] = -B[j + l * lda];
            }

            /* form -kron(E', Im) */
            for (i = 0; i < m; i++) {
                Z[(ik + mn + i) + (jk + i) * ldz] = -E[j + l * lda];
            }

            jk = jk + m;
        }

        ik = ik + m;
    }
}
