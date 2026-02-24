/**
 * @file zlakf2.c
 * @brief ZLAKF2 forms a 2*M*N by 2*M*N matrix using Kronecker products.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlakf2.f
 */

#include "verify.h"

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
 *     Complex array, dimension (lda, m).
 *
 * @param[in] lda
 *     The leading dimension of A, B, D, and E. (lda >= m+n)
 *
 * @param[in] B
 *     Complex array, dimension (lda, n).
 *
 * @param[in] D
 *     Complex array, dimension (lda, m).
 *
 * @param[in] E
 *     Complex array, dimension (lda, n).
 *
 * @param[out] Z
 *     Complex array, dimension (ldz, 2*m*n).
 *     The resultant Kronecker M*N*2 by M*N*2 matrix.
 *
 * @param[in] ldz
 *     The leading dimension of Z. (ldz >= 2*m*n)
 */
void zlakf2(const INT m, const INT n,
            const c128* A, const INT lda,
            const c128* B, const c128* D, const c128* E,
            c128* Z, const INT ldz)
{
    const c128 ZERO = CMPLX(0.0, 0.0);

    INT i, ik, j, jk, l, mn, mn2;

    mn = m * n;
    mn2 = 2 * mn;
    zlaset("Full", mn2, mn2, ZERO, ZERO, Z, ldz);

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
