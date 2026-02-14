/**
 * @file ztgexc.c
 * @brief ZTGEXC reorders the generalized Schur decomposition of a complex matrix pair.
 */

#include "semicolon_lapack_complex_double.h"

/**
 * ZTGEXC reorders the generalized Schur decomposition of a complex
 * matrix pair (A,B), using an unitary equivalence transformation
 * (A, B) := Q * (A, B) * Z**H, so that the diagonal block of (A, B) with
 * row index IFST is moved to row ILST.
 *
 * (A, B) must be in generalized Schur canonical form, that is, A and
 * B are both upper triangular.
 *
 * Optionally, the matrices Q and Z of generalized Schur vectors are
 * updated.
 *
 *        Q(in) * A(in) * Z(in)**H = Q(out) * A(out) * Z(out)**H
 *        Q(in) * B(in) * Z(in)**H = Q(out) * B(out) * Z(out)**H
 *
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Complex array of dimension (lda, n). Upper triangular.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Complex array of dimension (ldb, n). Upper triangular.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q       Complex array of dimension (ldq, n). The unitary matrix Q.
 *                        Not referenced if wantq = 0.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Complex array of dimension (ldz, n). The unitary matrix Z.
 *                        Not referenced if wantz = 0.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[in]     ifst    Index of block to move (0-based).
 * @param[in,out] ilst    On entry, target index (0-based). On exit, actual position.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: swap failed, matrices partially reordered,
 *                                ilst points to current position of block.
 */
void ztgexc(
    const int wantq,
    const int wantz,
    const int n,
    double complex* const restrict A,
    const int lda,
    double complex* const restrict B,
    const int ldb,
    double complex* const restrict Q,
    const int ldq,
    double complex* const restrict Z,
    const int ldz,
    const int ifst,
    int* ilst,
    int* info)
{
    int here;

    *info = 0;
    if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldq < 1 || (wantq && ldq < (1 > n ? 1 : n))) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < (1 > n ? 1 : n))) {
        *info = -11;
    } else if (ifst < 0 || ifst >= n) {
        *info = -12;
    } else if (*ilst < 0 || *ilst >= n) {
        *info = -13;
    }
    if (*info != 0) {
        xerbla("ZTGEXC", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 1) {
        return;
    }
    if (ifst == *ilst) {
        return;
    }

    if (ifst < *ilst) {

        here = ifst;

        while (1) {

            /* Swap with next one below */
            ztgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                   here, info);
            if (*info != 0) {
                *ilst = here;
                return;
            }
            here = here + 1;
            if (here >= *ilst) {
                break;
            }
        }
        here = here - 1;
    } else {
        here = ifst - 1;

        while (1) {

            /* Swap with next one above */
            ztgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                   here, info);
            if (*info != 0) {
                *ilst = here;
                return;
            }
            here = here - 1;
            if (here < *ilst) {
                break;
            }
        }
        here = here + 1;
    }
    *ilst = here;
}
