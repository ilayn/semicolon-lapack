/**
 * @file slarzt.c
 * @brief SLARZT forms the triangular factor T of a block reflector H = I - V**T*T*V.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLARZT forms the triangular factor T of a real block reflector
 * H of order > n, which is defined as a product of k elementary
 * reflectors.
 *
 * If DIRECT = "F", H = H(1) H(2) . . . H(k) and T is upper triangular;
 * If DIRECT = "B", H = H(k) . . . H(2) H(1) and T is lower triangular.
 *
 * If STOREV = "C", the vector which defines the elementary reflector
 * H(i) is stored in the i-th column of the array V, and
 *    H = I - V * T * V**T
 *
 * If STOREV = "R", the vector which defines the elementary reflector
 * H(i) is stored in the i-th row of the array V, and
 *    H = I - V**T * T * V
 *
 * Currently, only STOREV = 'R' and DIRECT = 'B' are supported.
 *
 * @param[in]     direct  Specifies the order in which the elementary reflectors
 *                        are multiplied to form the block reflector:
 *                        = 'F': H = H(1) H(2) . . . H(k) (Forward, not supported yet)
 *                        = 'B': H = H(k) . . . H(2) H(1) (Backward)
 * @param[in]     storev  Specifies how the vectors which define the elementary
 *                        reflectors are stored:
 *                        = 'C': columnwise (not supported yet)
 *                        = 'R': rowwise
 * @param[in]     n       The order of the block reflector H. n >= 0.
 * @param[in]     k       The order of the triangular factor T (= the number of
 *                        elementary reflectors). k >= 1.
 * @param[in,out] V       Double precision array, dimension (ldv, n).
 *                        The matrix V. See Further Details in LAPACK documentation.
 * @param[in]     ldv     The leading dimension of the array V. ldv >= k.
 * @param[in]     tau     Double precision array, dimension (k).
 *                        tau[i] must contain the scalar factor of the elementary
 *                        reflector H(i).
 * @param[out]    T       Double precision array, dimension (ldt, k).
 *                        The k by k triangular factor T of the block reflector.
 *                        T is lower triangular (since only DIRECT = 'B' is supported).
 *                        The rest of the array is not used.
 * @param[in]     ldt     The leading dimension of the array T. ldt >= k.
 */
void slarzt(const char* direct, const char* storev,
            const INT n, const INT k,
            f32* restrict V, const INT ldv,
            const f32* restrict tau,
            f32* restrict T, const INT ldt)
{
    const f32 ZERO = 0.0f;
    INT i, j, info;

    /* Check for currently supported options */
    info = 0;
    if (direct[0] != 'B' && direct[0] != 'b') {
        info = -1;
    } else if (storev[0] != 'R' && storev[0] != 'r') {
        info = -2;
    }
    if (info != 0) {
        xerbla("SLARZT", -info);
        return;
    }

    for (i = k - 1; i >= 0; i--) {
        if (tau[i] == ZERO) {

            /* H(i) = I */
            for (j = i; j < k; j++) {
                T[j + i * ldt] = ZERO;
            }
        } else {

            /* General case */
            if (i < k - 1) {

                /* T(i+1:k-1, i) = -tau(i) * V(i+1:k-1, 0:n-1) * V(i, 0:n-1)^T */
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            k - 1 - i, n, -tau[i],
                            &V[(i + 1) + 0 * ldv], ldv,
                            &V[i + 0 * ldv], ldv,
                            ZERO, &T[(i + 1) + i * ldt], 1);

                /* T(i+1:k-1, i) = T(i+1:k-1, i+1:k-1) * T(i+1:k-1, i) */
                cblas_strmv(CblasColMajor, CblasLower, CblasNoTrans,
                            CblasNonUnit, k - 1 - i,
                            &T[(i + 1) + (i + 1) * ldt], ldt,
                            &T[(i + 1) + i * ldt], 1);
            }
            T[i + i * ldt] = tau[i];
        }
    }
}
