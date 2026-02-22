/**
 * @file zlarft_lvl2.c
 * @brief ZLARFT_LVL2 Level 2 BLAS version for terminating case of ZLARFT.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFT_LVL2 forms the triangular factor T of a complex block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
 *
 * If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
 * If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
 *
 * If STOREV = 'C', the vector which defines the elementary reflector
 * H(i) is stored in the i-th column of the array V, and
 *    H = I - V * T * V**H
 *
 * If STOREV = 'R', the vector which defines the elementary reflector
 * H(i) is stored in the i-th row of the array V, and
 *    H = I - V**H * T * V
 *
 * @param[in]  direct  'F': Forward; 'B': Backward.
 * @param[in]  storev  'C': Columnwise; 'R': Rowwise.
 * @param[in]  n       The order of the block reflector H. n >= 0.
 * @param[in]  k       The order of T (number of reflectors). k >= 1.
 * @param[in]  V       The matrix V containing reflector vectors.
 *                     Dimension (ldv, k) if storev='C', (ldv, n) if storev='R'.
 * @param[in]  ldv     Leading dimension of V.
 * @param[in]  tau     Array of dimension k. Scalar factors of reflectors.
 * @param[out] T       The k-by-k triangular factor T. Dimension (ldt, k).
 * @param[in]  ldt     Leading dimension of T. ldt >= k.
 */
void zlarft_lvl2(const char* direct, const char* storev,
                 const INT n, const INT k,
                 const c128* restrict V, const INT ldv,
                 const c128* restrict tau,
                 c128* restrict T, const INT ldt)
{
    const c128 ONE  = 1.0;
    const c128 ZERO = 0.0;

    INT i, j, prevlastv, lastv;

    if (n == 0)
        return;

    if (direct[0] == 'F' || direct[0] == 'f') {
        prevlastv = n;
        for (i = 1; i <= k; i++) {
            prevlastv = (prevlastv > i) ? prevlastv : i;
            if (tau[i - 1] == ZERO) {

                for (j = 1; j <= i; j++) {
                    T[(j - 1) + (i - 1) * ldt] = ZERO;
                }
            } else {

                if (storev[0] == 'C' || storev[0] == 'c') {
                    for (lastv = n; lastv >= i + 1; lastv--) {
                        if (V[(lastv - 1) + (i - 1) * ldv] != ZERO) break;
                    }
                    for (j = 1; j <= i - 1; j++) {
                        T[(j - 1) + (i - 1) * ldt] = -tau[i - 1] * conj(V[(i - 1) + (j - 1) * ldv]);
                    }
                    j = (lastv < prevlastv) ? lastv : prevlastv;

                    c128 neg_tau_i = -tau[i - 1];
                    cblas_zgemv(CblasColMajor, CblasConjTrans, j - i, i - 1,
                                &neg_tau_i, &V[i + 0 * ldv], ldv,
                                &V[i + (i - 1) * ldv], 1,
                                &ONE, &T[0 + (i - 1) * ldt], 1);
                } else {
                    for (lastv = n; lastv >= i + 1; lastv--) {
                        if (V[(i - 1) + (lastv - 1) * ldv] != ZERO) break;
                    }
                    for (j = 1; j <= i - 1; j++) {
                        T[(j - 1) + (i - 1) * ldt] = -tau[i - 1] * V[(j - 1) + (i - 1) * ldv];
                    }
                    j = (lastv < prevlastv) ? lastv : prevlastv;

                    c128 neg_tau_i = -tau[i - 1];
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                i - 1, 1, j - i, &neg_tau_i,
                                &V[0 + i * ldv], ldv,
                                &V[(i - 1) + i * ldv], ldv,
                                &ONE, &T[0 + (i - 1) * ldt], ldt);
                }

                cblas_ztrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                            i - 1, T, ldt, &T[0 + (i - 1) * ldt], 1);
                T[(i - 1) + (i - 1) * ldt] = tau[i - 1];
                if (i > 1) {
                    prevlastv = (prevlastv > lastv) ? prevlastv : lastv;
                } else {
                    prevlastv = lastv;
                }
            }
        }
    } else {
        prevlastv = 1;
        for (i = k; i >= 1; i--) {
            if (tau[i - 1] == ZERO) {

                for (j = i; j <= k; j++) {
                    T[(j - 1) + (i - 1) * ldt] = ZERO;
                }
            } else {

                if (i < k) {
                    if (storev[0] == 'C' || storev[0] == 'c') {
                        for (lastv = 1; lastv <= i - 1; lastv++) {
                            if (V[(lastv - 1) + (i - 1) * ldv] != ZERO) break;
                        }
                        for (j = i + 1; j <= k; j++) {
                            T[(j - 1) + (i - 1) * ldt] = -tau[i - 1] * conj(V[(n - k + i - 1) + (j - 1) * ldv]);
                        }
                        j = (lastv > prevlastv) ? lastv : prevlastv;

                        c128 neg_tau_i = -tau[i - 1];
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    n - k + i - j, k - i,
                                    &neg_tau_i, &V[(j - 1) + i * ldv], ldv,
                                    &V[(j - 1) + (i - 1) * ldv], 1,
                                    &ONE, &T[i + (i - 1) * ldt], 1);
                    } else {
                        for (lastv = 1; lastv <= i - 1; lastv++) {
                            if (V[(i - 1) + (lastv - 1) * ldv] != ZERO) break;
                        }
                        for (j = i + 1; j <= k; j++) {
                            T[(j - 1) + (i - 1) * ldt] = -tau[i - 1] * V[(j - 1) + (n - k + i - 1) * ldv];
                        }
                        j = (lastv > prevlastv) ? lastv : prevlastv;

                        c128 neg_tau_i = -tau[i - 1];
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    k - i, 1, n - k + i - j,
                                    &neg_tau_i,
                                    &V[i + (j - 1) * ldv], ldv,
                                    &V[(i - 1) + (j - 1) * ldv], ldv,
                                    &ONE, &T[i + (i - 1) * ldt], ldt);
                    }

                    cblas_ztrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                                k - i, &T[i + i * ldt], ldt,
                                &T[i + (i - 1) * ldt], 1);
                    if (i > 1) {
                        prevlastv = (prevlastv < lastv) ? prevlastv : lastv;
                    } else {
                        prevlastv = lastv;
                    }
                }
                T[(i - 1) + (i - 1) * ldt] = tau[i - 1];
            }
        }
    }
}
