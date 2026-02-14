/**
 * @file dlarft.c
 * @brief DLARFT forms the triangular factor T of a block reflector H = I - V*T*V**T.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARFT forms the triangular factor T of a real block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
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
 * Uses the recursive Elmroth-Gustavson algorithm.
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
void dlarft(const char* direct, const char* storev,
            const int n, const int k,
            const f64 * const restrict V, const int ldv,
            const f64 * const restrict tau,
            f64 * const restrict T, const int ldt)
{
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;
    int l, i, j;
    int dirf, colv, qr, lq, ql;

    /* Quick return if possible */
    if (n == 0 || k == 0) {
        return;
    }

    /* Base case */
    if (n == 1 || k == 1) {
        T[0] = tau[0];
        return;
    }

    /* For small K, use level-2 implementation directly.
     * ILAENV(3, 'DLARFT', ...) has no special case in ilaenv.f,
     * so NX defaults to 0 — meaning always recurse. But we still
     * need the level-2 code as the recursion base. Use a fixed
     * crossover of 0 (always recurse until k=1). */

    l = k / 2;

    /* Determine factorization type */
    dirf = (direct[0] == 'F' || direct[0] == 'f');
    colv = (storev[0] == 'C' || storev[0] == 'c');

    qr = dirf && colv;    /* Forward, Columnwise */
    lq = dirf && !colv;   /* Forward, Rowwise */
    ql = !dirf && colv;   /* Backward, Columnwise */
    /* rq = !dirf && !colv; -- Backward, Rowwise (else case) */

    if (qr) {
        /* QR case: Forward, Columnwise
         * V is unit lower triangular in first k rows, rectangular below.
         * T is upper triangular. */

        /* Compute T_{1,1} recursively (first l reflectors) */
        dlarft(direct, storev, n, l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively (remaining k-l reflectors) */
        dlarft(direct, storev, n - l, k - l,
               &V[(l) + (l) * ldv], ldv,
               &tau[l], &T[(l) + (l) * ldt], ldt);

        /* Compute T_{1,2} = V_{2,1}^T (copy transposed block) */
        for (j = 0; j < l; j++) {
            for (i = 0; i < k - l; i++) {
                T[j + (l + i) * ldt] = V[(l + i) + j * ldv];
            }
        }

        /* T_{1,2} = T_{1,2} * V_{2,2} (unit lower triangular) */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                    CblasNoTrans, CblasUnit,
                    l, k - l, ONE,
                    &V[(l) + (l) * ldv], ldv,
                    &T[0 + (l) * ldt], ldt);

        /* T_{1,2} += V_{3,1}^T * V_{3,2} (if n > k) */
        if (n > k) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        l, k - l, n - k, ONE,
                        &V[(k) + 0 * ldv], ldv,
                        &V[(k) + (l) * ldv], ldv,
                        ONE, &T[0 + (l) * ldt], ldt);
        }

        /* T_{1,2} = -T_{1,1} * T_{1,2} */
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, NEG_ONE,
                    T, ldt, &T[0 + (l) * ldt], ldt);

        /* T_{1,2} = T_{1,2} * T_{2,2} */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, ONE,
                    &T[(l) + (l) * ldt], ldt,
                    &T[0 + (l) * ldt], ldt);

    } else if (lq) {
        /* LQ case: Forward, Rowwise
         * V is unit upper triangular in first k columns, rectangular right.
         * T is upper triangular. */

        /* Compute T_{1,1} recursively */
        dlarft(direct, storev, n, l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        dlarft(direct, storev, n - l, k - l,
               &V[(l) + (l) * ldv], ldv,
               &tau[l], &T[(l) + (l) * ldt], ldt);

        /* T_{1,2} = V_{1,2} (copy) */
        dlacpy("A", l, k - l, &V[0 + (l) * ldv], ldv, &T[0 + (l) * ldt], ldt);

        /* T_{1,2} = T_{1,2} * V_{2,2}^T (unit upper triangular) */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                    CblasTrans, CblasUnit,
                    l, k - l, ONE,
                    &V[(l) + (l) * ldv], ldv,
                    &T[0 + (l) * ldt], ldt);

        /* T_{1,2} += V_{1,3} * V_{2,3}^T (if n > k) */
        if (n > k) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        l, k - l, n - k, ONE,
                        &V[0 + (k) * ldv], ldv,
                        &V[(l) + (k) * ldv], ldv,
                        ONE, &T[0 + (l) * ldt], ldt);
        }

        /* T_{1,2} = -T_{1,1} * T_{1,2} */
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, NEG_ONE,
                    T, ldt, &T[0 + (l) * ldt], ldt);

        /* T_{1,2} = T_{1,2} * T_{2,2} */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, ONE,
                    &T[(l) + (l) * ldt], ldt,
                    &T[0 + (l) * ldt], ldt);

    } else if (ql) {
        /* QL case: Backward, Columnwise
         * V is unit upper triangular in last k rows, rectangular above.
         * T is lower triangular. */

        /* Compute T_{1,1} recursively (first k-l reflectors) */
        dlarft(direct, storev, n - l, k - l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively (last l reflectors) */
        dlarft(direct, storev, n, l,
               &V[0 + (k - l) * ldv], ldv,
               &tau[k - l], &T[(k - l) + (k - l) * ldt], ldt);

        /* T_{2,1} = V_{2,2}^T (copy transposed block) */
        for (j = 0; j < k - l; j++) {
            for (i = 0; i < l; i++) {
                T[(k - l + i) + j * ldt] = V[(n - k + j) + (k - l + i) * ldv];
            }
        }

        /* T_{2,1} = T_{2,1} * V_{2,1} (unit upper triangular) */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                    CblasNoTrans, CblasUnit,
                    l, k - l, ONE,
                    &V[(n - k) + 0 * ldv], ldv,
                    &T[(k - l) + 0 * ldt], ldt);

        /* T_{2,1} += V_{1,2}^T * V_{1,1} (if n > k) */
        if (n > k) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        l, k - l, n - k, ONE,
                        &V[0 + (k - l) * ldv], ldv,
                        V, ldv,
                        ONE, &T[(k - l) + 0 * ldt], ldt);
        }

        /* T_{2,1} = -T_{2,2} * T_{2,1} */
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, NEG_ONE,
                    &T[(k - l) + (k - l) * ldt], ldt,
                    &T[(k - l) + 0 * ldt], ldt);

        /* T_{2,1} = T_{2,1} * T_{1,1} */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, ONE,
                    T, ldt, &T[(k - l) + 0 * ldt], ldt);

    } else {
        /* RQ case: Backward, Rowwise
         * V is unit lower triangular in last k columns, rectangular left.
         * T is lower triangular. */

        /* Compute T_{1,1} recursively (first k-l reflectors) */
        dlarft(direct, storev, n - l, k - l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively (last l reflectors) */
        dlarft(direct, storev, n, l,
               &V[(k - l) + 0 * ldv], ldv,
               &tau[k - l], &T[(k - l) + (k - l) * ldt], ldt);

        /* T_{2,1} = V_{2,2} (copy) */
        dlacpy("A", l, k - l, &V[(k - l) + (n - k) * ldv], ldv,
               &T[(k - l) + 0 * ldt], ldt);

        /* T_{2,1} = T_{2,1} * V_{1,2}^T (unit lower triangular) */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                    CblasTrans, CblasUnit,
                    l, k - l, ONE,
                    &V[0 + (n - k) * ldv], ldv,
                    &T[(k - l) + 0 * ldt], ldt);

        /* T_{2,1} += V_{2,1} * V_{1,1}^T (if n > k) */
        if (n > k) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        l, k - l, n - k, ONE,
                        &V[(k - l) + 0 * ldv], ldv,
                        V, ldv,
                        ONE, &T[(k - l) + 0 * ldt], ldt);
        }

        /* T_{2,1} = -T_{2,2} * T_{2,1} */
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, NEG_ONE,
                    &T[(k - l) + (k - l) * ldt], ldt,
                    &T[(k - l) + 0 * ldt], ldt);

        /* T_{2,1} = T_{2,1} * T_{1,1} */
        cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    l, k - l, ONE,
                    T, ldt, &T[(k - l) + 0 * ldt], ldt);
    }
}
