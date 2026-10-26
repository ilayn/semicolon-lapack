/**
 * @file slarft.c
 * @brief SLARFT forms the triangular factor T of a block reflector H = I - V*T*V**T.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SLARFT forms the triangular factor T of a real block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
 *
 * If `direct='F'`, H = H(0) H(1) . . . H(k-1) and T is upper triangular;
 *
 * If `direct='B'`, H = H(k-1) . . . H(1) H(0) and T is lower triangular.
 *
 * If `storev='C'`, the vector which defines the elementary reflector
 * H(i) is stored in column `i` of the array `V`, and
 *
 *    H  =  I - V * T * V**T
 *
 * If `storev='R'`, the vector which defines the elementary reflector
 * H(i) is stored in row `i` of the array `V`, and
 *
 *    H  =  I - V**T * T * V
 *
 * If `direct` or `storev` = `'T'`, see Further Details for the shape of T
 *
 * @param[in]  direct  Specifies the order in which the elementary reflectors are
 *                     multiplied to form the block reflector:
 *                     - `'F'`: H = H(0) H(1) . . . H(k-1) (Forward)
 *                     - `'B'`: H = H(k-1) . . . H(1) H(0) (Backward)
 *                     - `'T'` (with `storev='R'`): H = H(k-1) . . . H(1) H(0)
 *                       (backward) but we return the T matrix that is already
 *                       (conjugate) transposed
 * @param[in]  storev  Specifies how the vectors which define the elementary
 *                     reflectors are stored (see also Further Details):
 *                     - `'C'`: column-wise
 *                     - `'R'`: row-wise
 *                     - `'T'`: (With `direct='F'`) Row-wise, but we return the T
 *                       matrix that is already (conjugate) transposed.
 * @param[in]  n       The order of the block reflector H. `n>=0`.
 * @param[in]  k       The order of the triangular factor T (= the number of
 *                     elementary reflectors). `k>=1`.
 * @param[in]  V       Array of dimension (`ldv`, `k`) if `storev='C'`,
 *                     (`ldv`, `n`) if `storev='R'`.
 *                     The matrix V. See further details.
 * @param[in]  ldv     The leading dimension of the array `V`.
 *                     If `storev='C'`, `ldv>=max(1,n)`; if `storev='R'`, `ldv>=k`.
 * @param[in]  tau     Array of dimension (`k`).
 *                     `tau[i]` must contain the scalar factor of the elementary
 *                     reflector H(i).
 * @param[out] T       Array of dimension (`ldt`, `k`).
 *                     The k by k triangular factor T of the block reflector.
 *                     If `direct='F'`, T is upper triangular; if `direct='B'`, T is
 *                     lower triangular. The rest of the array is not used.
 * @param[in]  ldt     The leading dimension of the array `T`. `ldt>=k`.
 *
 * @par Further Details:
 * @rst
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored.
 *
 * .. code-block:: text
 *
 *     DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and (STOREV = 'R' or STOREV = 'T'):
 *
 *                  V = (  1       )                 V = (  1 v0 v0 v0 v0 )
 *                      ( v0  1    )                     (     1 v1 v1 v1 )
 *                      ( v0 v1  1 )                     (        1 v2 v2 )
 *                      ( v0 v1 v2 )
 *                      ( v0 v1 v2 )
 *
 *     DIRECT = 'B' and STOREV = 'C':         (DIRECT = 'B' or DIRECT = 'T') and STOREV = 'R':
 *
 *                  V = ( v0 v1 v2 )                 V = ( v0 v0  1       )
 *                      ( v0 v1 v2 )                     ( v1 v1 v1  1    )
 *                      (  1 v1 v2 )                     ( v2 v2 v2 v2  1 )
 *                      (     1 v2 )
 *                      (        1 )
 *
 * In addition, the shape of T is determined by these same flags as
 * in the below table.
 * 'U' denotes upper triangular
 * 'L' denotes lower triangular
 * 'X' denotes no current implementation
 * We also provide the logical variable that represents the case
 * in the code if it is implemented
 *
 * .. code-block:: text
 *
 *     |-----------------------------------------------------------|
 *     |              | DIRECT = 'F' | DIRECT = 'B' | DIRECT = 'T' |
 *     |--------------+--------------+--------------+--------------|
 *     | STOREV = 'C' | U (QR)       | L (QL)       | X            |
 *     | STOREV = 'R' | U (LQ)       | L (RQ)       | U (RQT)      |
 *     | STOREV = 'T' | L (LQT)      | X            | X            |
 *     |-----------------------------------------------------------|
 *
 * Finally, the relationship between the (conjugate) transposed T matrices
 * are as follows: (Note that T_{FC} denotes the T associated with calling
 * this routine with DIRECT = 'F' and STOREV = 'C')
 *
 * .. code-block:: text
 *
 *     T_{TR} = (T_{BR})**H
 *     T_{FT} = (T_{FR})**H
 * @endrst
 */
void slarft(const char* direct, const char* storev,
            const INT n, const INT k,
            const f32* restrict V, const INT ldv,
            const f32* restrict tau,
            f32* restrict T, const INT ldt)
{
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;
    INT i, j, l, nx, info;
    INT qr, lq, ql, rq, lqt, rqt;
    INT dirf, dirb, dirt;
    INT storec, storer, storet;

    /*
     * The general scheme used is inspired by the approach inside SGEQRT3
     * which was (at the time of writing this code):
     * Based on the algorithm of Elmroth and Gustavson,
     * IBM J. Res. Develop. Vol 44 No. 4 July 2000.
     */

    /* Convert our character flags to logical flags for later */
    dirf = (direct[0] == 'F' || direct[0] == 'f');
    dirb = (direct[0] == 'B' || direct[0] == 'b');
    dirt = (direct[0] == 'T' || direct[0] == 't');
    storec = (storev[0] == 'C' || storev[0] == 'c');
    storer = (storev[0] == 'R' || storev[0] == 'r');
    storet = (storev[0] == 'T' || storev[0] == 't');

    /* Error handling for our character flags */
    info = 0;
    if (!(dirf || dirb || dirt)) {

        /* DIRECT holds an illegal value */
        info = 1;
    } else if (!(storec || storer || storet)) {

        /* STOREV holds an illegal value */
        info = 2;
    } else if (dirb && storet) {

        /*
         * This case is purposefully not implemented, but any other value for
         * STOREV is valid, so we report STOREV as the invalid input
         */
        info = 2;
    } else if (dirt && storec) {

        /*
         * This case is purposefully not implemented, but any other value for
         * DIRECT is valid, so we report DIRECT as the invalid input
         */
        info = 1;
    } else if (dirt && storet) {

        /*
         * This case is purposefully not implemented, and is ambiguous what
         * the user wants to do, so we arbitrarily say DIRECT is the incorrect
         * character flag.
         */
        info = 1;
    }

    if (info != 0) {
        xerbla("SLARFT", info);
        return;
    }

    /* Quick return if possible */
    if (n == 0 || k == 0) {
        return;
    }

    /* Now we determine what factorization our flags are associated with */

    /* QR happens when we have forward direction in column storage */
    qr = dirf && storec;

    /*
     * LQT happens when we have forward direction in row storage and want to compute the transpose of
     * the T we would normally compute
     */
    lqt = dirf && storet;

    /*
     * LQ happens when we have forward direction in row storage and want to compute the T we would
     * normally compute
     */
    lq = dirf && storer;

    /* QL happens when we have backward direction in column storage */
    ql = dirb && storec;

    /*
     * RQT happens when we have backward direction in row storage and want to compute the transpose
     * of the T we would normally compute
     */
    rqt = dirt && storer;

    /*
     * RQ happens when we have backward direction in row storage and want to compute the T that we
     * would normally compute
     */
    rq = dirb && storer;

    /* Base case */
    if (n == 1 || k == 1) {
        T[0] = tau[0];
        return;
    }

    /* Determine crossover point from level 2 to level 3 BLAS implementation */
    nx = lapack_get_nx("LARFT");
    if (k < nx) {

        /* Finish this component with a level 2 BLAS implementation */
        slarft_lvl2(direct, storev, n, k, V, ldv, tau, T, ldt);
        return;
    }

    /* Beginning of executable statements for the recursive case */
    l = k / 2;
    if (qr) {

        /*
         * Break V apart into 6 components
         *
         * V = |---------------|
         *     |V_{1,1} 0      |
         *     |V_{2,1} V_{2,2}|
         *     |V_{3,1} V_{3,2}|
         *     |---------------|
         *
         * V_{1,1}\in\R^{l,l}      unit lower triangular
         * V_{2,1}\in\R^{k-l,l}    rectangular
         * V_{3,1}\in\R^{n-k,l}    rectangular
         *
         * V_{2,2}\in\R^{k-l,k-l}  unit lower triangular
         * V_{3,2}\in\R^{n-k,k-l}  rectangular
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} T_{1,2}|
         *     |0       T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{l, l}     upper triangular
         * T_{2,2}\in\R^{k-l, k-l} upper triangular
         * T_{1,2}\in\R^{l, k-l}   rectangular
         *
         * Where l = floor(k/2)
         *
         * Then, consider the product:
         *
         * (I - V_1*T_{1,1}*V_1')*(I - V_2*T_{2,2}*V_2')
         * = I - V_1*T_{1,1}*V_1' - V_2*T_{2,2}*V_2' + V_1*T_{1,1}*V_1'*V_2*T_{2,2}*V_2'
         *
         * Define T_{1,2} = -T_{1,1}*V_1'*V_2*T_{2,2}
         *
         * Then, we can define the matrix V as
         * V = |-------|
         *     |V_1 V_2|
         *     |-------|
         *
         * So, our product is equivalent to the matrix product
         * I - V*T*V'
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{1,2}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n, l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n - l, k - l, &V[l + l * ldv], ldv,
               &tau[l], &T[l + l * ldt], ldt);

        /*
         * Compute T_{1,2}
         * T_{1,2} = V_{2,1}'
         */
        for (j = 0; j < l; j++) {
            for (i = 0; i < k - l; i++) {
                T[j + (l + i) * ldt] = V[(l + i) + j * ldv];
            }
        }

        /* T_{1,2} = T_{1,2}*V_{2,2} */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasUnit, l, k - l, ONE, &V[l + l * ldv], ldv,
                    &T[l * ldt], ldt);

        /*
         * T_{1,2} = V_{3,1}'*V_{3,2} + T_{1,2}
         * Note: We assume K <= N, and GEMM will do nothing if N=K
         */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, l, k - l, n - k,
                    ONE, &V[k], ldv, &V[k + l * ldv], ldv, ONE,
                    &T[l * ldt], ldt);

        /*
         * At this point, we have that T_{1,2} = V_1'*V_2
         * All that is left is to pre and post multiply by -T_{1,1} and T_{2,2}
         * respectively.
         */

        /* T_{1,2} = -T_{1,1}*T_{1,2} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, k - l, NEG_ONE, T, ldt,
                    &T[l * ldt], ldt);

        /* T_{1,2} = T_{1,2}*T_{2,2} */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, k - l, ONE, &T[l + l * ldt], ldt,
                    &T[l * ldt], ldt);

    } else if (lq) {

        /*
         * Break V apart into 6 components
         *
         * V = |----------------------|
         *     |V_{1,1} V_{1,2} V{1,3}|
         *     |0       V_{2,2} V{2,3}|
         *     |----------------------|
         *
         * V_{1,1}\in\R^{l,l}      unit upper triangular
         * V_{1,2}\in\R^{l,k-l}    rectangular
         * V_{1,3}\in\R^{l,n-k}    rectangular
         *
         * V_{2,2}\in\R^{k-l,k-l}  unit upper triangular
         * V_{2,3}\in\R^{k-l,n-k}  rectangular
         *
         * Where l = floor(k/2)
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} T_{1,2}|
         *     |0       T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{l, l}     upper triangular
         * T_{2,2}\in\R^{k-l, k-l} upper triangular
         * T_{1,2}\in\R^{l, k-l}   rectangular
         *
         * Then, consider the product:
         *
         * (I - V_1'*T_{1,1}'*V_1)*(I - V_2'*T_{2,2}'*V_2)
         * = I - V_1'*T_{1,1}'*V_1 - V_2'*T_{2,2}'*V_2 + V_1'*T_{1,1}'*V_1*V_2'*T_{2,2}'*V_2
         *
         * Define T_{1,2} = -T_{1,1}*V_1*V_2'*T_{2,2}
         *
         * Then, we can define the matrix V as
         * V = |---|
         *     |V_1|
         *     |V_2|
         *     |---|
         *
         * So, our product is equivalent to the matrix product
         * I - V'*T'*V
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{1,2}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n, l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n - l, k - l, &V[l + l * ldv], ldv,
               &tau[l], &T[l + l * ldt], ldt);

        /*
         * Compute T_{1,2}
         * T_{1,2} = V_{1,2}
         */
        slacpy("All", l, k - l, &V[l * ldv], ldv, &T[l * ldt], ldt);

        /* T_{1,2} = T_{1,2}*V_{2,2}' */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                    CblasUnit, l, k - l, ONE, &V[l + l * ldv], ldv,
                    &T[l * ldt], ldt);

        /*
         * T_{1,2} = V_{1,3}*V_{2,3}' + T_{1,2}
         * Note: We assume K <= N, and GEMM will do nothing if N=K
         */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, l, k - l, n - k,
                    ONE, &V[k * ldv], ldv, &V[l + k * ldv], ldv, ONE,
                    &T[l * ldt], ldt);

        /*
         * At this point, we have that T_{1,2} = V_1*V_2'
         * All that is left is to pre and post multiply by -T_{1,1} and T_{2,2}
         * respectively.
         */

        /* T_{1,2} = -T_{1,1}*T_{1,2} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, k - l, NEG_ONE, T, ldt,
                    &T[l * ldt], ldt);

        /* T_{1,2} = T_{1,2}*T_{2,2} */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, k - l, ONE, &T[l + l * ldt], ldt,
                    &T[l * ldt], ldt);
    } else if (lqt) {

        /*
         * Break V apart into 6 components
         *
         * V = |----------------------|
         *     |V_{1,1} V_{1,2} V{1,3}|
         *     |0       V_{2,2} V{2,3}|
         *     |----------------------|
         *
         * V_{1,1}\in\R^{l,l}      unit upper triangular
         * V_{1,2}\in\R^{l,k-l}    rectangular
         * V_{1,3}\in\R^{l,n-k}    rectangular
         *
         * V_{2,2}\in\R^{k-l,k-l}  unit upper triangular
         * V_{2,3}\in\R^{k-l,n-k}  rectangular
         *
         * Where l = floor(k/2)
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} 0      |
         *     |T_{2,1} T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{l, l}     lower triangular
         * T_{2,2}\in\R^{k-l, k-l} lower triangular
         * T_{2,1}\in\R^{k-l, l}   rectangular
         *
         * Then, consider the product:
         *
         * (I - V_1'*T_{1,1}*V_1)*(I - V_2'*T_{2,2}*V_2)
         * = I - V_1'*T_{1,1}*V_1 - V_2'*T_{2,2}*V_2 + V_1'*T_{1,1}*V_1*V_2'*T_{2,2}*V_2
         *
         * Define T_{2,1} = -T_{2,2}*V_2*V_1'*T_{1,1}
         *
         * Then, we can define the matrix V as
         * V = |---|
         *     |V_1|
         *     |V_2|
         *     |---|
         *
         * So, our product is equivalent to the matrix product
         * I - V'*T*V
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{2,1}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n, l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n - l, k - l, &V[l + l * ldv], ldv,
               &tau[l], &T[l + l * ldt], ldt);

        /*
         * Compute T_{2,1}
         * T_{2,1} = V_{1,2}'
         */
        for (i = 0; i < k - l; i++) {
            for (j = 0; j < l; j++) {
                T[(l + i) + j * ldt] = V[j + (l + i) * ldv];
            }
        }

        /* T_{2,1} = V_{2,2}*T_{2,1} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasUnit, k - l, l, ONE, &V[l + l * ldv], ldv,
                    &T[l], ldt);

        /*
         * T_{2,1} = V_{2,3}*V_{1,3}' + T_{2,1}
         * Note: We assume K <= N, and GEMM will do nothing if N=K
         */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, k - l, l, n - k,
                    ONE, &V[l + k * ldv], ldv, &V[k * ldv], ldv, ONE,
                    &T[l], ldt);

        /*
         * At this point, we have that T_{2,1} = V_2*V_1'
         * All that is left is to pre and post multiply by -T_{2,2} and T_{1,1}
         * respectively.
         */

        /* T_{2,1} = -T_{2,2}*T_{2,1} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, k - l, l, NEG_ONE, &T[l + l * ldt], ldt,
                    &T[l], ldt);

        /* T_{2,1} = T_{2,1}*T_{1,1} */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, k - l, l, ONE, T, ldt, &T[l], ldt);
    } else if (ql) {

        /*
         * Break V apart into 6 components
         *
         * V = |---------------|
         *     |V_{1,1} V_{1,2}|
         *     |V_{2,1} V_{2,2}|
         *     |0       V_{3,2}|
         *     |---------------|
         *
         * V_{1,1}\in\R^{n-k,k-l}  rectangular
         * V_{2,1}\in\R^{k-l,k-l}  unit upper triangular
         *
         * V_{1,2}\in\R^{n-k,l}    rectangular
         * V_{2,2}\in\R^{k-l,l}    rectangular
         * V_{3,2}\in\R^{l,l}      unit upper triangular
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} 0      |
         *     |T_{2,1} T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{k-l, k-l} non-unit lower triangular
         * T_{2,2}\in\R^{l, l}     non-unit lower triangular
         * T_{2,1}\in\R^{k-l, l}   rectangular
         *
         * Where l = floor(k/2)
         *
         * Then, consider the product:
         *
         * (I - V_2*T_{2,2}*V_2')*(I - V_1*T_{1,1}*V_1')
         * = I - V_2*T_{2,2}*V_2' - V_1*T_{1,1}*V_1' + V_2*T_{2,2}*V_2'*V_1*T_{1,1}*V_1'
         *
         * Define T_{2,1} = -T_{2,2}*V_2'*V_1*T_{1,1}
         *
         * Then, we can define the matrix V as
         * V = |-------|
         *     |V_1 V_2|
         *     |-------|
         *
         * So, our product is equivalent to the matrix product
         * I - V*T*V'
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{2,1}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n - l, k - l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n, l, &V[(k - l) * ldv], ldv,
               &tau[k - l], &T[(k - l) + (k - l) * ldt], ldt);

        /*
         * Compute T_{2,1}
         * T_{2,1} = V_{2,2}'
         */
        for (j = 0; j < k - l; j++) {
            for (i = 0; i < l; i++) {
                T[(k - l + i) + j * ldt] = V[(n - k + j) + (k - l + i) * ldv];
            }
        }

        /* T_{2,1} = T_{2,1}*V_{2,1} */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasUnit, l, k - l, ONE, &V[n - k], ldv,
                    &T[k - l], ldt);

        /*
         * T_{2,1} = V_{2,2}'*V_{2,1} + T_{2,1}
         * Note: We assume K <= N, and GEMM will do nothing if N=K
         */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, l, k - l, n - k,
                    ONE, &V[(k - l) * ldv], ldv, V, ldv, ONE,
                    &T[k - l], ldt);

        /*
         * At this point, we have that T_{2,1} = V_2'*V_1
         * All that is left is to pre and post multiply by -T_{2,2} and T_{1,1}
         * respectively.
         */

        /* T_{2,1} = -T_{2,2}*T_{2,1} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, k - l, NEG_ONE,
                    &T[(k - l) + (k - l) * ldt], ldt, &T[k - l], ldt);

        /* T_{2,1} = T_{2,1}*T_{1,1} */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, k - l, ONE, T, ldt, &T[k - l], ldt);
    } else if (rq) {

        /*
         * Break V apart into 6 components
         *
         * V = |-----------------------|
         *     |V_{1,1} V_{1,2} 0      |
         *     |V_{2,1} V_{2,2} V_{2,3}|
         *     |-----------------------|
         *
         * V_{1,1}\in\R^{k-l,n-k}  rectangular
         * V_{1,2}\in\R^{k-l,k-l}  unit lower triangular
         *
         * V_{2,1}\in\R^{l,n-k}    rectangular
         * V_{2,2}\in\R^{l,k-l}    rectangular
         * V_{2,3}\in\R^{l,l}      unit lower triangular
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} 0      |
         *     |T_{2,1} T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{k-l, k-l} non-unit lower triangular
         * T_{2,2}\in\R^{l, l}     non-unit lower triangular
         * T_{2,1}\in\R^{k-l, l}   rectangular
         *
         * Where l = floor(k/2)
         *
         * Then, consider the product:
         *
         * (I - V_2'*T_{2,2}'*V_2)*(I - V_1'*T_{1,1}'*V_1)
         * = I - V_2'*T_{2,2}'*V_2 - V_1'*T_{1,1}'*V_1 + V_2'*T_{2,2}'*V_2*V_1'*T_{1,1}'*V_1
         *
         * Define T_{2,1} = -T_{2,2}*V_2*V_1'*T_{1,1}
         *
         * Then, we can define the matrix V as
         * V = |---|
         *     |V_1|
         *     |V_2|
         *     |---|
         *
         * So, our product is equivalent to the matrix product
         * I - V'*T'*V
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{2,1}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n - l, k - l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n, l, &V[k - l], ldv,
               &tau[k - l], &T[(k - l) + (k - l) * ldt], ldt);

        /*
         * Compute T_{2,1}
         * T_{2,1} = V_{2,2}
         */
        slacpy("All", l, k - l, &V[(k - l) + (n - k) * ldv], ldv,
               &T[k - l], ldt);

        /* T_{2,1} = T_{2,1}*V_{1,2}' */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                    CblasUnit, l, k - l, ONE, &V[(n - k) * ldv], ldv,
                    &T[k - l], ldt);

        /*
         * T_{2,1} = V_{2,1}*V_{1,1}' + T_{2,1}
         * Note: We assume K <= N, and GEMM will do nothing if N=K
         */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, l, k - l, n - k,
                    ONE, &V[k - l], ldv, V, ldv, ONE, &T[k - l], ldt);

        /*
         * At this point, we have that T_{2,1} = V_2*V_1'
         * All that is left is to pre and post multiply by -T_{2,2} and T_{1,1}
         * respectively.
         */

        /* T_{2,1} = -T_{2,2}*T_{2,1} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, k - l, NEG_ONE,
                    &T[(k - l) + (k - l) * ldt], ldt, &T[k - l], ldt);

        /* T_{2,1} = T_{2,1}*T_{1,1} */
        cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, k - l, ONE, T, ldt, &T[k - l], ldt);
    } else if (rqt) {

        /*
         * Break V apart into 6 components
         *
         * V = |-----------------------|
         *     |V_{1,1} V_{1,2} 0      |
         *     |V_{2,1} V_{2,2} V_{2,3}|
         *     |-----------------------|
         *
         * V_{1,1}\in\R^{k-l,n-k}  rectangular
         * V_{1,2}\in\R^{k-l,k-l}  unit lower triangular
         *
         * V_{2,1}\in\R^{l,n-k}    rectangular
         * V_{2,2}\in\R^{l,k-l}    rectangular
         * V_{2,3}\in\R^{l,l}      unit lower triangular
         *
         * We will construct the T matrix
         * T = |---------------|
         *     |T_{1,1} T_{1,2}|
         *     | 0      T_{2,2}|
         *     |---------------|
         *
         * T is the triangular factor obtained from block reflectors.
         * To motivate the structure, assume we have already computed T_{1,1}
         * and T_{2,2}. Then collect the associated reflectors in V_1 and V_2
         *
         * T_{1,1}\in\R^{k-l, k-l} non-unit upper triangular
         * T_{2,2}\in\R^{l, l}     non-unit upper triangular
         * T_{1,2}\in\R^{k-l, l}   rectangular
         *
         * Where l = floor(k/2)
         *
         * Then, consider the product:
         *
         * (I - V_1'*T_{1,1}*V_1)*(I - V_2'*T_{2,2}*V_2)
         * = I - V_1'*T_{1,1}*V_1 - V_2'*T_{2,2}*V_2 + V_1'*T_{1,1}*V_1*V_2'*T_{2,2}*V_2
         *
         * Define T_{1,2} = -T_{1,1}*V_1*V_2'*T_{2,2}
         *
         * Then, we can define the matrix V as
         * V = |---|
         *     |V_1|
         *     |V_2|
         *     |---|
         *
         * So, our product is equivalent to the matrix product
         * I - V'*T*V
         * This means, we can compute T_{1,1} and T_{2,2}, then use this information
         * to compute T_{1,2}
         */

        /* Compute T_{1,1} recursively */
        slarft(direct, storev, n - l, k - l, V, ldv, tau, T, ldt);

        /* Compute T_{2,2} recursively */
        slarft(direct, storev, n, l, &V[k - l], ldv,
               &tau[k - l], &T[(k - l) + (k - l) * ldt], ldt);

        /*
         * Compute T_{1,2}
         * T_{1,2} = V_{2,2}'
         */
        for (i = 0; i < k - l; i++) {
            for (j = 0; j < l; j++) {
                T[i + (k - l + j) * ldt] = V[(k - l + j) + (n - k + i) * ldv];
            }
        }

        /* T_{1,2} = V_{1,2}T_{1,2} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasUnit, k - l, l, ONE, &V[(n - k) * ldv], ldv,
                    &T[(k - l) * ldt], ldt);

        /* T_{1,2} = V_{1,1}V_{2,1}' + T_{1,2} */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, k - l, l, n - k,
                    ONE, V, ldv, &V[k - l], ldv, ONE,
                    &T[(k - l) * ldt], ldt);

        /*
         * At this point, we have that T_{1,2} = V_1*V_2'
         * All that is left is to pre and post multiply by -T_{1,1} and T_{2,2}
         * respectively.
         */

        /* T_{1,2} = -T_{1,1}*T_{1,2} */
        cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, k - l, l, NEG_ONE, T, ldt,
                    &T[(k - l) * ldt], ldt);

        /* T_{1,2} = T_{1,2}*T_{2,2} */
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, k - l, l, ONE, &T[(k - l) + (k - l) * ldt],
                    ldt, &T[(k - l) * ldt], ldt);
    }
}
