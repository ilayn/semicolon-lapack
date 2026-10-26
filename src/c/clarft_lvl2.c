/**
 * @file clarft_lvl2.c
 * @brief CLARFT_LVL2 forms the triangular factor T of a block reflector H = I - V*T*V**H.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CLARFT_LVL2 forms the triangular factor T of a complex block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
 *
 * If `direct='F'`, H = H(0) H(1) . . . H(k-1) and T is upper triangular;
 *
 * If `direct='B'`, H = H(k-1) . . . H(1) H(0) and T is lower triangular.
 *
 * If `storev='C'`, the vector which defines the elementary reflector
 * H(i) is stored in column `i` of the array `V`, and
 *
 *    H  =  I - V * T * V**H
 *
 * If `storev='R'`, the vector which defines the elementary reflector
 * H(i) is stored in row `i` of the array `V`, and
 *
 *    H  =  I - V**H * T * V
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
 *     DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
 *
 *                  V = (  1       )                 V = (  1 v0 v0 v0 v0 )
 *                      ( v0  1    )                     (     1 v1 v1 v1 )
 *                      ( v0 v1  1 )                     (        1 v2 v2 )
 *                      ( v0 v1 v2 )
 *                      ( v0 v1 v2 )
 *
 *     DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
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
void clarft_lvl2(const char* direct, const char* storev,
                 const INT n, const INT k,
                 const c64* restrict V, const INT ldv,
                 const c64* restrict tau,
                 c64* restrict T, const INT ldt)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    c64 alpha;
    INT i, j, kmi, nmi, info;
    INT qr, lq, ql, rq, lqt, rqt;
    INT dirf, dirb, dirt;
    INT storec, storer, storet;

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
        xerbla("CLARFT_LVL2", info);
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
    if (n == 1 || k == 1) {
        if (lqt || rqt) {
            T[0] = conjf(tau[0]);
        } else {
            T[0] = tau[0];
        }
        return;
    }
    if (qr) {

        /*
         * Break V into 9 components
         *
         * V = |-----------------------|
         *     |V_{1,1} 0       0      | i
         *     |V_{2,1} V_{2,2} 0      | 1
         *     |V_{3,1} V_{3,2} V_{3,3}| n-i-1
         *     |-----------------------|
         *      i       1       k-i-1
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit lower triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{3,3} and repeat until we
         * reach the end. On each iteration V_{3,3} is not referenced
         *
         * We will construct T one column at a time from left to right
         * after initializing T(0,0) = tau(0)
         *
         * T = |-------------------------|
         *     | T_{1,1} T_{1,2} T_{1,3} | i
         *     | 0       T_{2,2} T_{2,3} | 1
         *     | 0       0       T_{3,3} | k-i-1
         *     |-------------------------|
         *       i       1       k-i-1
         *
         * T_{1,1}, T_{2,2}, and T_{3,3} are non-unit lower triangular
         *
         * Similarly as above, we will construct T_{1,2} and T_{2,2} at
         * each iteration i = 1, \dots k-1, and then grow into T_{1:3,3}. On
         * each iteration, T_{1:3,3} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We now get
         *
         * T_{1,2} = -T_{1,1}[V_{1,1}\\V_{2,1}\\V_{3,1}]'
         *                [0\\V_{2,2}\\V_{3,2}]T_{2,2}
         *
         * T_{1,2} = -T_{1,1}(V_{2,1}' + V_{3,1}'V_{3,2})T_{2,2}
         *
         * This means we will do the following
         *
         * T_{1,2} = -V_{2,1}'T_{2,2} = -\tau_{i}V_{2,1}'
         * T_{1,2} = -\tau_{i}V_{3,2}' V_{3,1} + T_{1,2}
         * T_{1,2} = T_{1,1}T_{1,2}
         * T_{2,2} = \tau{i}
         */
        T[0] = tau[0];

        for (i = 1; i < k; i++) {

            /*
             * T_{1,2} = -V_{2,1}'V_{2,2}T_{2,2} = -\tau_i V_{2,1}'
             * We must do this at copy time as otherwise gemv will do nothing
             * on the last column when n=k, but we need to make sure we are
             * scaled by this value
             */
            for (j = 0; j < i; j++) {
                T[j + i * ldt] = -tau[i] * conjf(V[i + j * ldv]);
            }

            /*
             * T_{1,2} = -V_{3,1}'V_{3,2}T_{2,2} + T_{1,2}
             *         = -\tau{i} V_{3,2}'V_{3,1} + T_{1,2}
             */
            alpha = -tau[i];
            cblas_cgemv(CblasColMajor, CblasConjTrans, n - i - 1, i, &alpha,
                        &V[i + 1], ldv, &V[(i + 1) + i * ldv], 1, &ONE,
                        &T[i * ldt], 1);

            /* T_{1,2} = T_{1,1}T_{1,2} */
            cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        i, T, ldt, &T[i * ldt], 1);

            /* T_{2,2} = \tau{i} */
            T[i + i * ldt] = tau[i];
        }
    } else if (lq) {

        /*
         * Break V into 9 components
         *
         * V = |-------------------------|
         *     | V_{1,1} V_{1,2} V_{1,3} | i
         *     | 0       V_{2,2} V_{2,3} | 1
         *     | 0       0       V_{3,3} | k-i-1
         *     |-------------------------|
         *       i       1       n-i-1
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit upper triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{3,3} and repeat until we
         * reach the end. On each iteration V_{3,3} is not referenced
         *
         * We will construct T one column at a time from left to right
         * after initializing T(0,0) = tau(0)
         *
         * T = |-------------------------|
         *     | T_{1,1} T_{1,2} T_{1,3} | i
         *     | 0       T_{2,2} T_{2,3} | 1
         *     | 0       0       T_{3,3} | k-i-1
         *     |-------------------------|
         *       i       1       k-i-1
         *
         * Similarly as above, we will construct T_{1,2} and T_{2,2} at
         * each iteration i = 1, \dots k-1, and then grow into T_{1:3,3}. On
         * each iteration, T_{1:3,3} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We now get
         *
         * T_{1,2} = -T_{1,1}[V_{1,1} V_{1,2} V_{1,3}][ 0 V_{2,2} V_{2,3} ]'T_{2,2}
         *
         * T_{1,2} = -T_{1,1}(V_{1,2} + V_{1,3}V_{2,3}')T_{2,2}
         *
         * This means we will do the following
         *
         * T_{1,2} = -V_{1,2}T_{2,2} = -\tau_{i}V_{1,2}
         * T_{1,2} = -\tau_{i}V_{1,3}V_{2,3}' + T_{1,2}
         * T_{1,2} = T_{1,1}T_{1,2}
         * T_{2,2} = \tau{i}
         */
        T[0] = tau[0];

        for (i = 1; i < k; i++) {

            /* T_{1,2} = -\tau_{i}V_{1,2} */
            for (j = 0; j < i; j++) {
                T[j + i * ldt] = -tau[i] * V[j + i * ldv];
            }

            /* T_{1,2} = -\tau_{i}V_{1,3}V_{2,3}' + T_{1,2} */
            alpha = -tau[i];
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, i, 1,
                        n - i - 1, &alpha, &V[(i + 1) * ldv], ldv,
                        &V[i + (i + 1) * ldv], ldv, &ONE, &T[i * ldt], ldt);

            /* T_{1,2} = T_{1,1}T_{1,2} */
            cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        i, T, ldt, &T[i * ldt], 1);

            /* T_{2,2} = \tau{i} */
            T[i + i * ldt] = tau[i];
        }
    } else if (lqt) {

        /*
         * Break V into 9 components
         *
         * V = |-------------------------|
         *     | V_{1,1} V_{1,2} V_{1,3} | i
         *     | 0       V_{2,2} V_{2,3} | 1
         *     | 0       0       V_{3,3} | k-i-1
         *     |-------------------------|
         *       i       1       n-i-1
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit upper triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{3,3} and repeat until we
         * reach the end. On each iteration V_{3,3} is not referenced
         *
         * We will construct T one column at a time from left to right
         * after initializing T(0,0) = tau(0)
         *
         * T = |-------------------------|
         *     | T_{1,1} 0       0       | i
         *     | T_{2,1} T_{2,2} 0       | 1
         *     | T_{3,1} T_{3,2} T_{3,3} | k-i-1
         *     |-------------------------|
         *       i       1       k-i-1
         *
         * Similarly as above, we will construct T_{2,1} and T_{2,2} at
         * each iteration i = 1, \dots k-1, and then grow into T_{3,1:3}. On
         * each iteration, T_{3,1:3} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We now get
         *
         * T_{2,1} = -T_{2,2}[0 V_{2,2} V_{2,3}][V_{1,1} V_{1,2} V_{1,3}]'T_{1,1}
         *
         * T_{2,1} = -T_{2,2}(V_{1,2}' + V_{2,3}V_{1,3}')T_{1,1}
         *
         * This means we will do the following
         *
         * T_{2,1} = -T_{2,2}V_{1,2}' = -\tau_{i}V_{1,2}'
         * T_{2,1} = -\tau_{i}V_{1,3}V_{2,3}' + T_{2,1}
         * T_{2,1} = T_{1,1}'T_{2,1}
         * T_{2,2} = \tau{i}
         */
        T[0] = conjf(tau[0]);

        for (i = 1; i < k; i++) {

            /* T_{2,1} = -\tau_{i}V_{1,2}' */
            for (j = 0; j < i; j++) {
                T[i + j * ldt] = -conjf(tau[i] * V[j + i * ldv]);
            }

            /* T_{2,1} = -\tau_{i}V_{2,3}V_{1,3}' + T_{2,1} */
            alpha = -conjf(tau[i]);
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, 1, i,
                        n - i - 1, &alpha, &V[i + (i + 1) * ldv], ldv,
                        &V[(i + 1) * ldv], ldv, &ONE, &T[i], ldt);

            /* T_{2,1} = T_{1,1}'T_{2,1} */
            cblas_ctrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                        i, T, ldt, &T[i], ldt);

            T[i + i * ldt] = conjf(tau[i]);
        }
    } else if (ql) {

        /*
         * Break V into 9 components
         *
         * V = |-------------------------|
         *     | V_{1,1} V_{1,2} V_{1,3} | n-i-1
         *     | 0       V_{2,2} V_{2,3} | 1
         *     | 0       0       V_{3,3} | i
         *     |-------------------------|
         *       k-i-1   1       i
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit upper triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{1,1} and repeat until we
         * reach the end. On each iteration V_{1,1} is not referenced
         *
         * We will construct T one column at a time from right to left
         * after initializing T(k-1,k-1) = tau(k-1)
         *
         * T = |-------------------------|
         *     | T_{1,1} 0       0       | k-i-1
         *     | T_{2,1} T_{2,2} 0       | 1
         *     | T_{3,1} T_{3,2} T_{3,3} | i
         *     |-------------------------|
         *       k-i-1   1       i
         *
         * T_{1,1}, T_{2,2}, and T_{3,3} are non-unit lower triangular
         *
         * Similarly as above, we will construct T_{2,2} and T_{3,2} at
         * each iteration i = 1, \dots k-1, and then grow into T_{1:3,1}. On
         * each iteration, T_{1:3,1} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We get that
         *
         * T_{3,2} = -T_{3,3}[V_{1,3}\\V_{2,3}\\V_{3,3}]'
         * [V_{1,2}\\V_{2,2}\\0]T_{2,2}
         *
         * T_{3,2} = -T_{3,3}(V_{1,3}'V_{1,2} + V_{2,3}')T_{2,2}
         *
         * Thus, we will compute
         *
         * T_{2,2} = \tau_{k-i-1}
         * T_{3,2} = -T_{3,3}V_{3,2}' = -\tau_{k-i-1}V_{3,2}'
         * T_{3,2} = -\tau_{k-i-1}V_{1,3}'V_{1,2} + T_{3,2}
         * T_{3,2} = T_{3,3}T_{3,2}
         */
        T[(k - 1) + (k - 1) * ldt] = tau[k - 1];
        for (i = 1; i < k; i++) {
            kmi = k - i - 1;
            nmi = n - i - 1;

            /* T_{2,2} = \tau(k-i-1) */
            T[kmi + kmi * ldt] = tau[kmi];

            /* T_{3,2} = -\tau(k-i-1)V_{2,3}' */
            for (j = 0; j < i; j++) {
                T[(kmi + 1 + j) + kmi * ldt] =
                    -tau[kmi] * conjf(V[nmi + (kmi + 1 + j) * ldv]);
            }

            /* T_{3,2} = -\tau(k-i-1)V_{1,3}'V_{1,2} + T_{3,2} */
            alpha = -tau[kmi];
            cblas_cgemv(CblasColMajor, CblasConjTrans, n - i - 1, i, &alpha,
                        &V[(kmi + 1) * ldv], ldv, &V[kmi * ldv], 1, &ONE,
                        &T[(kmi + 1) + kmi * ldt], 1);

            /* T_{3,2} = T_{3,3}T_{3,2} */
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        i, &T[(kmi + 1) + (kmi + 1) * ldt], ldt,
                        &T[(kmi + 1) + kmi * ldt], 1);
        }
    } else if (rq) {

        /*
         * Break V into 9 components
         *
         * V = |-------------------------|
         *     | V_{1,1} 0       0       | k-i-1
         *     | V_{2,1} V_{2,2} 0       | 1
         *     | V_{3,1} V_{3,2} V_{3,3} | i
         *     |-------------------------|
         *       n-i-1   1       i
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit lower triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{1,1} and repeat until we
         * reach the end. On each iteration V_{1,1} is not referenced
         *
         * We will construct T one column at a time from right to left
         * after initializing T(k-1,k-1) = tau(k-1)
         *
         * T = |-------------------------|
         *     | T_{1,1} 0       0       | k-i-1
         *     | T_{2,1} T_{2,2} 0       | 1
         *     | T_{3,1} T_{3,2} T_{3,3} | i
         *     |-------------------------|
         *       k-i-1   1       i
         *
         * T_{1,1}, T_{2,2}, and T_{3,3} are non-unit lower triangular
         *
         * Similarly as above, we will construct T_{2,2} and T_{3,2} at
         * each iteration i = 1, \dots k-1, and then grow into T_{1:3,1}. On
         * each iteration, T_{1:3,1} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We get that
         *
         * T_{3,2} = -T_{3,3}[V_{3,1} V_{3,2} V_{3,3}][V_{2,1} V_{2,2} 0]'T_{2,2}
         *
         * T_{3,2} = -T_{3,3}(V_{3,1}V_{2,1}' + V_{3,2})T_{2,2}
         *
         * Thus, we will compute
         *
         * T_{2,2} = \tau_{k-i-1}
         * T_{3,2} = -\tau_{k-i-1}V_{3,2}
         * T_{3,2} = -\tau_{k-i-1}V_{3,1}V_{2,1}' + T_{3,2}
         * T_{3,2} = T_{3,3}T_{3,2}
         */
        T[(k - 1) + (k - 1) * ldt] = tau[k - 1];
        for (i = 1; i < k; i++) {
            kmi = k - i - 1;
            nmi = n - i - 1;

            /* T_{2,2} = \tau_{k-i-1} */
            T[kmi + kmi * ldt] = tau[kmi];

            /* T_{3,2} = -\tau_{k-i-1}V_{3,2} */
            for (j = 0; j < i; j++) {
                T[(kmi + 1 + j) + kmi * ldt] =
                    -tau[kmi] * V[(kmi + 1 + j) + nmi * ldv];
            }

            /* T_{3,2} = -\tau_{k-i-1}V_{3,1}V_{2,1}' + T_{3,2} */
            alpha = -tau[kmi];
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, i, 1,
                        n - i - 1, &alpha, &V[kmi + 1], ldv, &V[kmi], ldv,
                        &ONE, &T[(kmi + 1) + kmi * ldt], ldt);

            /* T_{3,2} = T_{3,3}T_{3,2} */
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        i, &T[(kmi + 1) + (kmi + 1) * ldt], ldt,
                        &T[(kmi + 1) + kmi * ldt], 1);
        }
    } else if (rqt) {

        /*
         * Break V into 9 components
         *
         * V = |-------------------------|
         *     | V_{1,1} 0       0       | k-i-1
         *     | V_{2,1} V_{2,2} 0       | 1
         *     | V_{3,1} V_{3,2} V_{3,3} | i
         *     |-------------------------|
         *       n-i-1   1       i
         *
         * V_{1,1}, V_{2,2} and V_{3,3} are unit lower triangular
         *
         * This is how we are going to view the matrix V at each step
         * i=1,\dots,k-1, then we grow into V_{1,1} and repeat until we
         * reach the end. On each iteration V_{1,1} is not referenced
         *
         * We will construct T one column at a time from right to left
         * after initializing T(k-1,k-1) = tau(k-1)
         *
         * T = |-------------------------|
         *     | T_{1,1} T_{1,2} T_{1,3} | k-i-1
         *     | 0       T_{2,2} T_{2,3} | 1
         *     | 0       0       T_{3,3} | i
         *     |-------------------------|
         *       k-i-1   1       i
         *
         * T_{1,1}, T_{2,2}, and T_{3,3} are non-unit lower triangular
         *
         * Similarly as above, we will construct T_{2,2} and T_{2,3} at
         * each iteration i = 1, \dots k-1, and then grow into T_{1,1:3}. On
         * each iteration, T_{1,1:3} are not referenced. See clarft.c
         * for details on how these formulae were constructed.
         *
         * We get that
         *
         * T_{2,3} = -T_{2,2}[V_{2,1} V_{2,2} 0][V_{3,1} V_{3,2} V_{3,3}]'T_{3,3}
         *
         * T_{3,2} = -T_{2,2}(V_{2,1}V_{3,1}' + V_{3,2}')T_{3,3}
         *
         * Thus, we will compute
         *
         * T_{2,2} = \tau_{k-i-1}
         * T_{2,3} = -\tau_{k-i-1}V_{3,2}'
         * T_{2,3} = -\tau_{k-i-1}V_{2,1}V_{3,1}' + T_{2,3}
         * T_{2,3} = T_{3,3}'T_{2,3}
         */
        T[(k - 1) + (k - 1) * ldt] = conjf(tau[k - 1]);
        for (i = 1; i < k; i++) {
            kmi = k - i - 1;
            nmi = n - i - 1;

            /* T_{2,2} = \tau_{k-i-1} */
            T[kmi + kmi * ldt] = conjf(tau[kmi]);

            /* T_{2,3} = -\tau_{k-i-1}V_{3,2}' */
            for (j = 0; j < i; j++) {
                T[kmi + (kmi + 1 + j) * ldt] =
                    -conjf(tau[kmi] * V[(kmi + 1 + j) + nmi * ldv]);
            }

            /* T_{2,3} = -\tau_{k-i-1}V_{2,1}V_{3,1}' + T_{2,3} */
            alpha = -conjf(tau[kmi]);
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, 1, i,
                        n - i - 1, &alpha, &V[kmi], ldv, &V[kmi + 1], ldv,
                        &ONE, &T[kmi + (kmi + 1) * ldt], ldt);

            /* T_{2,3} = T_{3,3}'T_{2,3} */
            cblas_ctrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        i, &T[(kmi + 1) + (kmi + 1) * ldt], ldt,
                        &T[kmi + (kmi + 1) * ldt], ldt);
        }
    }
}
