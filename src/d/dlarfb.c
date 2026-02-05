/**
 * @file dlarfb.c
 * @brief DLARFB applies a block reflector or its transpose to a general
 *        rectangular matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARFB applies a real block reflector H or its transpose H**T to a
 * real m by n matrix C, from either the left or the right.
 *
 * @param[in]     side    'L': apply from left; 'R': apply from right.
 * @param[in]     trans   'N': apply H; 'T': apply H**T.
 * @param[in]     direct  'F': H = H(1)H(2)...H(k) (Forward);
 *                        'B': H = H(k)...H(2)H(1) (Backward).
 * @param[in]     storev  'C': columnwise; 'R': rowwise.
 * @param[in]     m       Number of rows of C.
 * @param[in]     n       Number of columns of C.
 * @param[in]     k       Number of elementary reflectors.
 * @param[in]     V       Reflector matrix.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     T       k-by-k triangular factor.
 * @param[in]     ldt     Leading dimension of T. ldt >= k.
 * @param[in,out] C       m-by-n matrix. Overwritten on exit.
 * @param[in]     ldc     Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work    Workspace, dimension (ldwork, k).
 * @param[in]     ldwork  Leading dimension of work.
 *                        ldwork >= max(1, n) if side='L';
 *                        ldwork >= max(1, m) if side='R'.
 */
void dlarfb(const char* side, const char* trans, const char* direct,
            const char* storev, const int m, const int n, const int k,
            const double * const restrict V, const int ldv,
            const double * const restrict T, const int ldt,
            double * const restrict C, const int ldc,
            double * const restrict work, const int ldwork)
{
    const double ONE = 1.0;
    const double NEG_ONE = -1.0;
    int i, j;
    CBLAS_TRANSPOSE transt;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) return;

    /* Determine opposite transpose for T multiplication */
    if (trans[0] == 'N' || trans[0] == 'n') {
        transt = CblasTrans;
    } else {
        transt = CblasNoTrans;
    }

    if (storev[0] == 'C' || storev[0] == 'c') {
        if (direct[0] == 'F' || direct[0] == 'f') {
            /* STOREV='C', DIRECT='F':
             * V = (V1) where V1 is unit lower triangular (first k rows)
             *     (V2)
             */
            if (side[0] == 'L' || side[0] == 'l') {
                /* Form H*C or H^T*C where C = (C1)
                 *                              (C2)
                 * W := C^T * V = C1^T*V1 + C2^T*V2 */

                /* W := C1^T (copy rows 0..k-1 of C, transposed) */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(n, &C[j], ldc, &work[j * ldwork], 1);
                }

                /* W := W * V1 (unit lower triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            n, k, ONE, V, ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C2^T * V2 */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n, k, m - k, ONE,
                                &C[k], ldc, &V[k], ldv,
                                ONE, work, ldwork);
                }

                /* W := W * T^T or W * T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            transt, CblasNonUnit,
                            n, k, ONE, T, ldt, work, ldwork);

                /* C := C - V * W^T */
                if (m > k) {
                    /* C2 := C2 - V2 * W^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m - k, n, k, NEG_ONE,
                                &V[k], ldv, work, ldwork,
                                ONE, &C[k], ldc);
                }

                /* W := W * V1^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasUnit,
                            n, k, ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W^T */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[j + i * ldc] -= work[i + j * ldwork];
                    }
                }

            } else {
                /* SIDE='R': Form C*H or C*H^T where C = (C1 C2)
                 * W := C * V = C1*V1 + C2*V2 */

                /* W := C1 (copy columns 0..k-1 of C) */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(m, &C[j * ldc], 1, &work[j * ldwork], 1);
                }

                /* W := W * V1 (unit lower triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            m, k, ONE, V, ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C2 * V2 */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, k, n - k, ONE,
                                &C[k * ldc], ldc, &V[k], ldv,
                                ONE, work, ldwork);
                }

                /* W := W * T or W * T^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasTrans),
                            CblasNonUnit,
                            m, k, ONE, T, ldt, work, ldwork);

                /* C := C - W * V^T */
                if (n > k) {
                    /* C2 := C2 - W * V2^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m, n - k, k, NEG_ONE,
                                work, ldwork, &V[k], ldv,
                                ONE, &C[k * ldc], ldc);
                }

                /* W := W * V1^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasUnit,
                            m, k, ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + j * ldc] -= work[i + j * ldwork];
                    }
                }
            }

        } else {
            /* STOREV='C', DIRECT='B':
             * V = (V1) where V2 is unit upper triangular (last k rows)
             *     (V2)
             */
            if (side[0] == 'L' || side[0] == 'l') {
                /* Form H*C or H^T*C where C = (C1)
                 *                              (C2)
                 * W := C^T * V = C1^T*V1 + C2^T*V2 */

                /* W := C2^T (copy rows m-k..m-1 of C, transposed) */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(n, &C[(m - k + j)], ldc,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2 (unit upper triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            n, k, ONE, &V[m - k], ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C1^T * V1 */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n, k, m - k, ONE,
                                C, ldc, V, ldv, ONE, work, ldwork);
                }

                /* W := W * T^T or W * T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            transt, CblasNonUnit,
                            n, k, ONE, T, ldt, work, ldwork);

                /* C := C - V * W^T */
                if (m > k) {
                    /* C1 := C1 - V1 * W^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m - k, n, k, NEG_ONE,
                                V, ldv, work, ldwork, ONE, C, ldc);
                }

                /* W := W * V2^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasUnit,
                            n, k, ONE, &V[m - k], ldv, work, ldwork);

                /* C2 := C2 - W^T */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[(m - k + j) + i * ldc] -= work[i + j * ldwork];
                    }
                }

            } else {
                /* SIDE='R': Form C*H or C*H^T where C = (C1 C2)
                 * W := C * V = C1*V1 + C2*V2 */

                /* W := C2 (copy columns n-k..n-1 of C) */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(m, &C[(n - k + j) * ldc], 1,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2 (unit upper triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            m, k, ONE, &V[n - k], ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C1 * V1 */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, k, n - k, ONE,
                                C, ldc, V, ldv, ONE, work, ldwork);
                }

                /* W := W * T or W * T^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasTrans),
                            CblasNonUnit,
                            m, k, ONE, T, ldt, work, ldwork);

                /* C := C - W * V^T */
                if (n > k) {
                    /* C1 := C1 - W * V1^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m, n - k, k, NEG_ONE,
                                work, ldwork, V, ldv, ONE, C, ldc);
                }

                /* W := W * V2^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasUnit,
                            m, k, ONE, &V[n - k], ldv, work, ldwork);

                /* C2 := C2 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + (n - k + j) * ldc] -= work[i + j * ldwork];
                    }
                }
            }
        }

    } else {
        /* STOREV = 'R' */
        if (direct[0] == 'F' || direct[0] == 'f') {
            /* STOREV='R', DIRECT='F':
             * V = (V1 V2) where V1 is unit upper triangular (first k columns)
             */
            if (side[0] == 'L' || side[0] == 'l') {
                /* Form H*C or H^T*C where C = (C1)
                 *                              (C2)
                 * W := C^T * V^T = C1^T*V1^T + C2^T*V2^T */

                /* W := C1^T */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(n, &C[j], ldc, &work[j * ldwork], 1);
                }

                /* W := W * V1^T (unit upper triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasUnit,
                            n, k, ONE, V, ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C2^T * V2^T */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                                n, k, m - k, ONE,
                                &C[k], ldc, &V[k * ldv], ldv,
                                ONE, work, ldwork);
                }

                /* W := W * T^T or W * T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            transt, CblasNonUnit,
                            n, k, ONE, T, ldt, work, ldwork);

                /* C := C - V^T * W^T */
                if (m > k) {
                    /* C2 := C2 - V2^T * W^T */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                                m - k, n, k, NEG_ONE,
                                &V[k * ldv], ldv, work, ldwork,
                                ONE, &C[k], ldc);
                }

                /* W := W * V1 */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            n, k, ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W^T */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[j + i * ldc] -= work[i + j * ldwork];
                    }
                }

            } else {
                /* SIDE='R': Form C*H or C*H^T where C = (C1 C2)
                 * W := C * V^T = C1*V1^T + C2*V2^T */

                /* W := C1 */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(m, &C[j * ldc], 1, &work[j * ldwork], 1);
                }

                /* W := W * V1^T (unit upper triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasTrans, CblasUnit,
                            m, k, ONE, V, ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C2 * V2^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m, k, n - k, ONE,
                                &C[k * ldc], ldc, &V[k * ldv], ldv,
                                ONE, work, ldwork);
                }

                /* W := W * T or W * T^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasTrans),
                            CblasNonUnit,
                            m, k, ONE, T, ldt, work, ldwork);

                /* C := C - W * V */
                if (n > k) {
                    /* C2 := C2 - W * V2 */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n - k, k, NEG_ONE,
                                work, ldwork, &V[k * ldv], ldv,
                                ONE, &C[k * ldc], ldc);
                }

                /* W := W * V1 */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            m, k, ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + j * ldc] -= work[i + j * ldwork];
                    }
                }
            }

        } else {
            /* STOREV='R', DIRECT='B':
             * V = (V1 V2) where V2 is unit lower triangular (last k columns)
             */
            if (side[0] == 'L' || side[0] == 'l') {
                /* Form H*C or H^T*C where C = (C1)
                 *                              (C2)
                 * W := C^T * V^T = C1^T*V1^T + C2^T*V2^T */

                /* W := C2^T */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(n, &C[(m - k + j)], ldc,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2^T (unit lower triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasUnit,
                            n, k, ONE, &V[(m - k) * ldv], ldv,
                            work, ldwork);

                if (m > k) {
                    /* W := W + C1^T * V1^T */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                                n, k, m - k, ONE,
                                C, ldc, V, ldv, ONE, work, ldwork);
                }

                /* W := W * T^T or W * T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            transt, CblasNonUnit,
                            n, k, ONE, T, ldt, work, ldwork);

                /* C := C - V^T * W^T */
                if (m > k) {
                    /* C1 := C1 - V1^T * W^T */
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                                m - k, n, k, NEG_ONE,
                                V, ldv, work, ldwork, ONE, C, ldc);
                }

                /* W := W * V2 */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            n, k, ONE, &V[(m - k) * ldv], ldv,
                            work, ldwork);

                /* C2 := C2 - W^T */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[(m - k + j) + i * ldc] -= work[i + j * ldwork];
                    }
                }

            } else {
                /* SIDE='R': Form C*H or C*H^T where C = (C1 C2)
                 * W := C * V^T = C1*V1^T + C2*V2^T */

                /* W := C2 */
                for (j = 0; j < k; j++) {
                    cblas_dcopy(m, &C[(n - k + j) * ldc], 1,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2^T (unit lower triangular) */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasUnit,
                            m, k, ONE, &V[(n - k) * ldv], ldv,
                            work, ldwork);

                if (n > k) {
                    /* W := W + C1 * V1^T */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m, k, n - k, ONE,
                                C, ldc, V, ldv, ONE, work, ldwork);
                }

                /* W := W * T or W * T^T */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasTrans),
                            CblasNonUnit,
                            m, k, ONE, T, ldt, work, ldwork);

                /* C := C - W * V */
                if (n > k) {
                    /* C1 := C1 - W * V1 */
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n - k, k, NEG_ONE,
                                work, ldwork, V, ldv, ONE, C, ldc);
                }

                /* W := W * V2 */
                cblas_dtrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            m, k, ONE, &V[(n - k) * ldv], ldv,
                            work, ldwork);

                /* C2 := C2 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + (n - k + j) * ldc] -= work[i + j * ldwork];
                    }
                }
            }
        }
    }
}
