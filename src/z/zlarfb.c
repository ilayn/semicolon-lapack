/**
 * @file zlarfb.c
 * @brief ZLARFB applies a complex block reflector or its conjugate transpose
 *        to a complex rectangular matrix.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFB applies a complex block reflector H or its conjugate transpose
 * H**H to a complex m by n matrix C, from either the left or the right.
 *
 * @param[in]     side    'L': apply from left; 'R': apply from right.
 * @param[in]     trans   'N': apply H; 'C': apply H**H.
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
void zlarfb(const char* side, const char* trans, const char* direct,
            const char* storev, const INT m, const INT n, const INT k,
            const c128* restrict V, const INT ldv,
            const c128* restrict T, const INT ldt,
            c128* restrict C, const INT ldc,
            c128* restrict work, const INT ldwork)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);
    INT i, j;
    CBLAS_TRANSPOSE transt;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) return;

    if (trans[0] == 'N' || trans[0] == 'n') {
        transt = CblasConjTrans;
    } else {
        transt = CblasNoTrans;
    }

    if (storev[0] == 'C' || storev[0] == 'c') {
        if (direct[0] == 'F' || direct[0] == 'f') {
            if (side[0] == 'L' || side[0] == 'l') {

                /* W := C**H * V = C1**H * V1 + C2**H * V2 */

                /* W := C1**H */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(n, &C[j], ldc, &work[j * ldwork], 1);
                    zlacgv(n, &work[j * ldwork], 1);
                }

                /* W := W * V1 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            n, k, &ONE, V, ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C2**H * V2 */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n, k, m - k, &ONE,
                                &C[k], ldc, &V[k], ldv,
                                &ONE, work, ldwork);
                }

                /* W := W * T**H or W * T */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            transt, CblasNonUnit,
                            n, k, &ONE, T, ldt, work, ldwork);

                /* C := C - V * W**H */
                if (m > k) {
                    /* C2 := C2 - V2 * W**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m - k, n, k, &NEG_ONE,
                                &V[k], ldv, work, ldwork,
                                &ONE, &C[k], ldc);
                }

                /* W := W * V1**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasUnit,
                            n, k, &ONE, V, ldv, work, ldwork);

                /* C1 := C1 - conj(W**H) */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[j + i * ldc] -= conj(work[i + j * ldwork]);
                    }
                }

            } else {

                /* W := C * V = C1 * V1 + C2 * V2 */

                /* W := C1 */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(m, &C[j * ldc], 1, &work[j * ldwork], 1);
                }

                /* W := W * V1 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            m, k, &ONE, V, ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C2 * V2 */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, k, n - k, &ONE,
                                &C[k * ldc], ldc, &V[k], ldv,
                                &ONE, work, ldwork);
                }

                /* W := W * T or W * T**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasConjTrans),
                            CblasNonUnit,
                            m, k, &ONE, T, ldt, work, ldwork);

                /* C := C - W * V**H */
                if (n > k) {
                    /* C2 := C2 - W * V2**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m, n - k, k, &NEG_ONE,
                                work, ldwork, &V[k], ldv,
                                &ONE, &C[k * ldc], ldc);
                }

                /* W := W * V1**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasUnit,
                            m, k, &ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + j * ldc] -= work[i + j * ldwork];
                    }
                }
            }

        } else {

            if (side[0] == 'L' || side[0] == 'l') {

                /* W := C**H * V = C1**H * V1 + C2**H * V2 */

                /* W := C2**H */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(n, &C[(m - k + j)], ldc,
                                &work[j * ldwork], 1);
                    zlacgv(n, &work[j * ldwork], 1);
                }

                /* W := W * V2 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            n, k, &ONE, &V[m - k], ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C1**H * V1 */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n, k, m - k, &ONE,
                                C, ldc, V, ldv, &ONE, work, ldwork);
                }

                /* W := W * T**H or W * T */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            transt, CblasNonUnit,
                            n, k, &ONE, T, ldt, work, ldwork);

                /* C := C - V * W**H */
                if (m > k) {
                    /* C1 := C1 - V1 * W**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m - k, n, k, &NEG_ONE,
                                V, ldv, work, ldwork, &ONE, C, ldc);
                }

                /* W := W * V2**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasUnit,
                            n, k, &ONE, &V[m - k], ldv, work, ldwork);

                /* C2 := C2 - conj(W**H) */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[(m - k + j) + i * ldc] -= conj(work[i + j * ldwork]);
                    }
                }

            } else {

                /* W := C * V = C1 * V1 + C2 * V2 */

                /* W := C2 */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(m, &C[(n - k + j) * ldc], 1,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            m, k, &ONE, &V[n - k], ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C1 * V1 */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, k, n - k, &ONE,
                                C, ldc, V, ldv, &ONE, work, ldwork);
                }

                /* W := W * T or W * T**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasConjTrans),
                            CblasNonUnit,
                            m, k, &ONE, T, ldt, work, ldwork);

                /* C := C - W * V**H */
                if (n > k) {
                    /* C1 := C1 - W * V1**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m, n - k, k, &NEG_ONE,
                                work, ldwork, V, ldv, &ONE, C, ldc);
                }

                /* W := W * V2**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasUnit,
                            m, k, &ONE, &V[n - k], ldv, work, ldwork);

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
            if (side[0] == 'L' || side[0] == 'l') {

                /* W := C**H * V**H = C1**H * V1**H + C2**H * V2**H */

                /* W := C1**H */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(n, &C[j], ldc, &work[j * ldwork], 1);
                    zlacgv(n, &work[j * ldwork], 1);
                }

                /* W := W * V1**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasUnit,
                            n, k, &ONE, V, ldv, work, ldwork);

                if (m > k) {
                    /* W := W + C2**H * V2**H */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasConjTrans,
                                n, k, m - k, &ONE,
                                &C[k], ldc, &V[k * ldv], ldv,
                                &ONE, work, ldwork);
                }

                /* W := W * T**H or W * T */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            transt, CblasNonUnit,
                            n, k, &ONE, T, ldt, work, ldwork);

                /* C := C - V**H * W**H */
                if (m > k) {
                    /* C2 := C2 - V2**H * W**H */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasConjTrans,
                                m - k, n, k, &NEG_ONE,
                                &V[k * ldv], ldv, work, ldwork,
                                &ONE, &C[k], ldc);
                }

                /* W := W * V1 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            n, k, &ONE, V, ldv, work, ldwork);

                /* C1 := C1 - conj(W**H) */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[j + i * ldc] -= conj(work[i + j * ldwork]);
                    }
                }

            } else {

                /* W := C * V**H = C1 * V1**H + C2 * V2**H */

                /* W := C1 */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(m, &C[j * ldc], 1, &work[j * ldwork], 1);
                }

                /* W := W * V1**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasConjTrans, CblasUnit,
                            m, k, &ONE, V, ldv, work, ldwork);

                if (n > k) {
                    /* W := W + C2 * V2**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m, k, n - k, &ONE,
                                &C[k * ldc], ldc, &V[k * ldv], ldv,
                                &ONE, work, ldwork);
                }

                /* W := W * T or W * T**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasConjTrans),
                            CblasNonUnit,
                            m, k, &ONE, T, ldt, work, ldwork);

                /* C := C - W * V */
                if (n > k) {
                    /* C2 := C2 - W * V2 */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n - k, k, &NEG_ONE,
                                work, ldwork, &V[k * ldv], ldv,
                                &ONE, &C[k * ldc], ldc);
                }

                /* W := W * V1 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasUnit,
                            m, k, &ONE, V, ldv, work, ldwork);

                /* C1 := C1 - W */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < m; i++) {
                        C[i + j * ldc] -= work[i + j * ldwork];
                    }
                }
            }

        } else {

            if (side[0] == 'L' || side[0] == 'l') {

                /* W := C**H * V**H = C1**H * V1**H + C2**H * V2**H */

                /* W := C2**H */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(n, &C[(m - k + j)], ldc,
                                &work[j * ldwork], 1);
                    zlacgv(n, &work[j * ldwork], 1);
                }

                /* W := W * V2**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasUnit,
                            n, k, &ONE, &V[(m - k) * ldv], ldv,
                            work, ldwork);

                if (m > k) {
                    /* W := W + C1**H * V1**H */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasConjTrans,
                                n, k, m - k, &ONE,
                                C, ldc, V, ldv, &ONE, work, ldwork);
                }

                /* W := W * T**H or W * T */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            transt, CblasNonUnit,
                            n, k, &ONE, T, ldt, work, ldwork);

                /* C := C - V**H * W**H */
                if (m > k) {
                    /* C1 := C1 - V1**H * W**H */
                    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasConjTrans,
                                m - k, n, k, &NEG_ONE,
                                V, ldv, work, ldwork, &ONE, C, ldc);
                }

                /* W := W * V2 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            n, k, &ONE, &V[(m - k) * ldv], ldv,
                            work, ldwork);

                /* C2 := C2 - conj(W**H) */
                for (j = 0; j < k; j++) {
                    for (i = 0; i < n; i++) {
                        C[(m - k + j) + i * ldc] -= conj(work[i + j * ldwork]);
                    }
                }

            } else {

                /* W := C * V**H = C1 * V1**H + C2 * V2**H */

                /* W := C2 */
                for (j = 0; j < k; j++) {
                    cblas_zcopy(m, &C[(n - k + j) * ldc], 1,
                                &work[j * ldwork], 1);
                }

                /* W := W * V2**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasConjTrans, CblasUnit,
                            m, k, &ONE, &V[(n - k) * ldv], ldv,
                            work, ldwork);

                if (n > k) {
                    /* W := W + C1 * V1**H */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m, k, n - k, &ONE,
                                C, ldc, V, ldv, &ONE, work, ldwork);
                }

                /* W := W * T or W * T**H */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            (trans[0] == 'N' || trans[0] == 'n' ? CblasNoTrans : CblasConjTrans),
                            CblasNonUnit,
                            m, k, &ONE, T, ldt, work, ldwork);

                /* C := C - W * V */
                if (n > k) {
                    /* C1 := C1 - W * V1 */
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n - k, k, &NEG_ONE,
                                work, ldwork, V, ldv, &ONE, C, ldc);
                }

                /* W := W * V2 */
                cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                            CblasNoTrans, CblasUnit,
                            m, k, &ONE, &V[(n - k) * ldv], ldv,
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
