/**
 * @file ztprfb.c
 * @brief ZTPRFB applies a complex "triangular-pentagonal" block reflector to a
 *        complex matrix, which is composed of two blocks.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTPRFB applies a complex "triangular-pentagonal" block reflector H or its
 * conjugate transpose H**H to a complex matrix C, which is composed of two
 * blocks A and B, either from the left or right.
 *
 * @param[in]     side    'L': apply H or H**H from the Left;
 *                        'R': apply H or H**H from the Right.
 * @param[in]     trans   'N': apply H (No transpose);
 *                        'C': apply H**H (Conjugate transpose).
 * @param[in]     direct  Indicates how H is formed from a product of
 *                        elementary reflectors:
 *                        'F': H = H(1) H(2) . . . H(k) (Forward);
 *                        'B': H = H(k) . . . H(2) H(1) (Backward).
 * @param[in]     storev  Indicates how the vectors which define the
 *                        elementary reflectors are stored:
 *                        'C': Columns; 'R': Rows.
 * @param[in]     m       The number of rows of the matrix B. m >= 0.
 * @param[in]     n       The number of columns of the matrix B. n >= 0.
 * @param[in]     k       The order of the matrix T, i.e. the number of
 *                        elementary reflectors whose product defines the
 *                        block reflector. k >= 0.
 * @param[in]     l       The order of the trapezoidal part of V.
 *                        k >= l >= 0. See Further Details.
 * @param[in]     V       The pentagonal matrix V, which contains the
 *                        elementary reflectors H(1), H(2), ..., H(k).
 * @param[in]     ldv     The leading dimension of V.
 * @param[in]     T       The triangular k-by-k matrix T in the representation
 *                        of the block reflector.
 * @param[in]     ldt     The leading dimension of T. ldt >= k.
 * @param[in,out] A       On entry, the k-by-n or m-by-k matrix A.
 *                        On exit, A is overwritten by the corresponding block
 *                        of H*C or H**H*C or C*H or C*H**H.
 * @param[in]     lda     The leading dimension of A.
 * @param[in,out] B       On entry, the m-by-n matrix B.
 *                        On exit, B is overwritten by the corresponding block
 *                        of H*C or H**H*C or C*H or C*H**H.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,m).
 * @param[out]    work    Workspace array.
 * @param[in]     ldwork  The leading dimension of work.
 */
void ztprfb(const char* side, const char* trans, const char* direct,
            const char* storev, const int m, const int n, const int k,
            const int l, const double complex* const restrict V, const int ldv,
            const double complex* const restrict T, const int ldt,
            double complex* const restrict A, const int lda,
            double complex* const restrict B, const int ldb,
            double complex* const restrict work, const int ldwork)
{
    const double complex ONE = CMPLX(1.0, 0.0);
    const double complex ZERO = CMPLX(0.0, 0.0);
    const double complex NEG_ONE = CMPLX(-1.0, 0.0);
    int i, j, mp, np, kp;
    int left, right, forward, backward, column, row;

    if (m <= 0 || n <= 0 || k <= 0 || l < 0) return;

    if (storev[0] == 'C' || storev[0] == 'c') {
        column = 1;
        row = 0;
    } else if (storev[0] == 'R' || storev[0] == 'r') {
        column = 0;
        row = 1;
    } else {
        column = 0;
        row = 0;
    }

    if (side[0] == 'L' || side[0] == 'l') {
        left = 1;
        right = 0;
    } else if (side[0] == 'R' || side[0] == 'r') {
        left = 0;
        right = 1;
    } else {
        left = 0;
        right = 0;
    }

    if (direct[0] == 'F' || direct[0] == 'f') {
        forward = 1;
        backward = 0;
    } else if (direct[0] == 'B' || direct[0] == 'b') {
        forward = 0;
        backward = 1;
    } else {
        forward = 0;
        backward = 0;
    }

    if (column && forward && left) {
        /*
         * Let  W =  [ I ]    (K-by-K)
         *           [ V ]    (M-by-K)
         *
         * Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
         *                                   [ B ]  (M-by-N)
         *
         * H = I - W T W**H          or  H**H = I - W T**H W**H
         *
         * A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
         * B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
         */
        mp = (m - l + 1) < m ? (m - l + 1) : m;  /* MIN(M-L+1, M) */
        kp = (l + 1) < k ? (l + 1) : k;          /* MIN(L+1, K) */

        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                work[i + j * ldwork] = B[(m - l + i) + j * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, l, n, &ONE, &V[(mp - 1)], ldv, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    l, n, m - l, &ONE, V, ldv, B, ldb, &ONE, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    k - l, n, m, &ONE, &V[(kp - 1) * ldv], ldv,
                    B, ldb, &ZERO, &work[(kp - 1)], ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, k, n, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - l, n, k, &NEG_ONE, V, ldv, work, ldwork, &ONE, B, ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    l, n, k - l, &NEG_ONE, &V[(mp - 1) + (kp - 1) * ldv], ldv,
                    &work[(kp - 1)], ldwork, &ONE, &B[(mp - 1)], ldb);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, n, &ONE, &V[(mp - 1)], ldv, work, ldwork);
        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                B[(m - l + i) + j * ldb] = B[(m - l + i) + j * ldb] - work[i + j * ldwork];
            }
        }

    } else if (column && forward && right) {
        /*
         * Let  W =  [ I ]    (K-by-K)
         *           [ V ]    (N-by-K)
         *
         * Form  C H or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
         *
         * H = I - W T W**H          or  H**H = I - W T**H W**H
         *
         * A = A - (A + B V) T      or  A = A - (A + B V) T**H
         * B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
         */
        np = (n - l + 1) < n ? (n - l + 1) : n;  /* MIN(N-L+1, N) */
        kp = (l + 1) < k ? (l + 1) : k;          /* MIN(L+1, K) */

        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = B[i + (n - l + j) * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, m, l, &ONE, &V[(np - 1)], ldv, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, l, n - l, &ONE, B, ldb, V, ldv, &ONE, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, k - l, n, &ONE, B, ldb, &V[(kp - 1) * ldv], ldv,
                    &ZERO, &work[(kp - 1) * ldwork], ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, m, k, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, n - l, k, &NEG_ONE, work, ldwork, V, ldv, &ONE, B, ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, l, k - l, &NEG_ONE, &work[(kp - 1) * ldwork], ldwork,
                    &V[(np - 1) + (kp - 1) * ldv], ldv, &ONE, &B[(np - 1) * ldb], ldb);
        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans,
                    CblasNonUnit, m, l, &ONE, &V[(np - 1)], ldv, work, ldwork);
        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                B[i + (n - l + j) * ldb] = B[i + (n - l + j) * ldb] - work[i + j * ldwork];
            }
        }

    } else if (column && backward && left) {
        /*
         * Let  W =  [ V ]    (M-by-K)
         *           [ I ]    (K-by-K)
         *
         * Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
         *                                   [ A ]  (K-by-N)
         *
         * H = I - W T W**H          or  H**H = I - W T**H W**H
         *
         * A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
         * B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
         */
        mp = (l + 1) < m ? (l + 1) : m;          /* MIN(L+1, M) */
        kp = (k - l + 1) < k ? (k - l + 1) : k;  /* MIN(K-L+1, K) */

        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                work[(k - l + i) + j * ldwork] = B[i + j * ldb];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                    CblasNonUnit, l, n, &ONE, &V[(kp - 1) * ldv], ldv,
                    &work[(kp - 1)], ldwork);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    l, n, m - l, &ONE, &V[(mp - 1) + (kp - 1) * ldv], ldv,
                    &B[(mp - 1)], ldb, &ONE, &work[(kp - 1)], ldwork);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    k - l, n, m, &ONE, V, ldv, B, ldb, &ZERO, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, k, n, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - l, n, k, &NEG_ONE, &V[(mp - 1)], ldv, work, ldwork,
                    &ONE, &B[(mp - 1)], ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    l, n, k - l, &NEG_ONE, V, ldv, work, ldwork, &ONE, B, ldb);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, n, &ONE, &V[(kp - 1) * ldv], ldv,
                    &work[(kp - 1)], ldwork);
        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                B[i + j * ldb] = B[i + j * ldb] - work[(k - l + i) + j * ldwork];
            }
        }

    } else if (column && backward && right) {
        /*
         * Let  W =  [ V ]    (N-by-K)
         *           [ I ]    (K-by-K)
         *
         * Form  C H  or  C H**H  where  C = [ B A ] (B is M-by-N, A is M-by-K)
         *
         * H = I - W T W**H          or  H**H = I - W T**H W**H
         *
         * A = A - (A + B V) T      or  A = A - (A + B V) T**H
         * B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
         */
        np = (l + 1) < n ? (l + 1) : n;          /* MIN(L+1, N) */
        kp = (k - l + 1) < k ? (k - l + 1) : k;  /* MIN(K-L+1, K) */

        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                work[i + (k - l + j) * ldwork] = B[i + j * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, m, l, &ONE, &V[(kp - 1) * ldv], ldv,
                    &work[(kp - 1) * ldwork], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, l, n - l, &ONE, &B[(np - 1) * ldb], ldb,
                    &V[(np - 1) + (kp - 1) * ldv], ldv, &ONE, &work[(kp - 1) * ldwork], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, k - l, n, &ONE, B, ldb, V, ldv, &ZERO, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, m, k, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, n - l, k, &NEG_ONE, work, ldwork, &V[(np - 1)], ldv,
                    &ONE, &B[(np - 1) * ldb], ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, l, k - l, &NEG_ONE, work, ldwork, V, ldv, &ONE, B, ldb);
        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                    CblasNonUnit, m, l, &ONE, &V[(kp - 1) * ldv], ldv,
                    &work[(kp - 1) * ldwork], ldwork);
        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                B[i + j * ldb] = B[i + j * ldb] - work[i + (k - l + j) * ldwork];
            }
        }

    } else if (row && forward && left) {
        /*
         * Let  W =  [ I V ] ( I is K-by-K, V is K-by-M )
         *
         * Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
         *                                   [ B ]  (M-by-N)
         *
         * H = I - W**H T W          or  H**H = I - W**H T**H W
         *
         * A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
         * B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
         */
        mp = (m - l + 1) < m ? (m - l + 1) : m;  /* MIN(M-L+1, M) */
        kp = (l + 1) < k ? (l + 1) : k;          /* MIN(L+1, K) */

        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                work[i + j * ldwork] = B[(m - l + i) + j * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, l, n, &ONE, &V[(mp - 1) * ldv], ldv, work, ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    l, n, m - l, &ONE, V, ldv, B, ldb, &ONE, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    k - l, n, m, &ONE, &V[(kp - 1)], ldv, B, ldb,
                    &ZERO, &work[(kp - 1)], ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, k, n, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m - l, n, k, &NEG_ONE, V, ldv, work, ldwork, &ONE, B, ldb);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    l, n, k - l, &NEG_ONE, &V[(kp - 1) + (mp - 1) * ldv], ldv,
                    &work[(kp - 1)], ldwork, &ONE, &B[(mp - 1)], ldb);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                    CblasNonUnit, l, n, &ONE, &V[(mp - 1) * ldv], ldv, work, ldwork);
        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                B[(m - l + i) + j * ldb] = B[(m - l + i) + j * ldb] - work[i + j * ldwork];
            }
        }

    } else if (row && forward && right) {
        /*
         * Let  W =  [ I V ] ( I is K-by-K, V is K-by-N )
         *
         * Form  C H  or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
         *
         * H = I - W**H T W            or  H**H = I - W**H T**H W
         *
         * A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
         * B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
         */
        np = (n - l + 1) < n ? (n - l + 1) : n;  /* MIN(N-L+1, N) */
        kp = (l + 1) < k ? (l + 1) : k;          /* MIN(L+1, K) */

        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = B[i + (n - l + j) * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                    CblasNonUnit, m, l, &ONE, &V[(np - 1) * ldv], ldv, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, l, n - l, &ONE, B, ldb, V, ldv, &ONE, work, ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, k - l, n, &ONE, B, ldb, &V[(kp - 1)], ldv,
                    &ZERO, &work[(kp - 1) * ldwork], ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, m, k, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n - l, k, &NEG_ONE, work, ldwork, V, ldv, &ONE, B, ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, l, k - l, &NEG_ONE, &work[(kp - 1) * ldwork], ldwork,
                    &V[(kp - 1) + (np - 1) * ldv], ldv, &ONE, &B[(np - 1) * ldb], ldb);
        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, m, l, &ONE, &V[(np - 1) * ldv], ldv, work, ldwork);
        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                B[i + (n - l + j) * ldb] = B[i + (n - l + j) * ldb] - work[i + j * ldwork];
            }
        }

    } else if (row && backward && left) {
        /*
         * Let  W =  [ V I ] ( I is K-by-K, V is K-by-M )
         *
         * Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
         *                                   [ A ]  (K-by-N)
         *
         * H = I - W**H T W          or  H**H = I - W**H T**H W
         *
         * A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
         * B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
         */
        mp = (l + 1) < m ? (l + 1) : m;          /* MIN(L+1, M) */
        kp = (k - l + 1) < k ? (k - l + 1) : k;  /* MIN(K-L+1, K) */

        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                work[(k - l + i) + j * ldwork] = B[i + j * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, l, n, &ONE, &V[(kp - 1)], ldv,
                    &work[(kp - 1)], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    l, n, m - l, &ONE, &V[(kp - 1) + (mp - 1) * ldv], ldv,
                    &B[(mp - 1)], ldb, &ONE, &work[(kp - 1)], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    k - l, n, m, &ONE, V, ldv, B, ldb, &ZERO, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, k, n, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < n; j++) {
            for (i = 0; i < k; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m - l, n, k, &NEG_ONE, &V[(mp - 1) * ldv], ldv, work, ldwork,
                    &ONE, &B[(mp - 1)], ldb);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    l, n, k - l, &NEG_ONE, V, ldv, work, ldwork, &ONE, B, ldb);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, l, n, &ONE, &V[(kp - 1)], ldv,
                    &work[(kp - 1)], ldwork);
        for (j = 0; j < n; j++) {
            for (i = 0; i < l; i++) {
                B[i + j * ldb] = B[i + j * ldb] - work[(k - l + i) + j * ldwork];
            }
        }

    } else if (row && backward && right) {
        /*
         * Let  W =  [ V I ] ( I is K-by-K, V is K-by-N )
         *
         * Form  C H  or  C H**H  where  C = [ B A ] (A is M-by-K, B is M-by-N)
         *
         * H = I - W**H T W            or  H**H = I - W**H T**H W
         *
         * A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
         * B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
         */
        np = (l + 1) < n ? (l + 1) : n;          /* MIN(L+1, N) */
        kp = (k - l + 1) < k ? (k - l + 1) : k;  /* MIN(K-L+1, K) */

        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                work[i + (k - l + j) * ldwork] = B[i + j * ldb];
            }
        }
        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans,
                    CblasNonUnit, m, l, &ONE, &V[(kp - 1)], ldv,
                    &work[(kp - 1) * ldwork], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, l, n - l, &ONE, &B[(np - 1) * ldb], ldb,
                    &V[(kp - 1) + (np - 1) * ldv], ldv, &ONE, &work[(kp - 1) * ldwork], ldwork);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m, k - l, n, &ONE, B, ldb, V, ldv, &ZERO, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                work[i + j * ldwork] = work[i + j * ldwork] + A[i + j * lda];
            }
        }

        cblas_ztrmm(CblasColMajor, CblasRight, CblasLower,
                    (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasConjTrans,
                    CblasNonUnit, m, k, &ONE, T, ldt, work, ldwork);

        for (j = 0; j < k; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
            }
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n - l, k, &NEG_ONE, work, ldwork, &V[(np - 1) * ldv], ldv,
                    &ONE, &B[(np - 1) * ldb], ldb);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, l, k - l, &NEG_ONE, work, ldwork, V, ldv, &ONE, B, ldb);
        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, m, l, &ONE, &V[(kp - 1)], ldv,
                    &work[(kp - 1) * ldwork], ldwork);
        for (j = 0; j < l; j++) {
            for (i = 0; i < m; i++) {
                B[i + j * ldb] = B[i + j * ldb] - work[i + (k - l + j) * ldwork];
            }
        }
    }
}
