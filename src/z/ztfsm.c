/**
 * @file ztfsm.c
 * @brief ZTFSM solves a matrix equation (one operand is a triangular matrix in RFP format).
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * Level 3 BLAS like routine for A in RFP Format.
 *
 * ZTFSM solves the matrix equation
 *
 *    op( A )*X = alpha*B  or  X*op( A ) = alpha*B
 *
 * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 * non-unit, upper or lower triangular matrix and op( A ) is one of
 *
 *    op( A ) = A   or   op( A ) = A**H.
 *
 * A is in Rectangular Full Packed (RFP) Format.
 *
 * The matrix X is overwritten on B.
 *
 * @param[in] transr
 *          = 'N':  The Normal Form of RFP A is stored;
 *          = 'C':  The Conjugate-transpose Form of RFP A is stored.
 *
 * @param[in] side
 *          On entry, SIDE specifies whether op( A ) appears on the left
 *          or right of X as follows:
 *             SIDE = 'L' or 'l'   op( A )*X = alpha*B.
 *             SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
 *
 * @param[in] uplo
 *          On entry, UPLO specifies whether the RFP matrix A came from
 *          an upper or lower triangular matrix as follows:
 *          UPLO = 'U' or 'u' RFP A came from an upper triangular matrix
 *          UPLO = 'L' or 'l' RFP A came from a lower triangular matrix
 *
 * @param[in] trans
 *          On entry, TRANS specifies the form of op( A ) to be used
 *          in the matrix multiplication as follows:
 *             TRANS = 'N' or 'n'   op( A ) = A.
 *             TRANS = 'C' or 'c'   op( A ) = conjg( A' ).
 *
 * @param[in] diag
 *          On entry, DIAG specifies whether or not RFP A is unit
 *          triangular as follows:
 *             DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *             DIAG = 'N' or 'n'   A is not assumed to be unit triangular.
 *
 * @param[in] m
 *          On entry, M specifies the number of rows of B. M must be at
 *          least zero.
 *
 * @param[in] n
 *          On entry, N specifies the number of columns of B. N must be
 *          at least zero.
 *
 * @param[in] alpha
 *          On entry, ALPHA specifies the scalar alpha. When alpha is
 *          zero then A is not referenced and B need not be set before
 *          entry.
 *
 * @param[in] A
 *          Complex*16 array, dimension (NT)
 *          NT = N*(N+1)/2 if SIDE='R' and NT = M*(M+1)/2 otherwise.
 *          On entry, the matrix A in RFP Format.
 *
 * @param[in,out] B
 *          Complex*16 array, dimension (LDB,N)
 *          Before entry, the leading m by n part of the array B must
 *          contain the right-hand side matrix B, and on exit is
 *          overwritten by the solution matrix X.
 *
 * @param[in] ldb
 *          On entry, LDB specifies the first dimension of B as declared
 *          in the calling (sub) program. LDB must be at least max( 1, m ).
 */
void ztfsm(
    const char* transr,
    const char* side,
    const char* uplo,
    const char* trans,
    const char* diag,
    const INT m,
    const INT n,
    const c128 alpha,
    const c128* restrict A,
    c128* restrict B,
    const INT ldb)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

    INT lower, lside, misodd, nisodd, normaltransr, notrans;
    INT m1, m2, n1, n2, k, info;

    CBLAS_DIAG cblas_diag;

    info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lside = (side[0] == 'L' || side[0] == 'l');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    notrans = (trans[0] == 'N' || trans[0] == 'n');

    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
        info = -1;
    } else if (!lside && !(side[0] == 'R' || side[0] == 'r')) {
        info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        info = -3;
    } else if (!notrans && !(trans[0] == 'C' || trans[0] == 'c')) {
        info = -4;
    } else if (!(diag[0] == 'N' || diag[0] == 'n') &&
               !(diag[0] == 'U' || diag[0] == 'u')) {
        info = -5;
    } else if (m < 0) {
        info = -6;
    } else if (n < 0) {
        info = -7;
    } else if (ldb < (m > 1 ? m : 1)) {
        info = -11;
    }
    if (info != 0) {
        xerbla("ZTFSM ", -info);
        return;
    }

    if ((m == 0) || (n == 0)) {
        return;
    }

    if (alpha == CZERO) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < m; i++) {
                B[i + j * ldb] = CZERO;
            }
        }
        return;
    }

    cblas_diag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

    if (lside) {

        if (m % 2 == 0) {
            misodd = 0;
            k = m / 2;
        } else {
            misodd = 1;
            if (lower) {
                m2 = m / 2;
                m1 = m - m2;
            } else {
                m1 = m / 2;
                m2 = m - m1;
            }
        }

        if (misodd) {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        if (m == 1) {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                        m1, n, &alpha, A, m, B, ldb);
                        } else {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                        m1, n, &alpha, &A[0], m, B, ldb);
                            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m2, n, m1, &NEG_CONE, &A[m1], m, B, ldb, &alpha, &B[m1], ldb);
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                        m2, n, &CONE, &A[m], m, &B[m1], ldb);
                        }

                    } else {

                        if (m == 1) {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                        m1, n, &alpha, &A[0], m, B, ldb);
                        } else {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                        m2, n, &alpha, &A[m], m, &B[m1], ldb);
                            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                        m1, n, m2, &NEG_CONE, &A[m1], m, &B[m1], ldb, &alpha, B, ldb);
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                        m1, n, &CONE, &A[0], m, B, ldb);
                        }

                    }

                } else {

                    if (!notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    m1, n, &alpha, &A[m2], m, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    m2, n, m1, &NEG_CONE, &A[0], m, B, ldb, &alpha, &B[m1], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    m2, n, &CONE, &A[m1], m, &B[m1], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    m2, n, &alpha, &A[m1], m, &B[m1], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m1, n, m2, &NEG_CONE, &A[0], m, &B[m1], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    m1, n, &CONE, &A[m2], m, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        if (m == 1) {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                        m1, n, &alpha, &A[0], m1, B, ldb);
                        } else {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                        m1, n, &alpha, &A[0], m1, B, ldb);
                            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                        m2, n, m1, &NEG_CONE, &A[m1 * m1], m1, B, ldb, &alpha, &B[m1], ldb);
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                        m2, n, &CONE, &A[1], m1, &B[m1], ldb);
                        }

                    } else {

                        if (m == 1) {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                        m1, n, &alpha, &A[0], m1, B, ldb);
                        } else {
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                        m2, n, &alpha, &A[1], m1, &B[m1], ldb);
                            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m1, n, m2, &NEG_CONE, &A[m1 * m1], m1, &B[m1], ldb, &alpha, B, ldb);
                            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                        m1, n, &CONE, &A[0], m1, B, ldb);
                        }

                    }

                } else {

                    if (!notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    m1, n, &alpha, &A[m2 * m2], m2, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m2, n, m1, &NEG_CONE, &A[0], m2, B, ldb, &alpha, &B[m1], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    m2, n, &CONE, &A[m1 * m2], m2, &B[m1], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    m2, n, &alpha, &A[m1 * m2], m2, &B[m1], ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    m1, n, m2, &NEG_CONE, &A[0], m2, &B[m1], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    m1, n, &CONE, &A[m2 * m2], m2, B, ldb);

                    }

                }

            }

        } else {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    k, n, &alpha, &A[1], m + 1, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[k + 1], m + 1, B, ldb, &alpha, &B[k], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    k, n, &CONE, &A[0], m + 1, &B[k], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    k, n, &alpha, &A[0], m + 1, &B[k], ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[k + 1], m + 1, &B[k], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    k, n, &CONE, &A[1], m + 1, B, ldb);

                    }

                } else {

                    if (!notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    k, n, &alpha, &A[k + 1], m + 1, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[0], m + 1, B, ldb, &alpha, &B[k], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    k, n, &CONE, &A[k], m + 1, &B[k], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    k, n, &alpha, &A[k], m + 1, &B[k], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[0], m + 1, &B[k], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    k, n, &CONE, &A[k + 1], m + 1, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    k, n, &alpha, &A[k], k, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[k * (k + 1)], k, B, ldb, &alpha, &B[k], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    k, n, &CONE, &A[0], k, &B[k], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    k, n, &alpha, &A[0], k, &B[k], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[k * (k + 1)], k, &B[k], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    k, n, &CONE, &A[k], k, B, ldb);

                    }

                } else {

                    if (!notrans) {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, cblas_diag,
                                    k, n, &alpha, &A[k * (k + 1)], k, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[0], k, B, ldb, &alpha, &B[k], ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_diag,
                                    k, n, &CONE, &A[k * k], k, &B[k], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, cblas_diag,
                                    k, n, &alpha, &A[k * k], k, &B[k], ldb);
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    k, n, k, &NEG_CONE, &A[0], k, &B[k], ldb, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_diag,
                                    k, n, &CONE, &A[k * (k + 1)], k, B, ldb);

                    }

                }

            }

        }

    } else {

        if (n % 2 == 0) {
            nisodd = 0;
            k = n / 2;
        } else {
            nisodd = 1;
            if (lower) {
                n2 = n / 2;
                n1 = n - n2;
            } else {
                n1 = n / 2;
                n2 = n - n1;
            }
        }

        if (nisodd) {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, n2, &alpha, &A[n], n, &B[n1 * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n1, n2, &NEG_CONE, &B[n1 * ldb], ldb, &A[n1], n, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, n1, &CONE, &A[0], n, B, ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, n1, &alpha, &A[0], n, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, n2, n1, &NEG_CONE, B, ldb, &A[n1], n, &alpha, &B[n1 * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, n2, &CONE, &A[n], n, &B[n1 * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, n1, &alpha, &A[n2], n, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n2, n1, &NEG_CONE, B, ldb, &A[0], n, &alpha, &B[n1 * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, n2, &CONE, &A[n1], n, &B[n1 * ldb], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, n2, &alpha, &A[n1], n, &B[n1 * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, n1, n2, &NEG_CONE, &B[n1 * ldb], ldb, &A[0], n, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, n1, &CONE, &A[n2], n, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, n2, &alpha, &A[1], n1, &B[n1 * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, n1, n2, &NEG_CONE, &B[n1 * ldb], ldb, &A[n1 * n1], n1, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, n1, &CONE, &A[0], n1, B, ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, n1, &alpha, &A[0], n1, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n2, n1, &NEG_CONE, B, ldb, &A[n1 * n1], n1, &alpha, &B[n1 * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, n2, &CONE, &A[1], n1, &B[n1 * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, n1, &alpha, &A[n2 * n2], n2, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, n2, n1, &NEG_CONE, B, ldb, &A[0], n2, &alpha, &B[n1 * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, n2, &CONE, &A[n1 * n2], n2, &B[n1 * ldb], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, n2, &alpha, &A[n1 * n2], n2, &B[n1 * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n1, n2, &NEG_CONE, &B[n1 * ldb], ldb, &A[0], n2, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, n1, &CONE, &A[n2 * n2], n2, B, ldb);

                    }

                }

            }

        } else {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, k, &alpha, &A[0], n + 1, &B[k * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, &NEG_CONE, &B[k * ldb], ldb, &A[k + 1], n + 1, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, k, &CONE, &A[1], n + 1, B, ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, k, &alpha, &A[1], n + 1, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, k, k, &NEG_CONE, B, ldb, &A[k + 1], n + 1, &alpha, &B[k * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, k, &CONE, &A[0], n + 1, &B[k * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, k, &alpha, &A[k + 1], n + 1, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, &NEG_CONE, B, ldb, &A[0], n + 1, &alpha, &B[k * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, k, &CONE, &A[k], n + 1, &B[k * ldb], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, k, &alpha, &A[k], n + 1, &B[k * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, k, k, &NEG_CONE, &B[k * ldb], ldb, &A[0], n + 1, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, k, &CONE, &A[k + 1], n + 1, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, k, &alpha, &A[0], k, &B[k * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, k, k, &NEG_CONE, &B[k * ldb], ldb, &A[(k + 1) * k], k, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, k, &CONE, &A[k], k, B, ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, k, &alpha, &A[k], k, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, &NEG_CONE, B, ldb, &A[(k + 1) * k], k, &alpha, &B[k * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, k, &CONE, &A[0], k, &B[k * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_diag,
                                    m, k, &alpha, &A[(k + 1) * k], k, B, ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    m, k, k, &NEG_CONE, B, ldb, &A[0], k, &alpha, &B[k * ldb], ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, cblas_diag,
                                    m, k, &CONE, &A[k * k], k, &B[k * ldb], ldb);

                    } else {

                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_diag,
                                    m, k, &alpha, &A[k * k], k, &B[k * ldb], ldb);
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, &NEG_CONE, &B[k * ldb], ldb, &A[0], k, &alpha, B, ldb);
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans, cblas_diag,
                                    m, k, &CONE, &A[(k + 1) * k], k, B, ldb);

                    }

                }

            }

        }

    }
}
