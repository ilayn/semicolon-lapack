/**
 * @file zlarhs.c
 * @brief ZLARHS generates right-hand sides for linear system tests.
 *
 * Port of LAPACK TESTING/LIN/zlarhs.f
 */

#include <string.h>
#include <math.h>
#include "test_rng.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZLARHS chooses a set of NRHS random solution vectors and sets
 * up the right hand sides for the linear system
 *    op(A) * X = B,
 * where op(A) = A, A**T, or A**H, depending on TRANS.
 *
 * @param[in] path   The type of the matrix A. First char is precision ('Z'),
 *                   next two chars are matrix type ('GE', 'PO', 'HE', etc.)
 * @param[in] xtype  'N': generate new random X; 'C': use X on entry.
 * @param[in] uplo   For symmetric/Hermitian/triangular: 'U' or 'L'.
 * @param[in] trans  'N': B := A * X; 'T': B := A**T * X; 'C': B := A**H * X.
 * @param[in] m      Number of rows of A.
 * @param[in] n      Number of columns of A.
 * @param[in] kl     Number of subdiagonals (for banded matrices).
 * @param[in] ku     Number of superdiagonals (for banded matrices).
 * @param[in] nrhs   Number of right-hand side vectors.
 * @param[in] A      The test matrix, dimension (lda, n).
 * @param[in] lda    Leading dimension of A.
 * @param[in,out] X  On entry, if xtype='C', contains exact solution.
 *                   On exit, if xtype='N', initialized with random values.
 *                   Dimension (ldx, nrhs).
 * @param[in] ldx    Leading dimension of X.
 * @param[out] B     The right-hand side vectors, dimension (ldb, nrhs).
 * @param[in] ldb    Leading dimension of B.
 * @param[out] info  = 0: success; < 0: -i means argument i is invalid.
 * @param[in,out] state  RNG state, modified on exit.
 */
void zlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const INT m, const INT n, const INT kl,
            const INT ku, const INT nrhs, const c128* A, const INT lda,
            c128* X, const INT ldx, c128* B, const INT ldb,
            INT* info, uint64_t state[static 4])
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT tran, notran, gen, sym, tri, qrs, band;
    char c2[3];
    INT j, mb, nx;

    c2[0] = path[1];
    c2[1] = path[2];
    c2[2] = '\0';

    tran = (trans[0] == 'T' || trans[0] == 't' ||
            trans[0] == 'C' || trans[0] == 'c');
    notran = !tran;
    gen = (path[1] == 'G' || path[1] == 'g');
    qrs = (path[1] == 'Q' || path[1] == 'q' ||
           path[2] == 'Q' || path[2] == 'q');
    sym = (path[1] == 'P' || path[1] == 'p' ||
           path[1] == 'S' || path[1] == 's' ||
           path[1] == 'H' || path[1] == 'h');
    tri = (path[1] == 'T' || path[1] == 't');
    band = (path[2] == 'B' || path[2] == 'b');

    *info = 0;
    if (!(path[0] == 'Z' || path[0] == 'z')) {
        *info = -1;
    } else if (!(xtype[0] == 'N' || xtype[0] == 'n' ||
                 xtype[0] == 'C' || xtype[0] == 'c')) {
        *info = -2;
    } else if ((sym || tri) &&
               !(uplo[0] == 'U' || uplo[0] == 'u' ||
                 uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if ((gen || qrs) && !tran &&
               !(trans[0] == 'N' || trans[0] == 'n')) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (band && kl < 0) {
        *info = -7;
    } else if (band && ku < 0) {
        *info = -8;
    } else if (nrhs < 0) {
        *info = -9;
    } else if ((!band && lda < (m > 1 ? m : 1)) ||
               (band && (sym || tri) && lda < kl + 1) ||
               (band && gen && lda < kl + ku + 1)) {
        *info = -11;
    } else if ((notran && ldx < (n > 1 ? n : 1)) ||
               (tran && ldx < (m > 1 ? m : 1))) {
        *info = -13;
    } else if ((notran && ldb < (m > 1 ? m : 1)) ||
               (tran && ldb < (n > 1 ? n : 1))) {
        *info = -15;
    }

    if (*info != 0) {
        xerbla("ZLARHS", -(*info));
        return;
    }

    if (tran) {
        nx = m;
        mb = n;
    } else {
        nx = n;
        mb = m;
    }

    if (!(xtype[0] == 'C' || xtype[0] == 'c')) {
        for (j = 0; j < nrhs; j++) {
            zlarnv_rng(2, n, &X[j * ldx], state);
        }
    }

    /* Determine CBLAS transpose enum */
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        cblas_trans = CblasConjTrans;
    } else {
        cblas_trans = CblasNoTrans;
    }

    CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    if (strcmp(c2, "GE") == 0 || strcmp(c2, "QR") == 0 ||
        strcmp(c2, "LQ") == 0 || strcmp(c2, "QL") == 0 ||
        strcmp(c2, "RQ") == 0) {

        cblas_zgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                    mb, nrhs, nx, &CONE, A, lda, X, ldx, &CZERO, B, ldb);

    } else if (strcmp(c2, "PO") == 0 || strcmp(c2, "HE") == 0) {

        cblas_zhemm(CblasColMajor, CblasLeft, uploC,
                    n, nrhs, &CONE, A, lda, X, ldx, &CZERO, B, ldb);

    } else if (strcmp(c2, "SY") == 0) {

        cblas_zsymm(CblasColMajor, CblasLeft, uploC,
                    n, nrhs, &CONE, A, lda, X, ldx, &CZERO, B, ldb);

    } else if (strcmp(c2, "GB") == 0) {

        for (j = 0; j < nrhs; j++) {
            cblas_zgbmv(CblasColMajor, cblas_trans, m, n, kl, ku,
                        &CONE, A, lda, &X[j * ldx], 1,
                        &CZERO, &B[j * ldb], 1);
        }

    } else if (strcmp(c2, "PB") == 0 || strcmp(c2, "HB") == 0) {

        for (j = 0; j < nrhs; j++) {
            cblas_zhbmv(CblasColMajor, uploC, n, kl,
                        &CONE, A, lda, &X[j * ldx], 1,
                        &CZERO, &B[j * ldb], 1);
        }

    } else if (strcmp(c2, "PP") == 0 || strcmp(c2, "HP") == 0) {

        for (j = 0; j < nrhs; j++) {
            cblas_zhpmv(CblasColMajor, uploC, n,
                        &CONE, A, &X[j * ldx], 1,
                        &CZERO, &B[j * ldb], 1);
        }

    } else if (strcmp(c2, "SP") == 0) {

        for (j = 0; j < nrhs; j++) {
            zspmv(uplo, n, CONE, A, &X[j * ldx], 1,
                  CZERO, &B[j * ldb], 1);
        }

    } else if (strcmp(c2, "TR") == 0) {

        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(c128));
        }
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        cblas_ztrmm(CblasColMajor, CblasLeft, uploC, cblas_trans, diagC,
                    n, nrhs, &CONE, A, lda, B, ldb);

    } else if (strcmp(c2, "TP") == 0) {

        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(c128));
        }
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        for (j = 0; j < nrhs; j++) {
            cblas_ztpmv(CblasColMajor, uploC, cblas_trans, diagC,
                        n, A, &B[j * ldb], 1);
        }

    } else if (strcmp(c2, "TB") == 0) {

        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(c128));
        }
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        for (j = 0; j < nrhs; j++) {
            cblas_ztbmv(CblasColMajor, uploC, cblas_trans, diagC,
                        n, kl, A, lda, &B[j * ldb], 1);
        }

    } else {
        *info = -1;
        xerbla("ZLARHS", -(*info));
    }
}
