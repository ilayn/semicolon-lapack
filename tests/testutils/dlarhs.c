/**
 * @file dlarhs.c
 * @brief DLARHS generates right-hand sides for linear system tests.
 *
 * Port of LAPACK TESTING/LIN/dlarhs.f
 */

#include <cblas.h>
#include <string.h>
#include <math.h>
#include "test_rng.h"
#include "verify.h"

/* Local helper for generating random vectors */
static void dlarhs_dlarnv(const int idist, const int n, f64* x,
                          uint64_t state[static 4]);

/**
 * DLARHS chooses a set of NRHS random solution vectors and sets
 * up the right hand sides for the linear system
 *    op(A) * X = B,
 * where op(A) = A or A**T, depending on TRANS.
 *
 * @param[in] path   The type of the matrix A. First char is precision ('D'),
 *                   next two chars are matrix type ('GE', 'PO', 'SY', etc.)
 * @param[in] xtype  'N': generate new random X; 'C': use X on entry.
 * @param[in] uplo   For symmetric/triangular: 'U' or 'L'. Ignored for general.
 * @param[in] trans  'N': B := A * X; 'T' or 'C': B := A**T * X.
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
 * @param[in] ldx    Leading dimension of X. ldx >= max(1, n) if trans='N',
 *                   ldx >= max(1, m) if trans='T'.
 * @param[out] B     The right-hand side vectors, dimension (ldb, nrhs).
 * @param[in] ldb    Leading dimension of B. ldb >= max(1, m) if trans='N',
 *                   ldb >= max(1, n) if trans='T'.
 * @param[in,out] seed  RNG seed, modified on exit.
 * @param[out] info  = 0: success; < 0: -i means argument i is invalid.
 */
void dlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const int m, const int n, const int kl,
            const int ku, const int nrhs, const f64* A, const int lda,
            f64* X, const int ldx, f64* B, const int ldb,
            int* info, uint64_t state[static 4])
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int tran, notran, gen, sym, tri, qrs, band;
    char c2[3];
    int j, mb, nx;

    (void)uplo;  /* Not used for general matrices yet */

    /* Extract matrix type from path */
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
           path[1] == 'S' || path[1] == 's');
    tri = (path[1] == 'T' || path[1] == 't');
    band = (path[2] == 'B' || path[2] == 'b');

    /* Test the input parameters */
    *info = 0;
    if (!(path[0] == 'D' || path[0] == 'd')) {
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
    } else if (!band && lda < (m > 1 ? m : 1)) {
        *info = -11;
    } else if ((notran && ldx < (n > 1 ? n : 1)) ||
               (tran && ldx < (m > 1 ? m : 1))) {
        *info = -13;
    } else if ((notran && ldb < (m > 1 ? m : 1)) ||
               (tran && ldb < (n > 1 ? n : 1))) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    /* Initialize X to NRHS random vectors unless XTYPE = 'C' */
    if (tran) {
        nx = m;
        mb = n;
    } else {
        nx = n;
        mb = m;
    }

    if (!(xtype[0] == 'C' || xtype[0] == 'c')) {
        for (j = 0; j < nrhs; j++) {
            dlarhs_dlarnv(2, n, &X[j * ldx], state);
        }
    }

    /* Multiply X by op(A) using appropriate matrix multiply routine */

    /* General matrix (GE, QR, LQ, QL, RQ) */
    if ((c2[0] == 'G' || c2[0] == 'g') && (c2[1] == 'E' || c2[1] == 'e')) {
        /* B = op(A) * X using DGEMM */
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        cblas_dgemm(CblasColMajor, transA, CblasNoTrans,
                    mb, nrhs, nx, ONE, A, lda, X, ldx, ZERO, B, ldb);
    } else if ((c2[0] == 'G' || c2[0] == 'g') && (c2[1] == 'B' || c2[1] == 'b')) {
        /* General matrix, band storage - use DGBMV */
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        for (j = 0; j < nrhs; j++) {
            cblas_dgbmv(CblasColMajor, transA, mb, nx, kl, ku, ONE,
                        A, lda, &X[j * ldx], 1, ZERO, &B[j * ldb], 1);
        }
    } else if ((c2[0] == 'Q' || c2[0] == 'q') ||
               ((c2[1] == 'Q' || c2[1] == 'q'))) {
        /* QR, LQ, QL, RQ - treat same as general */
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        cblas_dgemm(CblasColMajor, transA, CblasNoTrans,
                    mb, nrhs, nx, ONE, A, lda, X, ldx, ZERO, B, ldb);
    } else if ((c2[0] == 'P' || c2[0] == 'p') && (c2[1] == 'O' || c2[1] == 'o')) {
        /* Symmetric positive definite - use DSYMM */
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        cblas_dsymm(CblasColMajor, CblasLeft, uploC,
                    n, nrhs, ONE, A, lda, X, ldx, ZERO, B, ldb);
    } else if (((c2[0] == 'P' || c2[0] == 'p') && (c2[1] == 'P' || c2[1] == 'p')) ||
               ((c2[0] == 'S' || c2[0] == 's') && (c2[1] == 'P' || c2[1] == 'p'))) {
        /* Symmetric packed (positive definite or indefinite) - use DSPMV */
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        for (j = 0; j < nrhs; j++) {
            cblas_dspmv(CblasColMajor, uploC, n, ONE, A,
                        &X[j * ldx], 1, ZERO, &B[j * ldb], 1);
        }
    } else if ((c2[0] == 'S' || c2[0] == 's') && (c2[1] == 'Y' || c2[1] == 'y')) {
        /* Symmetric indefinite - use DSYMM */
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        cblas_dsymm(CblasColMajor, CblasLeft, uploC,
                    n, nrhs, ONE, A, lda, X, ldx, ZERO, B, ldb);
    } else if ((c2[0] == 'T' || c2[0] == 't') && (c2[1] == 'R' || c2[1] == 'r')) {
        /* Triangular - use DTRMM (multiply in place)
         * ku encodes diagonal type: ku=2 => unit triangular, ku=1 => non-unit */
        /* First copy X to B, then multiply */
        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(f64));
        }
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        cblas_dtrmm(CblasColMajor, CblasLeft, uploC, transA, diagC,
                    n, nrhs, ONE, A, lda, B, ldb);
    } else if ((c2[0] == 'T' || c2[0] == 't') && (c2[1] == 'P' || c2[1] == 'p')) {
        /* Triangular packed - use DTPMV (one column at a time)
         * ku encodes diagonal type: ku=2 => unit triangular, ku=1 => non-unit */
        /* First copy X to B, then multiply each column */
        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(f64));
        }
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        for (j = 0; j < nrhs; j++) {
            cblas_dtpmv(CblasColMajor, uploC, transA, diagC,
                        n, A, &B[j * ldb], 1);
        }
    } else if ((c2[0] == 'T' || c2[0] == 't') && (c2[1] == 'B' || c2[1] == 'b')) {
        /* Triangular banded - use DTBMV (one column at a time)
         * ku encodes diagonal type: ku=2 => unit triangular, ku=1 => non-unit */
        for (j = 0; j < nrhs; j++) {
            memcpy(&B[j * ldb], &X[j * ldx], n * sizeof(f64));
        }
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        CBLAS_DIAG diagC = (ku == 2) ? CblasUnit : CblasNonUnit;
        for (j = 0; j < nrhs; j++) {
            cblas_dtbmv(CblasColMajor, uploC, transA, diagC,
                        n, kl, A, lda, &B[j * ldb], 1);
        }
    } else if ((c2[0] == 'P' || c2[0] == 'p') && (c2[1] == 'B' || c2[1] == 'b')) {
        /* Symmetric positive definite band - use DSBMV */
        CBLAS_UPLO uploC = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        for (j = 0; j < nrhs; j++) {
            cblas_dsbmv(CblasColMajor, uploC, n, kl, ONE, A, lda,
                        &X[j * ldx], 1, ZERO, &B[j * ldb], 1);
        }
    } else {
        /* Default: treat as general matrix */
        CBLAS_TRANSPOSE transA = tran ? CblasTrans : CblasNoTrans;
        cblas_dgemm(CblasColMajor, transA, CblasNoTrans,
                    mb, nrhs, nx, ONE, A, lda, X, ldx, ZERO, B, ldb);
    }
}

/**
 * Generate a vector of random numbers from a uniform or normal distribution.
 * Uses the global RNG state from test_rng.h.
 *
 * @param[in] idist  Distribution type:
 *                   1: uniform (0, 1)
 *                   2: uniform (-1, 1)
 *                   3: normal (0, 1)
 * @param[in] n      Length of vector.
 * @param[out] x     Output vector.
 */
static void dlarhs_dlarnv(const int idist, const int n, f64* x,
                          uint64_t state[static 4])
{
    int i;

    for (i = 0; i < n; i++) {
        if (idist == 1) {
            /* Uniform (0, 1) */
            x[i] = rng_uniform(state);
        } else if (idist == 2) {
            /* Uniform (-1, 1) */
            x[i] = rng_uniform_symmetric(state);
        } else {
            /* Normal (0, 1) */
            x[i] = rng_normal(state);
        }
    }
}
