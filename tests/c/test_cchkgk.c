/**
 * @file test_cchkgk.c
 * @brief Tests CGGBAK, backward balancing of matrix pair eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/zchkgk.f with embedded test data from
 * TESTING/zgbak.in.
 *
 * Verification: after CGGBAL balances (A,B) and CGGBAK transforms the
 * eigenvectors, check that VL'*AF*VR == VLF'*A*VRF (and same for B),
 * where AF/BF are the original matrices and VLF/VRF are pre-balance copies.
 */

#include "test_harness.h"
#include "verify.h"
#include "semicolon_cblas.h"

#define LD 50

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c64* rm, c64* cm,
                                  INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(c64));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = rm[i * ncol + j];
}

/* ---------- test case data from TESTING/zgbak.in ---------- */

typedef struct {
    INT n;
    INT m;            /* number of eigenvector columns */
    const c64* a_rm; /* n*n, row-major */
    const c64* b_rm; /* n*n, row-major */
    const c64* vl_rm; /* n*m, row-major */
    const c64* vr_rm; /* n*m, row-major */
} zgbak_case_t;

/* Case 0: N=6, M=3, diagonal pair */
static const c64 c0_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c0_b[] = {
    CMPLXF(6.0f, 6.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c0_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(3.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(4.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(5.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(6.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c0_vr[] = {
    CMPLXF(-1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(-2.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(-3.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(-4.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(-5.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(-6.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};

/* Case 1: N=6, M=2, sub-diagonal + diagonal */
static const c64 c1_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c1_b[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c1_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
    CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c1_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f),
};

/* Case 2: N=6, M=3, lower triangular + same */
static const c64 c2_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c2_b[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c2_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
    CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c2_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f),
};

/* Case 3: N=5, M=3, lower triangular + identity */
static const c64 c3_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f),
};
static const c64 c3_b[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c3_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
};
static const c64 c3_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
};

/* Case 4: N=6, M=3, 1e11 upper triangular */
static const c64 c4_a[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
};
static const c64 c4_b[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
};
static const c64 c4_vl[] = {
    CMPLXF(1e4, 1e4), CMPLXF(1e4, 1e4), CMPLXF(1e4, 1e4),
    CMPLXF(2e4, 2e4), CMPLXF(2e4, 2e4), CMPLXF(2e4, 2e4),
    CMPLXF(3e4, 3e4), CMPLXF(3e4, 3e4), CMPLXF(3e4, 3e4),
    CMPLXF(4e4, 4e4), CMPLXF(4e4, 4e4), CMPLXF(4e4, 4e4),
    CMPLXF(5e4, 5e4), CMPLXF(5e4, 5e4), CMPLXF(5e4, 5e4),
    CMPLXF(6e4, 6e4), CMPLXF(6e4, 6e4), CMPLXF(6e4, 6e4),
};
static const c64 c4_vr[] = {
    CMPLXF(-1e4, -1e4), CMPLXF(-1e4, -1e4), CMPLXF(-1e4, -1e4),
    CMPLXF(-2e4, -2e4), CMPLXF(-2e4, -2e4), CMPLXF(-2e4, -2e4),
    CMPLXF(-3e4, -3e4), CMPLXF(-3e4, -3e4), CMPLXF(-3e4, -3e4),
    CMPLXF(-4e4, -4e4), CMPLXF(-4e4, -4e4), CMPLXF(-4e4, -4e4),
    CMPLXF(-5e4, -5e4), CMPLXF(-5e4, -5e4), CMPLXF(-5e4, -5e4),
    CMPLXF(-6e4, -6e4), CMPLXF(-6e4, -6e4), CMPLXF(-6e4, -6e4),
};

/* Case 5: N=6, M=3, structured with 1e6 entries */
static const c64 c5_a[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e6, 1e6),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e-6, 1e-6),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e6, 1e6),   CMPLXF(1e6, 1e6),
};
static const c64 c5_b[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e6, 1e6),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e-6, 1e-6),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e6, 1e6),   CMPLXF(1e6, 1e6),
};
static const c64 c5_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
    CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c5_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f),
};

/* Case 6: N=7, M=2, structured with isolation */
static const c64 c6_a[] = {
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c6_b[] = {
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c6_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
    CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f),
    CMPLXF(7.0f, 7.0f), CMPLXF(7.0f, 7.0f),
};
static const c64 c6_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f),
    CMPLXF(-7.0f, -7.0f), CMPLXF(-7.0f, -7.0f),
};

/* Case 7: N=7, M=3, large magnitude differences */
static const c64 c7_a[] = {
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e5, 1e5),     CMPLXF(0.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e-5, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-5, 1e-5),   CMPLXF(1e5, 1e5),      CMPLXF(1e-4, 1e-4),   CMPLXF(1e-5, 0.0f),    CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),
    CMPLXF(1e5, 1e5),   CMPLXF(1e5, 1e5),     CMPLXF(1e-5, 1e-5),    CMPLXF(1e5, 1.0f),     CMPLXF(1e5, 1.0f),     CMPLXF(1e5, 1e5),     CMPLXF(1e3, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-4, 1e-5),   CMPLXF(0.0f, 0.0f),      CMPLXF(0.1f, 0.1f),     CMPLXF(1.0f, 1.0f),     CMPLXF(1e-4, 1e-4),   CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e5, 1e5),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-5, 1e-5),   CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(1e-5, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e1, 1e1),     CMPLXF(0.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e-5, 1e-5),   CMPLXF(1e2, 0.0f),
};
static const c64 c7_b[] = {
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-2, 0.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e-5, 0.0f),    CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e5, 1e5),     CMPLXF(1.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(0.1f, 0.0f),     CMPLXF(1e2, 0.0f),
    CMPLXF(1e5, 1e5),   CMPLXF(1e2, 0.0f),     CMPLXF(1e3, 0.0f),      CMPLXF(1e3, 0.0f),     CMPLXF(1e-4, 1e-5),   CMPLXF(1.0f, 0.0f),     CMPLXF(1.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-5, 0.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e5, 1e5),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e5, 1e5),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),   CMPLXF(1e-4, 0.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e5, 1e5),     CMPLXF(1e-4, 1e-5),
};
static const c64 c7_vl[] = {
    CMPLXF(1e-5, 1e-5), CMPLXF(1e-5, 1e-5), CMPLXF(1e-5, 1e-5),
    CMPLXF(2e-5, 2e-5), CMPLXF(2e-5, 2e-5), CMPLXF(2e-5, 2e-5),
    CMPLXF(3e-5, 3e-5), CMPLXF(3e-5, 3e-5), CMPLXF(3e-5, 3e-5),
    CMPLXF(4e-5, 4e-5), CMPLXF(4e-5, 4e-5), CMPLXF(4e-5, 4e-5),
    CMPLXF(5e-5, 5e-5), CMPLXF(5e-5, 5e-5), CMPLXF(5e-5, 5e-5),
    CMPLXF(6e-5, 6e-5), CMPLXF(6e-5, 6e-5), CMPLXF(6e-5, 6e-5),
    CMPLXF(7e-5, 7e-5), CMPLXF(7e-5, 7e-5), CMPLXF(7e-5, 7e-5),
};
static const c64 c7_vr[] = {
    CMPLXF(-1e-5, -1e-5), CMPLXF(-1e-5, -1e-5), CMPLXF(-1e-5, -1e-5),
    CMPLXF(-2e-5, -2e-5), CMPLXF(-2e-5, -2e-5), CMPLXF(-2e-5, -2e-5),
    CMPLXF(-3e-5, -3e-5), CMPLXF(-3e-5, -3e-5), CMPLXF(-3e-5, -3e-5),
    CMPLXF(-4e-5, -4e-5), CMPLXF(-4e-5, -4e-5), CMPLXF(-4e-5, -4e-5),
    CMPLXF(-5e-5, -5e-5), CMPLXF(-5e-5, -5e-5), CMPLXF(-5e-5, -5e-5),
    CMPLXF(-6e-5, -6e-5), CMPLXF(-6e-5, -6e-5), CMPLXF(-6e-5, -6e-5),
    CMPLXF(-7e-5, -7e-5), CMPLXF(-7e-5, -7e-5), CMPLXF(-7e-5, -7e-5),
};

/* Case 8: N=6, M=3, structured real entries */
static const c64 c8_a[] = {
    CMPLXF(-20.0f, 1.0f),   CMPLXF(-1e3, 1e3),    CMPLXF(-2.0f, 0.0f),     CMPLXF(-1e3, 0.0f),    CMPLXF(-10.0f, 0.0f),   CMPLXF(-2e3, 1e3),
    CMPLXF(6e-5, 0.0f),    CMPLXF(4.0f, 0.0f),     CMPLXF(6e-3, 0.0f),     CMPLXF(200.0f, 0.0f),   CMPLXF(3e-5, 0.0f),    CMPLXF(30.0f, 0.0f),
    CMPLXF(-0.2f, 0.0f),    CMPLXF(-300.0f, 0.0f),  CMPLXF(-0.04f, 0.0f),    CMPLXF(-1e3, 1e3),    CMPLXF(0.0f, 0.0f),     CMPLXF(3000.0f, 1e3),
    CMPLXF(6e-5, 0.0f),    CMPLXF(0.04f, 0.0f),    CMPLXF(9e-3, 0.0f),     CMPLXF(9.0f, 0.0f),     CMPLXF(3e-5, 0.0f),    CMPLXF(0.5f, 0.0f),
    CMPLXF(0.06f, 0.0f),    CMPLXF(50.0f, 0.0f),    CMPLXF(8e-5, 0.0f),     CMPLXF(-4000.0f, 0.0f), CMPLXF(0.08f, 0.0f),    CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(1000.0f, 1e3),  CMPLXF(0.7f, 0.0f),      CMPLXF(-2e3, 1e3),    CMPLXF(13.0f, 0.0f),    CMPLXF(-6e3, 1e3),
};
static const c64 c8_b[] = {
    CMPLXF(-20.0f, 0.0f),   CMPLXF(-1e3, 1e3),    CMPLXF(2.0f, 0.0f),      CMPLXF(-2e3, 0.0f),    CMPLXF(10.0f, 0.0f),    CMPLXF(-1e3, 1e3),
    CMPLXF(5e-5, 0.0f),    CMPLXF(3.0f, 0.0f),     CMPLXF(-2e-3, 0.0f),    CMPLXF(400.0f, 0.0f),   CMPLXF(-1e-5, 0.0f),   CMPLXF(30.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(-100.0f, 0.0f),  CMPLXF(-0.08f, 0.0f),    CMPLXF(2e3, 0.0f),     CMPLXF(-0.4f, 0.0f),    CMPLXF(0.0f, 0.0f),
    CMPLXF(5e-5, 0.0f),    CMPLXF(0.03f, 0.0f),    CMPLXF(2e-3, 0.0f),     CMPLXF(4.0f, 0.0f),     CMPLXF(2e-5, 0.0f),    CMPLXF(0.1f, 0.0f),
    CMPLXF(0.04f, 0.0f),    CMPLXF(30.0f, 0.0f),    CMPLXF(-1e-5, 0.0f),    CMPLXF(3000.0f, 0.0f),  CMPLXF(-0.01f, 0.0f),   CMPLXF(600.0f, 0.0f),
    CMPLXF(-1.0f, 0.0f),    CMPLXF(0.0f, 0.0f),     CMPLXF(0.4f, 0.0f),      CMPLXF(-1e3, 1e3),    CMPLXF(4.0f, 0.0f),     CMPLXF(2e3, 0.0f),
};
static const c64 c8_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c8_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
};

/* Case 9: N=6, M=3, structured with isolation */
static const c64 c9_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
};
static const c64 c9_b[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
};
static const c64 c9_vl[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f), CMPLXF(2.0f, 2.0f),
    CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f), CMPLXF(3.0f, 3.0f),
    CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f), CMPLXF(4.0f, 4.0f),
    CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f), CMPLXF(5.0f, 5.0f),
    CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c9_vr[] = {
    CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f), CMPLXF(-1.0f, -1.0f),
    CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f), CMPLXF(-2.0f, -2.0f),
    CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f), CMPLXF(-3.0f, -3.0f),
    CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f), CMPLXF(-4.0f, -4.0f),
    CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f), CMPLXF(-5.0f, -5.0f),
    CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f), CMPLXF(-6.0f, -6.0f),
};

static const zgbak_case_t cases[] = {
    { 6, 3, c0_a, c0_b, c0_vl, c0_vr },
    { 6, 2, c1_a, c1_b, c1_vl, c1_vr },
    { 6, 3, c2_a, c2_b, c2_vl, c2_vr },
    { 5, 3, c3_a, c3_b, c3_vl, c3_vr },
    { 6, 3, c4_a, c4_b, c4_vl, c4_vr },
    { 6, 3, c5_a, c5_b, c5_vl, c5_vr },
    { 7, 2, c6_a, c6_b, c6_vl, c6_vr },
    { 7, 3, c7_a, c7_b, c7_vl, c7_vr },
    { 6, 3, c8_a, c8_b, c8_vl, c8_vr },
    { 6, 3, c9_a, c9_b, c9_vl, c9_vr },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_zggbak(void** state)
{
    (void)state;

    f32 eps = slamch("P");

    c64 a[LD * LD], b[LD * LD], af[LD * LD], bf[LD * LD];
    c64 vl[LD * LD], vr[LD * LD], vlf[LD * LD], vrf[LD * LD];
    c64 e[LD * LD], f[LD * LD], work[LD * LD];
    f32 lscale[LD], rscale[LD], rwork[6 * LD];
    INT ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax[4] = {0, 0, 0, 0};

    const c64 cone = CMPLXF(1.0f, 0.0f);
    const c64 czero = CMPLXF(0.0f, 0.0f);

    for (INT tc = 0; tc < NCASES; tc++) {
        const zgbak_case_t* c = &cases[tc];
        INT n = c->n;
        INT m = c->m;

        rowmajor_to_colmajor(c->a_rm, a, n, n, LD);
        rowmajor_to_colmajor(c->b_rm, b, n, n, LD);
        rowmajor_to_colmajor(c->vl_rm, vl, n, m, LD);
        rowmajor_to_colmajor(c->vr_rm, vr, n, m, LD);

        knt++;

        f32 anorm = clange("M", n, n, a, LD, rwork);
        f32 bnorm = clange("M", n, n, b, LD, rwork);

        clacpy("F", n, n, a, LD, af, LD);
        clacpy("F", n, n, b, LD, bf, LD);

        cggbal("B", n, a, LD, b, LD, &ilo, &ihi, lscale, rscale, rwork, &info);
        if (info != 0) {
            ninfo++;
            lmax[0] = knt;
        }

        clacpy("F", n, m, vl, LD, vlf, LD);
        clacpy("F", n, m, vr, LD, vrf, LD);

        cggbak("B", "L", n, ilo, ihi, lscale, rscale, m, vl, LD, &info);
        if (info != 0) {
            ninfo++;
            lmax[1] = knt;
        }

        cggbak("B", "R", n, ilo, ihi, lscale, rscale, m, vr, LD, &info);
        if (info != 0) {
            ninfo++;
            lmax[2] = knt;
        }

        /*
         * Verify for A:
         *   E = VL^H * AF * VR    (using back-transformed vectors and original matrix)
         *   F = VLF^H * A * VRF   (using pre-balance vectors and balanced matrix)
         * These should be equal.
         */

        /* E = VL^H * AF * VR: first work = AF * VR, then E = VL^H * work */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, af, LD, vr, LD, &czero, work, LD);
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vl, LD, work, LD, &czero, e, LD);

        /* F = VLF^H * A * VRF: first work = A * VRF, then F = VLF^H * work */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, a, LD, vrf, LD, &czero, work, LD);
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vlf, LD, work, LD, &czero, f, LD);

        vmax = 0.0f;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f32 diff = cabs1f(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        f32 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }

        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, bf, LD, vr, LD, &czero, work, LD);
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vl, LD, work, LD, &czero, e, LD);

        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, b, LD, vrf, LD, &czero, work, LD);
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vlf, LD, work, LD, &czero, f, LD);

        vmax = 0.0f;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f32 diff = cabs1f(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }
    }

    fprintf(stderr, "CGGBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax[3]);
    if (ninfo > 0)
        fprintf(stderr, "  INFO errors: %d (bal=%d, bakL=%d, bakR=%d)\n",
                      ninfo, lmax[0], lmax[1], lmax[2]);

    assert_true(ninfo == 0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zggbak),
    };
    return cmocka_run_group_tests_name("zchkgk", tests, NULL, NULL);
}
