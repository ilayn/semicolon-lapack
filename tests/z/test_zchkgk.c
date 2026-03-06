/**
 * @file test_zchkgk.c
 * @brief Tests ZGGBAK, backward balancing of matrix pair eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/zchkgk.f with embedded test data from
 * TESTING/zgbak.in.
 *
 * Verification: after ZGGBAL balances (A,B) and ZGGBAK transforms the
 * eigenvectors, check that VL'*AF*VR == VLF'*A*VRF (and same for B),
 * where AF/BF are the original matrices and VLF/VRF are pre-balance copies.
 */

#include "test_harness.h"
#include "verify.h"
#include "semicolon_cblas.h"

#define LD 50

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c128* rm, c128* cm,
                                  INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(c128));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = rm[i * ncol + j];
}

/* ---------- test case data from TESTING/zgbak.in ---------- */

typedef struct {
    INT n;
    INT m;            /* number of eigenvector columns */
    const c128* a_rm; /* n*n, row-major */
    const c128* b_rm; /* n*n, row-major */
    const c128* vl_rm; /* n*m, row-major */
    const c128* vr_rm; /* n*m, row-major */
} zgbak_case_t;

/* Case 0: N=6, M=3, diagonal pair */
static const c128 c0_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(6.0, 6.0),
};
static const c128 c0_b[] = {
    CMPLX(6.0, 6.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c0_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(3.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(4.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(5.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(6.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c0_vr[] = {
    CMPLX(-1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(-2.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(-3.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(-4.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(-5.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(-6.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};

/* Case 1: N=6, M=2, sub-diagonal + diagonal */
static const c128 c1_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c1_b[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c1_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
    CMPLX(6.0, 6.0), CMPLX(6.0, 6.0),
};
static const c128 c1_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
    CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0),
};

/* Case 2: N=6, M=3, lower triangular + same */
static const c128 c2_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0), CMPLX(6.0, 6.0),
};
static const c128 c2_b[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0), CMPLX(6.0, 6.0),
};
static const c128 c2_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
    CMPLX(6.0, 6.0), CMPLX(6.0, 6.0), CMPLX(6.0, 6.0),
};
static const c128 c2_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
    CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0),
};

/* Case 3: N=5, M=3, lower triangular + identity */
static const c128 c3_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0),
};
static const c128 c3_b[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c3_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
};
static const c128 c3_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
};

/* Case 4: N=6, M=3, 1e11 upper triangular */
static const c128 c4_a[] = {
    CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
};
static const c128 c4_b[] = {
    CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
};
static const c128 c4_vl[] = {
    CMPLX(1e4, 1e4), CMPLX(1e4, 1e4), CMPLX(1e4, 1e4),
    CMPLX(2e4, 2e4), CMPLX(2e4, 2e4), CMPLX(2e4, 2e4),
    CMPLX(3e4, 3e4), CMPLX(3e4, 3e4), CMPLX(3e4, 3e4),
    CMPLX(4e4, 4e4), CMPLX(4e4, 4e4), CMPLX(4e4, 4e4),
    CMPLX(5e4, 5e4), CMPLX(5e4, 5e4), CMPLX(5e4, 5e4),
    CMPLX(6e4, 6e4), CMPLX(6e4, 6e4), CMPLX(6e4, 6e4),
};
static const c128 c4_vr[] = {
    CMPLX(-1e4, -1e4), CMPLX(-1e4, -1e4), CMPLX(-1e4, -1e4),
    CMPLX(-2e4, -2e4), CMPLX(-2e4, -2e4), CMPLX(-2e4, -2e4),
    CMPLX(-3e4, -3e4), CMPLX(-3e4, -3e4), CMPLX(-3e4, -3e4),
    CMPLX(-4e4, -4e4), CMPLX(-4e4, -4e4), CMPLX(-4e4, -4e4),
    CMPLX(-5e4, -5e4), CMPLX(-5e4, -5e4), CMPLX(-5e4, -5e4),
    CMPLX(-6e4, -6e4), CMPLX(-6e4, -6e4), CMPLX(-6e4, -6e4),
};

/* Case 5: N=6, M=3, structured with 1e6 entries */
static const c128 c5_a[] = {
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e6, 1e6),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e-6, 1e-6),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e6, 1e6),   CMPLX(1e6, 1e6),
};
static const c128 c5_b[] = {
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e6, 1e6),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e-6, 1e-6),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e6, 1e6),   CMPLX(1e6, 1e6),
};
static const c128 c5_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
    CMPLX(6.0, 6.0), CMPLX(6.0, 6.0), CMPLX(6.0, 6.0),
};
static const c128 c5_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
    CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0),
};

/* Case 6: N=7, M=2, structured with isolation */
static const c128 c6_a[] = {
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c6_b[] = {
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c6_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
    CMPLX(6.0, 6.0), CMPLX(6.0, 6.0),
    CMPLX(7.0, 7.0), CMPLX(7.0, 7.0),
};
static const c128 c6_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
    CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0),
    CMPLX(-7.0, -7.0), CMPLX(-7.0, -7.0),
};

/* Case 7: N=7, M=3, large magnitude differences */
static const c128 c7_a[] = {
    CMPLX(0.0, 0.0),   CMPLX(1e5, 1e5),     CMPLX(0.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e-5, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e-5, 1e-5),   CMPLX(1e5, 1e5),      CMPLX(1e-4, 1e-4),   CMPLX(1e-5, 0.0),    CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),
    CMPLX(1e5, 1e5),   CMPLX(1e5, 1e5),     CMPLX(1e-5, 1e-5),    CMPLX(1e5, 1.0),     CMPLX(1e5, 1.0),     CMPLX(1e5, 1e5),     CMPLX(1e3, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e-4, 1e-5),   CMPLX(0.0, 0.0),      CMPLX(0.1, 0.1),     CMPLX(1.0, 1.0),     CMPLX(1e-4, 1e-4),   CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),   CMPLX(1e5, 1e5),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e-5, 1e-5),   CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(1e-5, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e1, 1e1),     CMPLX(0.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e-5, 1e-5),   CMPLX(1e2, 0.0),
};
static const c128 c7_b[] = {
    CMPLX(0.0, 0.0),   CMPLX(1e-2, 0.0),    CMPLX(0.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e-5, 0.0),    CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),   CMPLX(1e5, 1e5),     CMPLX(1.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(0.1, 0.0),     CMPLX(1e2, 0.0),
    CMPLX(1e5, 1e5),   CMPLX(1e2, 0.0),     CMPLX(1e3, 0.0),      CMPLX(1e3, 0.0),     CMPLX(1e-4, 1e-5),   CMPLX(1.0, 0.0),     CMPLX(1.0, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e-5, 0.0),    CMPLX(0.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),   CMPLX(1e5, 1e5),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(1e5, 1e5),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),     CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),   CMPLX(1e-4, 0.0),    CMPLX(0.0, 0.0),      CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e5, 1e5),     CMPLX(1e-4, 1e-5),
};
static const c128 c7_vl[] = {
    CMPLX(1e-5, 1e-5), CMPLX(1e-5, 1e-5), CMPLX(1e-5, 1e-5),
    CMPLX(2e-5, 2e-5), CMPLX(2e-5, 2e-5), CMPLX(2e-5, 2e-5),
    CMPLX(3e-5, 3e-5), CMPLX(3e-5, 3e-5), CMPLX(3e-5, 3e-5),
    CMPLX(4e-5, 4e-5), CMPLX(4e-5, 4e-5), CMPLX(4e-5, 4e-5),
    CMPLX(5e-5, 5e-5), CMPLX(5e-5, 5e-5), CMPLX(5e-5, 5e-5),
    CMPLX(6e-5, 6e-5), CMPLX(6e-5, 6e-5), CMPLX(6e-5, 6e-5),
    CMPLX(7e-5, 7e-5), CMPLX(7e-5, 7e-5), CMPLX(7e-5, 7e-5),
};
static const c128 c7_vr[] = {
    CMPLX(-1e-5, -1e-5), CMPLX(-1e-5, -1e-5), CMPLX(-1e-5, -1e-5),
    CMPLX(-2e-5, -2e-5), CMPLX(-2e-5, -2e-5), CMPLX(-2e-5, -2e-5),
    CMPLX(-3e-5, -3e-5), CMPLX(-3e-5, -3e-5), CMPLX(-3e-5, -3e-5),
    CMPLX(-4e-5, -4e-5), CMPLX(-4e-5, -4e-5), CMPLX(-4e-5, -4e-5),
    CMPLX(-5e-5, -5e-5), CMPLX(-5e-5, -5e-5), CMPLX(-5e-5, -5e-5),
    CMPLX(-6e-5, -6e-5), CMPLX(-6e-5, -6e-5), CMPLX(-6e-5, -6e-5),
    CMPLX(-7e-5, -7e-5), CMPLX(-7e-5, -7e-5), CMPLX(-7e-5, -7e-5),
};

/* Case 8: N=6, M=3, structured real entries */
static const c128 c8_a[] = {
    CMPLX(-20.0, 1.0),   CMPLX(-1e3, 1e3),    CMPLX(-2.0, 0.0),     CMPLX(-1e3, 0.0),    CMPLX(-10.0, 0.0),   CMPLX(-2e3, 1e3),
    CMPLX(6e-5, 0.0),    CMPLX(4.0, 0.0),     CMPLX(6e-3, 0.0),     CMPLX(200.0, 0.0),   CMPLX(3e-5, 0.0),    CMPLX(30.0, 0.0),
    CMPLX(-0.2, 0.0),    CMPLX(-300.0, 0.0),  CMPLX(-0.04, 0.0),    CMPLX(-1e3, 1e3),    CMPLX(0.0, 0.0),     CMPLX(3000.0, 1e3),
    CMPLX(6e-5, 0.0),    CMPLX(0.04, 0.0),    CMPLX(9e-3, 0.0),     CMPLX(9.0, 0.0),     CMPLX(3e-5, 0.0),    CMPLX(0.5, 0.0),
    CMPLX(0.06, 0.0),    CMPLX(50.0, 0.0),    CMPLX(8e-5, 0.0),     CMPLX(-4000.0, 0.0), CMPLX(0.08, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(1000.0, 1e3),  CMPLX(0.7, 0.0),      CMPLX(-2e3, 1e3),    CMPLX(13.0, 0.0),    CMPLX(-6e3, 1e3),
};
static const c128 c8_b[] = {
    CMPLX(-20.0, 0.0),   CMPLX(-1e3, 1e3),    CMPLX(2.0, 0.0),      CMPLX(-2e3, 0.0),    CMPLX(10.0, 0.0),    CMPLX(-1e3, 1e3),
    CMPLX(5e-5, 0.0),    CMPLX(3.0, 0.0),     CMPLX(-2e-3, 0.0),    CMPLX(400.0, 0.0),   CMPLX(-1e-5, 0.0),   CMPLX(30.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(-100.0, 0.0),  CMPLX(-0.08, 0.0),    CMPLX(2e3, 0.0),     CMPLX(-0.4, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(5e-5, 0.0),    CMPLX(0.03, 0.0),    CMPLX(2e-3, 0.0),     CMPLX(4.0, 0.0),     CMPLX(2e-5, 0.0),    CMPLX(0.1, 0.0),
    CMPLX(0.04, 0.0),    CMPLX(30.0, 0.0),    CMPLX(-1e-5, 0.0),    CMPLX(3000.0, 0.0),  CMPLX(-0.01, 0.0),   CMPLX(600.0, 0.0),
    CMPLX(-1.0, 0.0),    CMPLX(0.0, 0.0),     CMPLX(0.4, 0.0),      CMPLX(-1e3, 1e3),    CMPLX(4.0, 0.0),     CMPLX(2e3, 0.0),
};
static const c128 c8_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c8_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
};

/* Case 9: N=6, M=3, structured with isolation */
static const c128 c9_a[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
};
static const c128 c9_b[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
};
static const c128 c9_vl[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(2.0, 2.0), CMPLX(2.0, 2.0), CMPLX(2.0, 2.0),
    CMPLX(3.0, 3.0), CMPLX(3.0, 3.0), CMPLX(3.0, 3.0),
    CMPLX(4.0, 4.0), CMPLX(4.0, 4.0), CMPLX(4.0, 4.0),
    CMPLX(5.0, 5.0), CMPLX(5.0, 5.0), CMPLX(5.0, 5.0),
    CMPLX(6.0, 6.0), CMPLX(6.0, 6.0), CMPLX(6.0, 6.0),
};
static const c128 c9_vr[] = {
    CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0), CMPLX(-1.0, -1.0),
    CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0), CMPLX(-2.0, -2.0),
    CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0), CMPLX(-3.0, -3.0),
    CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0), CMPLX(-4.0, -4.0),
    CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0), CMPLX(-5.0, -5.0),
    CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0), CMPLX(-6.0, -6.0),
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

    f64 eps = dlamch("P");

    c128 a[LD * LD], b[LD * LD], af[LD * LD], bf[LD * LD];
    c128 vl[LD * LD], vr[LD * LD], vlf[LD * LD], vrf[LD * LD];
    c128 e[LD * LD], f[LD * LD], work[LD * LD];
    f64 lscale[LD], rscale[LD], rwork[6 * LD];
    INT ilo, ihi, info;
    f64 rmax = 0.0, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax[4] = {0, 0, 0, 0};

    const c128 cone = CMPLX(1.0, 0.0);
    const c128 czero = CMPLX(0.0, 0.0);

    for (INT tc = 0; tc < NCASES; tc++) {
        const zgbak_case_t* c = &cases[tc];
        INT n = c->n;
        INT m = c->m;

        rowmajor_to_colmajor(c->a_rm, a, n, n, LD);
        rowmajor_to_colmajor(c->b_rm, b, n, n, LD);
        rowmajor_to_colmajor(c->vl_rm, vl, n, m, LD);
        rowmajor_to_colmajor(c->vr_rm, vr, n, m, LD);

        knt++;

        f64 anorm = zlange("M", n, n, a, LD, rwork);
        f64 bnorm = zlange("M", n, n, b, LD, rwork);

        zlacpy("F", n, n, a, LD, af, LD);
        zlacpy("F", n, n, b, LD, bf, LD);

        zggbal("B", n, a, LD, b, LD, &ilo, &ihi, lscale, rscale, rwork, &info);
        if (info != 0) {
            ninfo++;
            lmax[0] = knt;
        }

        zlacpy("F", n, m, vl, LD, vlf, LD);
        zlacpy("F", n, m, vr, LD, vrf, LD);

        zggbak("B", "L", n, ilo, ihi, lscale, rscale, m, vl, LD, &info);
        if (info != 0) {
            ninfo++;
            lmax[1] = knt;
        }

        zggbak("B", "R", n, ilo, ihi, lscale, rscale, m, vr, LD, &info);
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
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, af, LD, vr, LD, &czero, work, LD);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vl, LD, work, LD, &czero, e, LD);

        /* F = VLF^H * A * VRF: first work = A * VRF, then F = VLF^H * work */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, a, LD, vrf, LD, &czero, work, LD);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vlf, LD, work, LD, &czero, f, LD);

        vmax = 0.0;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f64 diff = cabs1(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        f64 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, bf, LD, vr, LD, &czero, work, LD);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vl, LD, work, LD, &czero, e, LD);

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, &cone, b, LD, vrf, LD, &czero, work, LD);
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, m, n, &cone, vlf, LD, work, LD, &czero, f, LD);

        vmax = 0.0;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f64 diff = cabs1(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }
    }

    fprintf(stderr, "ZGGBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, rmax, lmax[3]);
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
