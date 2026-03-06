/**
 * @file test_cchkgl.c
 * @brief Tests CGGBAL, a routine for balancing a matrix pair (A, B).
 *
 * Port of LAPACK TESTING/EIG/zchkgl.f with embedded test data from
 * TESTING/zgbal.in.
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0f

#define LDA 20
#define LWORK (6 * LDA)

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c64* rm, c64* cm, INT n, INT ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(c64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/zgbal.in ---------- */

typedef struct {
    INT n;
    INT iloin;
    INT ihiin;
    const c64* a_rm;
    const c64* b_rm;
    const c64* ain_rm;
    const c64* bin_rm;
    const f32* lsclin;
    const f32* rsclin;
} zgbal_case_t;

/* Case 0: N=6, diagonal pair */
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
static const c64 c0_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(6.0f, 6.0f),
};
static const c64 c0_bin[] = {
    CMPLXF(6.0f, 6.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c0_ls[] = {1, 1, 2, 3, 4, 5};
static const f32 c0_rs[] = {1, 1, 2, 3, 4, 5};

/* Case 1: N=6, sub-diagonal A, identity B */
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
static const c64 c1_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c1_bin[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c1_ls[] = {1, 1, 2, 2, 1, 0};
static const f32 c1_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 2: N=6, lower triangular A, same B */
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
static const c64 c2_ain[] = {
    CMPLXF(6.0f, 6.0f), CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c2_bin[] = {
    CMPLXF(6.0f, 6.0f), CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c2_ls[] = {1, 1, 2, 2, 1, 0};
static const f32 c2_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 3: N=5, lower triangular + identity */
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
static const c64 c3_ain[] = {
    CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c3_bin[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c3_ls[] = {1, 1, 2, 1, 0};
static const f32 c3_rs[] = {1, 1, 2, 1, 0};

/* Case 4: N=6, 1e11 upper triangular pair */
static const c64 c4_a[] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
};
static const c64 c4_b[] = {
    CMPLXF(1.0f, 0.0f), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11), CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1e11, 1e11),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
};
static const c64 c4_ain[] = {
    CMPLXF(1e-5, 0.0f),  CMPLXF(1e4, 1e4),  CMPLXF(1e3, 1e3),  CMPLXF(1e1, 1e1),  CMPLXF(1e-1, 1e-1), CMPLXF(1e-3, 1e-3),
    CMPLXF(1e-3, 0.0f),  CMPLXF(1e-5, 0.0f), CMPLXF(1e5, 1e5),  CMPLXF(1e3, 1e3),  CMPLXF(1e1, 1e1),   CMPLXF(1e-1, 1e-1),
    CMPLXF(1e-1, 0.0f),  CMPLXF(1e-3, 0.0f), CMPLXF(1e-4, 0.0f), CMPLXF(1e5, 1e5),  CMPLXF(1e3, 1e3),   CMPLXF(1e1, 1e1),
    CMPLXF(1e1, 0.0f),   CMPLXF(1e-1, 0.0f), CMPLXF(1e-2, 0.0f), CMPLXF(1e-4, 0.0f), CMPLXF(1e5, 1e5),   CMPLXF(1e3, 1e3),
    CMPLXF(1e2, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e-1, 0.0f), CMPLXF(1e-3, 0.0f), CMPLXF(1e-5, 0.0f),  CMPLXF(1e4, 1e4),
    CMPLXF(1e4, 0.0f),   CMPLXF(1e2, 0.0f),  CMPLXF(1e1, 0.0f),  CMPLXF(1e-1, 0.0f), CMPLXF(1e-3, 0.0f),  CMPLXF(1e-5, 0.0f),
};
static const c64 c4_bin[] = {
    CMPLXF(1e-5, 0.0f),  CMPLXF(1e4, 1e4),  CMPLXF(1e3, 1e3),  CMPLXF(1e1, 1e1),  CMPLXF(1e-1, 1e-1), CMPLXF(1e-3, 1e-3),
    CMPLXF(1e-3, 0.0f),  CMPLXF(1e-5, 0.0f), CMPLXF(1e5, 1e5),  CMPLXF(1e3, 1e3),  CMPLXF(1e1, 1e1),   CMPLXF(1e-1, 1e-1),
    CMPLXF(1e-1, 0.0f),  CMPLXF(1e-3, 0.0f), CMPLXF(1e-4, 0.0f), CMPLXF(1e5, 1e5),  CMPLXF(1e3, 1e3),   CMPLXF(1e1, 1e1),
    CMPLXF(1e1, 0.0f),   CMPLXF(1e-1, 0.0f), CMPLXF(1e-2, 0.0f), CMPLXF(1e-4, 0.0f), CMPLXF(1e5, 1e5),   CMPLXF(1e3, 1e3),
    CMPLXF(1e2, 0.0f),   CMPLXF(1.0f, 0.0f),  CMPLXF(1e-1, 0.0f), CMPLXF(1e-3, 0.0f), CMPLXF(1e-5, 0.0f),  CMPLXF(1e4, 1e4),
    CMPLXF(1e4, 0.0f),   CMPLXF(1e2, 0.0f),  CMPLXF(1e1, 0.0f),  CMPLXF(1e-1, 0.0f), CMPLXF(1e-3, 0.0f),  CMPLXF(1e-5, 0.0f),
};
static const f32 c4_ls[] = {1e-7, 1e-5, 1e-3, 1e-1, 1, 1e2};
static const f32 c4_rs[] = {1e2, 1, 1e-1, 1e-3, 1e-5, 1e-7};

/* Case 5: N=6, structured with 1e6 entries and isolation */
static const c64 c5_a[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e6, 1e6),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e-6, 1e-6),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e6, 1e6),   CMPLXF(1e6, 1e6),
};
static const c64 c5_b[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1.0f, 0.0f),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e6, 1e6),
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),   CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-6, 1e-6), CMPLXF(1e-6, 1e-6),
    CMPLXF(1e6, 1e6),  CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e6, 1e6),   CMPLXF(1e6, 1e6),
};
static const c64 c5_ain[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 1.0f),    CMPLXF(1e-4, 1e-4), CMPLXF(1e4, 1e4),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e4, 1e4),    CMPLXF(1.0f, 1.0f),   CMPLXF(1e-4, 1e-4),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-4, 1e-4),  CMPLXF(1e4, 1e4),   CMPLXF(1.0f, 1.0f),
};
static const c64 c5_bin[] = {
    CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 0.0f),  CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 0.0f),   CMPLXF(1e-1, 0.0f),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1.0f, 1.0f),    CMPLXF(1e-4, 1e-4), CMPLXF(1e4, 1e4),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e4, 1e4),    CMPLXF(1.0f, 1.0f),   CMPLXF(1e-4, 1e-4),
    CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(1e-4, 1e-4),  CMPLXF(1e4, 1e4),   CMPLXF(1.0f, 1.0f),
};
static const f32 c5_ls[] = {3, 3, 3, 1e-1, 1e3, 1e-5};
static const f32 c5_rs[] = {1, 2, 3, 1e-5, 1e3, 1e-1};

/* Case 6: N=7, with isolation at top and bottom */
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
static const c64 c6_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c6_bin[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c6_ls[] = {2, 1, 1, 1, 1, 5, 4};
static const f32 c6_rs[] = {0, 2, 1, 1, 1, 1, 1};

/* Case 7: N=7, structured with 1e5 entries and isolation */
static const c64 c7_a[] = {
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e-3, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-5, 1e-5),  CMPLXF(1e5, 1e5),    CMPLXF(1e-4, 1e-4),  CMPLXF(1e-3, 0.0f),   CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),
    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e-3, 1e-3),  CMPLXF(1e5, 1.0f),    CMPLXF(1e5, 1.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e3, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-4, 1e-5),  CMPLXF(0.0f, 0.0f),    CMPLXF(1e-1, 1e-1),  CMPLXF(1.0f, 1.0f),    CMPLXF(1e-4, 1e-4),  CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-3, 1e-3),  CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-5, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e1, 1e1),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e-3, 1e-3),  CMPLXF(1e2, 0.0f),
};
static const c64 c7_b[] = {
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-2, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e-3, 0.0f),   CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e-1, 0.0f),   CMPLXF(1e2, 0.0f),
    CMPLXF(1e5, 1e5),    CMPLXF(1e2, 0.0f),    CMPLXF(1e3, 0.0f),    CMPLXF(1e3, 0.0f),    CMPLXF(1e-4, 1e-3),  CMPLXF(1.0f, 0.0f),    CMPLXF(1.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-5, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-4, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),    CMPLXF(1e-4, 1e-3),
};
static const c64 c7_ain[] = {
    CMPLXF(1e5, 1e5),    CMPLXF(1e-3, 1e-3),  CMPLXF(1e4, 1e4),    CMPLXF(1e2, 1e-3),   CMPLXF(1e4, 1e-1),   CMPLXF(1e3, 0.0f),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e4, 1e4),    CMPLXF(1e-7, 1e-7),  CMPLXF(1e-4, 0.0f),   CMPLXF(1e5, 1e5),    CMPLXF(1e-5, 1e-5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1.0f, 1.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1e-5, 0.0f),   CMPLXF(1e3, 1e3),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-5, 1e-5),  CMPLXF(1e-4, 1e-4),  CMPLXF(1e-1, 1e-1),  CMPLXF(1e5, 1e5),    CMPLXF(1e-4, 1e-5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-6, 1e-6),  CMPLXF(1.0f, 1.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1.0f, 0.0f),    CMPLXF(1e-1, 1e-1),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-5, 0.0f),   CMPLXF(1e-3, 1e-3),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),
};
static const c64 c7_bin[] = {
    CMPLXF(1e5, 1e5),    CMPLXF(1e3, 0.0f),    CMPLXF(1e-1, 0.0f),   CMPLXF(1.0f, 0.0f),    CMPLXF(1e-5, 1e-4),  CMPLXF(1.0f, 0.0f),    CMPLXF(1e2, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1.0f, 0.0f),    CMPLXF(1e-2, 0.0f),   CMPLXF(1e2, 1e2),    CMPLXF(1e4, 1e4),    CMPLXF(1e2, 0.0f),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1.0f, 1.0f),    CMPLXF(1e-6, 0.0f),   CMPLXF(1e3, 1e3),    CMPLXF(1e-4, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e4, 1e4),    CMPLXF(1e2, 1e2),    CMPLXF(1e4, 1e4),    CMPLXF(1e5, 1e5),    CMPLXF(1e-5, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1.0f, 1.0f),    CMPLXF(1e2, 1e2),    CMPLXF(1e-6, 1e-5),  CMPLXF(1e-6, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),    CMPLXF(1e5, 1e5),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e5, 1e5),
};
static const f32 c7_ls[] = {2, 1, 1e-2, 1, 1e-2, 5, 4};
static const f32 c7_rs[] = {0, 2, 1e-1, 1e-3, 1e-1, 1, 1};

/* Case 8: N=6, large magnitude differences */
static const c64 c8_a[] = {
    CMPLXF(-20.0f, 1.0f),  CMPLXF(-1e4, 1e4),    CMPLXF(-2.0f, 0.0f),   CMPLXF(-1e6, 0.0f),    CMPLXF(-10.0f, 0.0f),  CMPLXF(-2e5, 1e5),
    CMPLXF(6e-3, 0.0f),   CMPLXF(4.0f, 0.0f),     CMPLXF(6e-4, 0.0f),   CMPLXF(200.0f, 0.0f),   CMPLXF(3e-3, 0.0f),   CMPLXF(30.0f, 0.0f),
    CMPLXF(-0.2f, 0.0f),   CMPLXF(-300.0f, 0.0f),  CMPLXF(-0.04f, 0.0f),  CMPLXF(-1e4, 1e4),    CMPLXF(0.0f, 0.0f),    CMPLXF(3000.0f, 1000.0f),
    CMPLXF(6e-5, 0.0f),   CMPLXF(0.04f, 0.0f),    CMPLXF(9e-6, 0.0f),   CMPLXF(9.0f, 0.0f),     CMPLXF(3e-5, 0.0f),   CMPLXF(0.5f, 0.0f),
    CMPLXF(0.06f, 0.0f),   CMPLXF(50.0f, 0.0f),    CMPLXF(8e-3, 0.0f),   CMPLXF(-4000.0f, 0.0f), CMPLXF(0.08f, 0.0f),   CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1000.0f, 1000.0f), CMPLXF(0.7f, 0.0f),  CMPLXF(-2e5, 1e5),    CMPLXF(13.0f, 0.0f),   CMPLXF(-6e4, 1e4),
};
static const c64 c8_b[] = {
    CMPLXF(-20.0f, 0.0f),  CMPLXF(-1e4, 1e4),    CMPLXF(2.0f, 0.0f),    CMPLXF(-2e6, 0.0f),    CMPLXF(10.0f, 0.0f),   CMPLXF(-1e5, 1e5),
    CMPLXF(5e-3, 0.0f),   CMPLXF(3.0f, 0.0f),     CMPLXF(-2e-4, 0.0f),  CMPLXF(400.0f, 0.0f),   CMPLXF(-1e-3, 0.0f),  CMPLXF(30.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(-100.0f, 0.0f),  CMPLXF(-0.08f, 0.0f),  CMPLXF(2e4, 0.0f),     CMPLXF(-0.4f, 0.0f),   CMPLXF(0.0f, 0.0f),
    CMPLXF(5e-5, 0.0f),   CMPLXF(0.03f, 0.0f),    CMPLXF(2e-6, 0.0f),   CMPLXF(4.0f, 0.0f),     CMPLXF(2e-5, 0.0f),   CMPLXF(0.1f, 0.0f),
    CMPLXF(0.04f, 0.0f),   CMPLXF(30.0f, 0.0f),    CMPLXF(-1e-3, 0.0f),  CMPLXF(3000.0f, 0.0f),  CMPLXF(-0.01f, 0.0f),  CMPLXF(600.0f, 0.0f),
    CMPLXF(-1.0f, 0.0f),   CMPLXF(0.0f, 0.0f),     CMPLXF(0.4f, 0.0f),    CMPLXF(-1e5, 1e5),    CMPLXF(4.0f, 0.0f),    CMPLXF(2e4, 0.0f),
};
static const c64 c8_ain[] = {
    CMPLXF(-0.2f, 0.01f),  CMPLXF(-1.0f, 1.0f),   CMPLXF(-0.2f, 0.0f),   CMPLXF(-1.0f, 0.0f),    CMPLXF(-1.0f, 0.0f),   CMPLXF(-0.2f, 0.1f),
    CMPLXF(0.6f, 0.0f),    CMPLXF(4.0f, 0.0f),    CMPLXF(0.6f, 0.0f),    CMPLXF(2.0f, 0.0f),     CMPLXF(3.0f, 0.0f),    CMPLXF(0.3f, 0.0f),
    CMPLXF(-0.2f, 0.0f),   CMPLXF(-3.0f, 0.0f),   CMPLXF(-0.4f, 0.0f),   CMPLXF(-1.0f, 1.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(0.3f, 0.1f),
    CMPLXF(0.6f, 0.0f),    CMPLXF(4.0f, 0.0f),    CMPLXF(0.9f, 0.0f),    CMPLXF(9.0f, 0.0f),     CMPLXF(3.0f, 0.0f),    CMPLXF(0.5f, 0.0f),
    CMPLXF(0.6f, 0.0f),    CMPLXF(5.0f, 0.0f),    CMPLXF(0.8f, 0.0f),    CMPLXF(-4.0f, 0.0f),    CMPLXF(8.0f, 0.0f),    CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(1.0f, 1.0f),    CMPLXF(0.7f, 0.0f),    CMPLXF(-2.0f, 1.0f),    CMPLXF(13.0f, 0.0f),   CMPLXF(-0.6f, 0.1f),
};
static const c64 c8_bin[] = {
    CMPLXF(-0.2f, 0.0f),   CMPLXF(-1.0f, 1.0f),   CMPLXF(0.2f, 0.0f),    CMPLXF(-2.0f, 0.0f),    CMPLXF(1.0f, 0.0f),    CMPLXF(-0.1f, 0.1f),
    CMPLXF(0.5f, 0.0f),    CMPLXF(3.0f, 0.0f),    CMPLXF(-0.2f, 0.0f),   CMPLXF(4.0f, 0.0f),     CMPLXF(-1.0f, 0.0f),   CMPLXF(0.3f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(-1.0f, 0.0f),   CMPLXF(-0.8f, 0.0f),   CMPLXF(2.0f, 0.0f),     CMPLXF(-4.0f, 0.0f),   CMPLXF(0.0f, 0.0f),
    CMPLXF(0.5f, 0.0f),    CMPLXF(3.0f, 0.0f),    CMPLXF(0.2f, 0.0f),    CMPLXF(4.0f, 0.0f),     CMPLXF(2.0f, 0.0f),    CMPLXF(0.1f, 0.0f),
    CMPLXF(0.4f, 0.0f),    CMPLXF(3.0f, 0.0f),    CMPLXF(-0.1f, 0.0f),   CMPLXF(3.0f, 0.0f),     CMPLXF(-1.0f, 0.0f),   CMPLXF(0.6f, 0.0f),
    CMPLXF(-0.1f, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(0.4f, 0.0f),    CMPLXF(-1.0f, 1.0f),    CMPLXF(4.0f, 0.0f),    CMPLXF(0.2f, 0.0f),
};
static const f32 c8_ls[] = {1e-3, 1e1, 1e-1, 1e3, 1, 1e-2};
static const f32 c8_rs[] = {1e1, 1e-1, 1e2, 1e-3, 1e2, 1e-3};

/* Case 9: N=6, structured with isolation */
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
static const c64 c9_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c9_bin[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c9_ls[] = {0, 1, 1, 1, 4, 5};
static const f32 c9_rs[] = {1, 1, 1, 1, 4, 4};

static const zgbal_case_t cases[] = {
    { 6, 0, 0, c0_a, c0_b, c0_ain, c0_bin, c0_ls, c0_rs },
    { 6, 0, 0, c1_a, c1_b, c1_ain, c1_bin, c1_ls, c1_rs },
    { 6, 0, 0, c2_a, c2_b, c2_ain, c2_bin, c2_ls, c2_rs },
    { 5, 0, 0, c3_a, c3_b, c3_ain, c3_bin, c3_ls, c3_rs },
    { 6, 0, 5, c4_a, c4_b, c4_ain, c4_bin, c4_ls, c4_rs },
    { 6, 3, 5, c5_a, c5_b, c5_ain, c5_bin, c5_ls, c5_rs },
    { 7, 2, 4, c6_a, c6_b, c6_ain, c6_bin, c6_ls, c6_rs },
    { 7, 2, 4, c7_a, c7_b, c7_ain, c7_bin, c7_ls, c7_rs },
    { 6, 0, 5, c8_a, c8_b, c8_ain, c8_bin, c8_ls, c8_rs },
    { 6, 2, 3, c9_a, c9_b, c9_ain, c9_bin, c9_ls, c9_rs },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_zggbal(void** state)
{
    (void)state;

    f32 eps = slamch("P");

    c64 a[LDA * LDA], b[LDA * LDA];
    c64 ain[LDA * LDA], bin[LDA * LDA];
    f32 lscale[LDA], rscale[LDA], work[LWORK];
    f32 dummy[1];
    INT ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax_info = 0, lmax_idx = 0, lmax_resid = 0;

    for (INT tc = 0; tc < NCASES; tc++) {
        const zgbal_case_t* c = &cases[tc];
        INT n = c->n;
        INT iloin = c->iloin;
        INT ihiin = c->ihiin;

        rowmajor_to_colmajor(c->a_rm, a, n, LDA);
        rowmajor_to_colmajor(c->b_rm, b, n, LDA);
        rowmajor_to_colmajor(c->ain_rm, ain, n, LDA);
        rowmajor_to_colmajor(c->bin_rm, bin, n, LDA);

        f32 anorm = clange("M", n, n, a, LDA, dummy);
        f32 bnorm = clange("M", n, n, b, LDA, dummy);

        knt++;

        cggbal("B", n, a, LDA, b, LDA, &ilo, &ihi, lscale, rscale, work, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        if (ilo != iloin || ihi != ihiin) {
            ninfo++;
            lmax_idx = knt;
            print_message("Case %d: ilo/ihi mismatch: got (%d,%d) expected (%d,%d)\n",
                          tc, ilo, ihi, iloin, ihiin);
        }

        vmax = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                f32 diff = cabsf(a[i + j * LDA] - ain[i + j * LDA]);
                if (diff > vmax) vmax = diff;
                diff = cabsf(b[i + j * LDA] - bin[i + j * LDA]);
                if (diff > vmax) vmax = diff;
            }
        }

        for (INT i = 0; i < n; i++) {
            f32 diff = fabsf(lscale[i] - c->lsclin[i]);
            if (diff > vmax) vmax = diff;
            diff = fabsf(rscale[i] - c->rsclin[i]);
            if (diff > vmax) vmax = diff;
        }

        f32 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("CGGBAL: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax_resid);
    if (ninfo > 0)
        print_message("  INFO/index errors: %d (info case %d, idx case %d)\n",
                      ninfo, lmax_info, lmax_idx);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zggbal),
    };
    return cmocka_run_group_tests_name("zchkgl", tests, NULL, NULL);
}
