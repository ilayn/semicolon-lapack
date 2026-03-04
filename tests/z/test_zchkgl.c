/**
 * @file test_zchkgl.c
 * @brief Tests ZGGBAL, a routine for balancing a matrix pair (A, B).
 *
 * Port of LAPACK TESTING/EIG/zchkgl.f with embedded test data from
 * TESTING/zgbal.in.
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0

#define LDA 20
#define LWORK (6 * LDA)

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c128* rm, c128* cm, INT n, INT ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(c128));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/zgbal.in ---------- */

typedef struct {
    INT n;
    INT iloin;
    INT ihiin;
    const c128* a_rm;
    const c128* b_rm;
    const c128* ain_rm;
    const c128* bin_rm;
    const f64* lsclin;
    const f64* rsclin;
} zgbal_case_t;

/* Case 0: N=6, diagonal pair */
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
static const c128 c0_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(6.0, 6.0),
};
static const c128 c0_bin[] = {
    CMPLX(6.0, 6.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c0_ls[] = {1, 1, 2, 3, 4, 5};
static const f64 c0_rs[] = {1, 1, 2, 3, 4, 5};

/* Case 1: N=6, sub-diagonal A, identity B */
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
static const c128 c1_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c1_bin[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c1_ls[] = {1, 1, 2, 2, 1, 0};
static const f64 c1_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 2: N=6, lower triangular A, same B */
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
static const c128 c2_ain[] = {
    CMPLX(6.0, 6.0), CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c2_bin[] = {
    CMPLX(6.0, 6.0), CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c2_ls[] = {1, 1, 2, 2, 1, 0};
static const f64 c2_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 3: N=5, lower triangular + identity */
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
static const c128 c3_ain[] = {
    CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c3_bin[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c3_ls[] = {1, 1, 2, 1, 0};
static const f64 c3_rs[] = {1, 1, 2, 1, 0};

/* Case 4: N=6, 1e11 upper triangular pair */
static const c128 c4_a[] = {
    CMPLX(1.0, 0.0), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
};
static const c128 c4_b[] = {
    CMPLX(1.0, 0.0), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11), CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1e11, 1e11),
    CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
};
static const c128 c4_ain[] = {
    CMPLX(1e-5, 0.0),  CMPLX(1e4, 1e4),  CMPLX(1e3, 1e3),  CMPLX(1e1, 1e1),  CMPLX(1e-1, 1e-1), CMPLX(1e-3, 1e-3),
    CMPLX(1e-3, 0.0),  CMPLX(1e-5, 0.0), CMPLX(1e5, 1e5),  CMPLX(1e3, 1e3),  CMPLX(1e1, 1e1),   CMPLX(1e-1, 1e-1),
    CMPLX(1e-1, 0.0),  CMPLX(1e-3, 0.0), CMPLX(1e-4, 0.0), CMPLX(1e5, 1e5),  CMPLX(1e3, 1e3),   CMPLX(1e1, 1e1),
    CMPLX(1e1, 0.0),   CMPLX(1e-1, 0.0), CMPLX(1e-2, 0.0), CMPLX(1e-4, 0.0), CMPLX(1e5, 1e5),   CMPLX(1e3, 1e3),
    CMPLX(1e2, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e-1, 0.0), CMPLX(1e-3, 0.0), CMPLX(1e-5, 0.0),  CMPLX(1e4, 1e4),
    CMPLX(1e4, 0.0),   CMPLX(1e2, 0.0),  CMPLX(1e1, 0.0),  CMPLX(1e-1, 0.0), CMPLX(1e-3, 0.0),  CMPLX(1e-5, 0.0),
};
static const c128 c4_bin[] = {
    CMPLX(1e-5, 0.0),  CMPLX(1e4, 1e4),  CMPLX(1e3, 1e3),  CMPLX(1e1, 1e1),  CMPLX(1e-1, 1e-1), CMPLX(1e-3, 1e-3),
    CMPLX(1e-3, 0.0),  CMPLX(1e-5, 0.0), CMPLX(1e5, 1e5),  CMPLX(1e3, 1e3),  CMPLX(1e1, 1e1),   CMPLX(1e-1, 1e-1),
    CMPLX(1e-1, 0.0),  CMPLX(1e-3, 0.0), CMPLX(1e-4, 0.0), CMPLX(1e5, 1e5),  CMPLX(1e3, 1e3),   CMPLX(1e1, 1e1),
    CMPLX(1e1, 0.0),   CMPLX(1e-1, 0.0), CMPLX(1e-2, 0.0), CMPLX(1e-4, 0.0), CMPLX(1e5, 1e5),   CMPLX(1e3, 1e3),
    CMPLX(1e2, 0.0),   CMPLX(1.0, 0.0),  CMPLX(1e-1, 0.0), CMPLX(1e-3, 0.0), CMPLX(1e-5, 0.0),  CMPLX(1e4, 1e4),
    CMPLX(1e4, 0.0),   CMPLX(1e2, 0.0),  CMPLX(1e1, 0.0),  CMPLX(1e-1, 0.0), CMPLX(1e-3, 0.0),  CMPLX(1e-5, 0.0),
};
static const f64 c4_ls[] = {1e-7, 1e-5, 1e-3, 1e-1, 1, 1e2};
static const f64 c4_rs[] = {1e2, 1, 1e-1, 1e-3, 1e-5, 1e-7};

/* Case 5: N=6, structured with 1e6 entries and isolation */
static const c128 c5_a[] = {
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0), CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e6, 1e6),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e-6, 1e-6),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e6, 1e6),   CMPLX(1e6, 1e6),
};
static const c128 c5_b[] = {
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0), CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1.0, 0.0),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e6, 1e6),
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0), CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),   CMPLX(1.0, 0.0),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-6, 1e-6), CMPLX(1e-6, 1e-6),
    CMPLX(1e6, 1e6),  CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e6, 1e6),   CMPLX(1e6, 1e6),
};
static const c128 c5_ain[] = {
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 1.0),    CMPLX(1e-4, 1e-4), CMPLX(1e4, 1e4),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e4, 1e4),    CMPLX(1.0, 1.0),   CMPLX(1e-4, 1e-4),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-4, 1e-4),  CMPLX(1e4, 1e4),   CMPLX(1.0, 1.0),
};
static const c128 c5_bin[] = {
    CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 0.0),  CMPLX(1e-5, 0.0),   CMPLX(1e3, 0.0),   CMPLX(1e-1, 0.0),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1.0, 1.0),    CMPLX(1e-4, 1e-4), CMPLX(1e4, 1e4),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e4, 1e4),    CMPLX(1.0, 1.0),   CMPLX(1e-4, 1e-4),
    CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(1e-4, 1e-4),  CMPLX(1e4, 1e4),   CMPLX(1.0, 1.0),
};
static const f64 c5_ls[] = {3, 3, 3, 1e-1, 1e3, 1e-5};
static const f64 c5_rs[] = {1, 2, 3, 1e-5, 1e3, 1e-1};

/* Case 6: N=7, with isolation at top and bottom */
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
static const c128 c6_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c6_bin[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c6_ls[] = {2, 1, 1, 1, 1, 5, 4};
static const f64 c6_rs[] = {0, 2, 1, 1, 1, 1, 1};

/* Case 7: N=7, structured with 1e5 entries and isolation */
static const c128 c7_a[] = {
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e-3, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e-5, 1e-5),  CMPLX(1e5, 1e5),    CMPLX(1e-4, 1e-4),  CMPLX(1e-3, 0.0),   CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),
    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e-3, 1e-3),  CMPLX(1e5, 1.0),    CMPLX(1e5, 1.0),    CMPLX(1e5, 1e5),    CMPLX(1e3, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e-4, 1e-5),  CMPLX(0.0, 0.0),    CMPLX(1e-1, 1e-1),  CMPLX(1.0, 1.0),    CMPLX(1e-4, 1e-4),  CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e-3, 1e-3),  CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e-5, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e1, 1e1),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e-3, 1e-3),  CMPLX(1e2, 0.0),
};
static const c128 c7_b[] = {
    CMPLX(0.0, 0.0),    CMPLX(1e-2, 0.0),   CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e-3, 0.0),   CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e-1, 0.0),   CMPLX(1e2, 0.0),
    CMPLX(1e5, 1e5),    CMPLX(1e2, 0.0),    CMPLX(1e3, 0.0),    CMPLX(1e3, 0.0),    CMPLX(1e-4, 1e-3),  CMPLX(1.0, 0.0),    CMPLX(1.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e-5, 0.0),   CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(1e-4, 0.0),   CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),    CMPLX(1e-4, 1e-3),
};
static const c128 c7_ain[] = {
    CMPLX(1e5, 1e5),    CMPLX(1e-3, 1e-3),  CMPLX(1e4, 1e4),    CMPLX(1e2, 1e-3),   CMPLX(1e4, 1e-1),   CMPLX(1e3, 0.0),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e4, 1e4),    CMPLX(1e-7, 1e-7),  CMPLX(1e-4, 0.0),   CMPLX(1e5, 1e5),    CMPLX(1e-5, 1e-5),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e2, 1e2),    CMPLX(1.0, 1.0),    CMPLX(1e2, 1e2),    CMPLX(1e-5, 0.0),   CMPLX(1e3, 1e3),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e-5, 1e-5),  CMPLX(1e-4, 1e-4),  CMPLX(1e-1, 1e-1),  CMPLX(1e5, 1e5),    CMPLX(1e-4, 1e-5),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e-6, 1e-6),  CMPLX(1.0, 1.0),    CMPLX(1e2, 1e2),    CMPLX(1.0, 0.0),    CMPLX(1e-1, 1e-1),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e-5, 0.0),   CMPLX(1e-3, 1e-3),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),
};
static const c128 c7_bin[] = {
    CMPLX(1e5, 1e5),    CMPLX(1e3, 0.0),    CMPLX(1e-1, 0.0),   CMPLX(1.0, 0.0),    CMPLX(1e-5, 1e-4),  CMPLX(1.0, 0.0),    CMPLX(1e2, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1.0, 0.0),    CMPLX(1e-2, 0.0),   CMPLX(1e2, 1e2),    CMPLX(1e4, 1e4),    CMPLX(1e2, 0.0),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e2, 1e2),    CMPLX(1.0, 1.0),    CMPLX(1e-6, 0.0),   CMPLX(1e3, 1e3),    CMPLX(1e-4, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e4, 1e4),    CMPLX(1e2, 1e2),    CMPLX(1e4, 1e4),    CMPLX(1e5, 1e5),    CMPLX(1e-5, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e2, 1e2),    CMPLX(1.0, 1.0),    CMPLX(1e2, 1e2),    CMPLX(1e-6, 1e-5),  CMPLX(1e-6, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),    CMPLX(1e5, 1e5),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e5, 1e5),
};
static const f64 c7_ls[] = {2, 1, 1e-2, 1, 1e-2, 5, 4};
static const f64 c7_rs[] = {0, 2, 1e-1, 1e-3, 1e-1, 1, 1};

/* Case 8: N=6, large magnitude differences */
static const c128 c8_a[] = {
    CMPLX(-20.0, 1.0),  CMPLX(-1e4, 1e4),    CMPLX(-2.0, 0.0),   CMPLX(-1e6, 0.0),    CMPLX(-10.0, 0.0),  CMPLX(-2e5, 1e5),
    CMPLX(6e-3, 0.0),   CMPLX(4.0, 0.0),     CMPLX(6e-4, 0.0),   CMPLX(200.0, 0.0),   CMPLX(3e-3, 0.0),   CMPLX(30.0, 0.0),
    CMPLX(-0.2, 0.0),   CMPLX(-300.0, 0.0),  CMPLX(-0.04, 0.0),  CMPLX(-1e4, 1e4),    CMPLX(0.0, 0.0),    CMPLX(3000.0, 1000.0),
    CMPLX(6e-5, 0.0),   CMPLX(0.04, 0.0),    CMPLX(9e-6, 0.0),   CMPLX(9.0, 0.0),     CMPLX(3e-5, 0.0),   CMPLX(0.5, 0.0),
    CMPLX(0.06, 0.0),   CMPLX(50.0, 0.0),    CMPLX(8e-3, 0.0),   CMPLX(-4000.0, 0.0), CMPLX(0.08, 0.0),   CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1000.0, 1000.0), CMPLX(0.7, 0.0),  CMPLX(-2e5, 1e5),    CMPLX(13.0, 0.0),   CMPLX(-6e4, 1e4),
};
static const c128 c8_b[] = {
    CMPLX(-20.0, 0.0),  CMPLX(-1e4, 1e4),    CMPLX(2.0, 0.0),    CMPLX(-2e6, 0.0),    CMPLX(10.0, 0.0),   CMPLX(-1e5, 1e5),
    CMPLX(5e-3, 0.0),   CMPLX(3.0, 0.0),     CMPLX(-2e-4, 0.0),  CMPLX(400.0, 0.0),   CMPLX(-1e-3, 0.0),  CMPLX(30.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(-100.0, 0.0),  CMPLX(-0.08, 0.0),  CMPLX(2e4, 0.0),     CMPLX(-0.4, 0.0),   CMPLX(0.0, 0.0),
    CMPLX(5e-5, 0.0),   CMPLX(0.03, 0.0),    CMPLX(2e-6, 0.0),   CMPLX(4.0, 0.0),     CMPLX(2e-5, 0.0),   CMPLX(0.1, 0.0),
    CMPLX(0.04, 0.0),   CMPLX(30.0, 0.0),    CMPLX(-1e-3, 0.0),  CMPLX(3000.0, 0.0),  CMPLX(-0.01, 0.0),  CMPLX(600.0, 0.0),
    CMPLX(-1.0, 0.0),   CMPLX(0.0, 0.0),     CMPLX(0.4, 0.0),    CMPLX(-1e5, 1e5),    CMPLX(4.0, 0.0),    CMPLX(2e4, 0.0),
};
static const c128 c8_ain[] = {
    CMPLX(-0.2, 0.01),  CMPLX(-1.0, 1.0),   CMPLX(-0.2, 0.0),   CMPLX(-1.0, 0.0),    CMPLX(-1.0, 0.0),   CMPLX(-0.2, 0.1),
    CMPLX(0.6, 0.0),    CMPLX(4.0, 0.0),    CMPLX(0.6, 0.0),    CMPLX(2.0, 0.0),     CMPLX(3.0, 0.0),    CMPLX(0.3, 0.0),
    CMPLX(-0.2, 0.0),   CMPLX(-3.0, 0.0),   CMPLX(-0.4, 0.0),   CMPLX(-1.0, 1.0),    CMPLX(0.0, 0.0),    CMPLX(0.3, 0.1),
    CMPLX(0.6, 0.0),    CMPLX(4.0, 0.0),    CMPLX(0.9, 0.0),    CMPLX(9.0, 0.0),     CMPLX(3.0, 0.0),    CMPLX(0.5, 0.0),
    CMPLX(0.6, 0.0),    CMPLX(5.0, 0.0),    CMPLX(0.8, 0.0),    CMPLX(-4.0, 0.0),    CMPLX(8.0, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(1.0, 1.0),    CMPLX(0.7, 0.0),    CMPLX(-2.0, 1.0),    CMPLX(13.0, 0.0),   CMPLX(-0.6, 0.1),
};
static const c128 c8_bin[] = {
    CMPLX(-0.2, 0.0),   CMPLX(-1.0, 1.0),   CMPLX(0.2, 0.0),    CMPLX(-2.0, 0.0),    CMPLX(1.0, 0.0),    CMPLX(-0.1, 0.1),
    CMPLX(0.5, 0.0),    CMPLX(3.0, 0.0),    CMPLX(-0.2, 0.0),   CMPLX(4.0, 0.0),     CMPLX(-1.0, 0.0),   CMPLX(0.3, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(-1.0, 0.0),   CMPLX(-0.8, 0.0),   CMPLX(2.0, 0.0),     CMPLX(-4.0, 0.0),   CMPLX(0.0, 0.0),
    CMPLX(0.5, 0.0),    CMPLX(3.0, 0.0),    CMPLX(0.2, 0.0),    CMPLX(4.0, 0.0),     CMPLX(2.0, 0.0),    CMPLX(0.1, 0.0),
    CMPLX(0.4, 0.0),    CMPLX(3.0, 0.0),    CMPLX(-0.1, 0.0),   CMPLX(3.0, 0.0),     CMPLX(-1.0, 0.0),   CMPLX(0.6, 0.0),
    CMPLX(-0.1, 0.0),   CMPLX(0.0, 0.0),    CMPLX(0.4, 0.0),    CMPLX(-1.0, 1.0),    CMPLX(4.0, 0.0),    CMPLX(0.2, 0.0),
};
static const f64 c8_ls[] = {1e-3, 1e1, 1e-1, 1e3, 1, 1e-2};
static const f64 c8_rs[] = {1e1, 1e-1, 1e2, 1e-3, 1e2, 1e-3};

/* Case 9: N=6, structured with isolation */
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
static const c128 c9_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c9_bin[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c9_ls[] = {0, 1, 1, 1, 4, 5};
static const f64 c9_rs[] = {1, 1, 1, 1, 4, 4};

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

    f64 eps = dlamch("P");

    c128 a[LDA * LDA], b[LDA * LDA];
    c128 ain[LDA * LDA], bin[LDA * LDA];
    f64 lscale[LDA], rscale[LDA], work[LWORK];
    f64 dummy[1];
    INT ilo, ihi, info;
    f64 rmax = 0.0, vmax;
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

        f64 anorm = zlange("M", n, n, a, LDA, dummy);
        f64 bnorm = zlange("M", n, n, b, LDA, dummy);

        knt++;

        zggbal("B", n, a, LDA, b, LDA, &ilo, &ihi, lscale, rscale, work, &info);

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

        vmax = 0.0;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                f64 diff = cabs(a[i + j * LDA] - ain[i + j * LDA]);
                if (diff > vmax) vmax = diff;
                diff = cabs(b[i + j * LDA] - bin[i + j * LDA]);
                if (diff > vmax) vmax = diff;
            }
        }

        for (INT i = 0; i < n; i++) {
            f64 diff = fabs(lscale[i] - c->lsclin[i]);
            if (diff > vmax) vmax = diff;
            diff = fabs(rscale[i] - c->rsclin[i]);
            if (diff > vmax) vmax = diff;
        }

        f64 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("ZGGBAL: %d cases, max residual = %.3e (case %d)\n",
                  knt, rmax, lmax_resid);
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
