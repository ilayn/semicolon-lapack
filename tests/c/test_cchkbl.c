/**
 * @file test_cchkbl.c
 * @brief Tests CGEBAL, a routine for balancing a general complex matrix.
 *
 * Port of LAPACK TESTING/EIG/cchkbl.f with embedded test data from
 * TESTING/cbal.in. All test matrices and expected results are hardcoded
 * as static arrays.
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0f

#define LDA 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c64* rm, c64* cm, INT n, INT ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(c64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/cbal.in ---------- */

typedef struct {
    INT n;
    INT iloin;
    INT ihiin;
    const c64* a_rm;
    const c64* ain_rm;
    const f32* scalin;
} zbal_case_t;

/* Case 0: N=5 diagonal */
static const c64 c0_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f),
};
static const c64 c0_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(5.0f, 5.0f),
};
static const f32 c0_s[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

/* Case 1: N=5 lower triangular */
static const c64 c1_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(2.0f, 2.0f), CMPLXF(3.0f, 3.0f), CMPLXF(4.0f, 4.0f), CMPLXF(5.0f, 5.0f),
};
static const c64 c1_ain[] = {
    CMPLXF(5.0f, 5.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 4.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 3.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 2.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c1_s[] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};

/* Case 2: N=5 sub-diagonal */
static const c64 c2_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c2_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 0.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c2_s[] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};

/* Case 3: N=4 */
static const c64 c3_a[] = {
    CMPLXF(0.0f, 0.0f),   CMPLXF(2.0f, 0.0f),   CMPLXF(0.1f, 0.0f),   CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.1f, 0.0f),
    CMPLXF(100.0f, 0.0f), CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(2.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(100.0f, 0.0f), CMPLXF(2.0f, 0.0f),   CMPLXF(0.0f, 0.0f),
};
static const c64 c3_ain[] = {
    CMPLXF(0.0f, 0.0f),   CMPLXF(2.0f, 0.0f),   CMPLXF(3.2f, 0.0f),   CMPLXF(0.0f, 0.0f),
    CMPLXF(2.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(3.2f, 0.0f),
    CMPLXF(3.125f, 0.0f), CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(2.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),   CMPLXF(3.125f, 0.0f), CMPLXF(2.0f, 0.0f),   CMPLXF(0.0f, 0.0f),
};
static const f32 c3_s[] = {0.0625f, 0.0625f, 2.0f, 2.0f};

/* Case 4: N=6 */
static const c64 c4_a[] = {
    CMPLXF(1.0f, 1.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(1024.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(128.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 1.0f),      CMPLXF(3000.0f, 0.0f),    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(2.0f, 0.0f),
    CMPLXF(0.0f, 128.0f),   CMPLXF(4.0f, 0.0f),      CMPLXF(0.004f, 0.0f),     CMPLXF(5.0f, 0.0f),     CMPLXF(600.0f, 0.0f),     CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.002f),     CMPLXF(2.0f, 0.0f),
    CMPLXF(8.0f, 0.0f),     CMPLXF(0.0f, 8192.0f),   CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(2.0f, 0.0f),
};
static const c64 c4_ain[] = {
    CMPLXF(5.0f, 0.0f),     CMPLXF(0.004f, 0.0f),    CMPLXF(600.0f, 0.0f),     CMPLXF(0.0f, 1024.0f),  CMPLXF(0.5f, 0.0f),       CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(3000.0f, 0.0f),   CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.25f, 0.125f),    CMPLXF(2.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.002f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(2.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(1.0f, 1.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(128.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(1024.0f, 0.0f),
    CMPLXF(64.0f, 0.0f),    CMPLXF(0.0f, 1024.0f),   CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(2.0f, 0.0f),
};
static const f32 c4_s[] = {4.0f, 3.0f, 5.0f, 8.0f, 0.125f, 1.0f};

/* Case 5: N=5 */
static const c64 c5_a[] = {
    CMPLXF(1.0f, 1.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 1.0f),      CMPLXF(8192.0f, 0.0f),    CMPLXF(2.0f, 0.0f),     CMPLXF(4.0f, 0.0f),
    CMPLXF(2.5e-4f, 0.0f),  CMPLXF(1.25e-4f, 0.0f),  CMPLXF(4.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(64.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 0.0f),      CMPLXF(1024.0f, 1.024f),  CMPLXF(4.0f, 0.0f),     CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 8192.0f),   CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),     CMPLXF(8.0f, 0.0f),
};
static const c64 c5_ain[] = {
    CMPLXF(1.0f, 1.0f),     CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.25f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 1.0f),       CMPLXF(1024.0f, 0.0f),    CMPLXF(16.0f, 0.0f),     CMPLXF(16.0f, 0.0f),
    CMPLXF(0.256f, 0.0f),   CMPLXF(0.001f, 0.0f),     CMPLXF(4.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(2048.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.25f, 0.0f),      CMPLXF(16.0f, 0.016f),    CMPLXF(4.0f, 0.0f),      CMPLXF(4.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 2048.0f),    CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(8.0f, 0.0f),
};
static const f32 c5_s[] = {64.0f, 0.5f, 0.0625f, 4.0f, 2.0f};

/* Case 6: N=4 */
static const c64 c6_a[] = {
    CMPLXF(1.0f, 1.0f),     CMPLXF(1e6, 0.0f),    CMPLXF(1e6, 0.0f),     CMPLXF(1e6, 0.0f),
    CMPLXF(-2e6, 0.0f),    CMPLXF(3.0f, 1.0f),    CMPLXF(2e-6, 0.0f),    CMPLXF(3e-6, 0.0f),
    CMPLXF(-3e6, 0.0f),    CMPLXF(0.0f, 0.0f),    CMPLXF(1e-6, 1.0f),    CMPLXF(2.0f, 0.0f),
    CMPLXF(1e6, 0.0f),     CMPLXF(0.0f, 0.0f),    CMPLXF(3e-6, 0.0f),    CMPLXF(4e6, 1.0f),
};
static const c64 c6_ain[] = {
    CMPLXF(1.0f, 1.0f),     CMPLXF(1e6, 0.0f),    CMPLXF(2e6, 0.0f),     CMPLXF(1e6, 0.0f),
    CMPLXF(-2e6, 0.0f),    CMPLXF(3.0f, 1.0f),    CMPLXF(4e-6, 0.0f),    CMPLXF(3e-6, 0.0f),
    CMPLXF(-1.5e6f, 0.0f),  CMPLXF(0.0f, 0.0f),    CMPLXF(1e-6, 1.0f),    CMPLXF(1.0f, 0.0f),
    CMPLXF(1e6, 0.0f),     CMPLXF(0.0f, 0.0f),    CMPLXF(6e-6, 0.0f),    CMPLXF(4e6, 1.0f),
};
static const f32 c6_s[] = {1.0f, 1.0f, 2.0f, 1.0f};

/* Case 7: N=4 */
static const c64 c7_a[] = {
    CMPLXF(1.0f, 0.0f),     CMPLXF(0.0f, 1e4),    CMPLXF(0.0f, 1e4),     CMPLXF(0.0f, 1e4),
    CMPLXF(-2e4, 0.0f),    CMPLXF(3.0f, 0.0f),    CMPLXF(2e-3, 0.0f),    CMPLXF(3e-3, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 1.0f),    CMPLXF(0.0f, 0.0f),     CMPLXF(-3e4, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),    CMPLXF(1e4, 0.0f),     CMPLXF(0.0f, 0.0f),
};
static const c64 c7_ain[] = {
    CMPLXF(1.0f, 0.0f),     CMPLXF(0.0f, 1e4),    CMPLXF(0.0f, 1e4),     CMPLXF(0.0f, 5e3),
    CMPLXF(-2e4, 0.0f),    CMPLXF(3.0f, 0.0f),    CMPLXF(2e-3, 0.0f),    CMPLXF(1.5e-3f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(2.0f, 1.0f),    CMPLXF(0.0f, 0.0f),     CMPLXF(-1.5e4f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),    CMPLXF(2e4, 0.0f),     CMPLXF(0.0f, 0.0f),
};
static const f32 c7_s[] = {1.0f, 1.0f, 1.0f, 0.5f};

/* Case 8: N=5 */
static const c64 c8_a[] = {
    CMPLXF(1.0f, 0.0f),     CMPLXF(512.0f, 0.0f),    CMPLXF(4096.0f, 0.0f),    CMPLXF(32768.0f, 0.0f),  CMPLXF(262144.0f, 0.0f),
    CMPLXF(8.0f, 8.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(8.0f, 8.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(8.0f, 8.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(8.0f, 8.0f),      CMPLXF(0.0f, 0.0f),
};
static const c64 c8_ain[] = {
    CMPLXF(1.0f, 0.0f),      CMPLXF(64.0f, 0.0f),     CMPLXF(64.0f, 0.0f),      CMPLXF(64.0f, 0.0f),     CMPLXF(64.0f, 0.0f),
    CMPLXF(64.0f, 64.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),      CMPLXF(64.0f, 64.0f),    CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(64.0f, 64.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(64.0f, 64.0f),    CMPLXF(0.0f, 0.0f),
};
static const f32 c8_s[] = {128.0f, 16.0f, 2.0f, 0.25f, 0.03125f};

/* Case 9: N=6 with isolation */
static const c64 c9_a[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
};
static const c64 c9_ain[] = {
    CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f), CMPLXF(1.0f, 1.0f),
    CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 1.0f),
};
static const f32 c9_s[] = {3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f};

/* Case 10: N=7 */
static const c64 c10_a[] = {
    CMPLXF(6.0f, 0.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(1.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(4.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(2.5e-4f, 0.0f),   CMPLXF(0.0125f, 0.0f),   CMPLXF(0.02f, 0.0f),     CMPLXF(0.125f, 0.0f),
    CMPLXF(1.0f, 0.0f),    CMPLXF(128.0f, 0.0f),    CMPLXF(64.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(-2.0f, 0.0f),     CMPLXF(16.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(16384.0f, 0.0f),  CMPLXF(0.0f, 0.0f),      CMPLXF(1.0f, 0.0f),      CMPLXF(-400.0f, 0.0f),   CMPLXF(256.0f, 0.0f),    CMPLXF(-4000.0f, 0.0f),
    CMPLXF(-2.0f, 0.0f),   CMPLXF(-256.0f, 0.0f),   CMPLXF(0.0f, 0.0f),      CMPLXF(0.0125f, 0.0f),   CMPLXF(2.0f, 0.0f),      CMPLXF(2.0f, 0.0f),      CMPLXF(32.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),    CMPLXF(8.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.004f, 0.0f),    CMPLXF(0.125f, 0.0f),    CMPLXF(-0.2f, 0.0f),     CMPLXF(3.0f, 0.0f),
};
static const c64 c10_ain[] = {
    CMPLXF(64.0f, 0.0f),    CMPLXF(0.25f, 0.0f),     CMPLXF(0.5f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(1.0f, 0.0f),      CMPLXF(-2.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(4.0f, 0.0f),      CMPLXF(2.0f, 0.0f),       CMPLXF(4.096f, 0.0f),    CMPLXF(1.6f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(10.24f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.5f, 0.0f),      CMPLXF(3.0f, 0.0f),       CMPLXF(4.096f, 0.0f),    CMPLXF(1.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(-6.4f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(1.0f, 0.0f),      CMPLXF(-3.90625f, 0.0f),  CMPLXF(1.0f, 0.0f),      CMPLXF(-3.125f, 0.0f),   CMPLXF(0.0f, 0.0f),      CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(-2.0f, 0.0f),     CMPLXF(4.0f, 0.0f),       CMPLXF(1.6f, 0.0f),      CMPLXF(2.0f, 0.0f),      CMPLXF(-8.0f, 0.0f),     CMPLXF(8.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(6.0f, 0.0f),      CMPLXF(1.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),      CMPLXF(0.0f, 0.0f),
};
static const f32 c10_s[] = {3.0f, 1.953125e-3f, 0.03125f, 32.0f, 0.25f, 1.0f, 6.0f};

/* Case 11: N=5 */
static const c64 c11_a[] = {
    CMPLXF(1000.0f, 0.0f),  CMPLXF(2.0f, 0.0f),      CMPLXF(3.0f, 0.0f),      CMPLXF(4.0f, 0.0f),      CMPLXF(5e5, 0.0f),
    CMPLXF(9.0f, 0.0f),     CMPLXF(0.0f, 0.0f),      CMPLXF(2e-4, 0.0f),     CMPLXF(1.0f, 0.0f),      CMPLXF(3.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(-300.0f, 0.0f),   CMPLXF(2.0f, 0.0f),      CMPLXF(1.0f, 0.0f),      CMPLXF(1.0f, 0.0f),
    CMPLXF(9.0f, 0.0f),     CMPLXF(0.002f, 0.0f),    CMPLXF(1.0f, 0.0f),      CMPLXF(1.0f, 0.0f),      CMPLXF(-1000.0f, 0.0f),
    CMPLXF(6.0f, 0.0f),     CMPLXF(200.0f, 0.0f),    CMPLXF(1.0f, 0.0f),      CMPLXF(600.0f, 0.0f),    CMPLXF(3.0f, 0.0f),
};
static const c64 c11_ain[] = {
    CMPLXF(1000.0f, 0.0f),   CMPLXF(0.03125f, 0.0f),   CMPLXF(0.375f, 0.0f),     CMPLXF(0.0625f, 0.0f),   CMPLXF(3906.25f, 0.0f),
    CMPLXF(576.0f, 0.0f),    CMPLXF(0.0f, 0.0f),       CMPLXF(0.0016f, 0.0f),    CMPLXF(1.0f, 0.0f),      CMPLXF(1.5f, 0.0f),
    CMPLXF(0.0f, 0.0f),      CMPLXF(-37.5f, 0.0f),     CMPLXF(2.0f, 0.0f),       CMPLXF(0.125f, 0.0f),    CMPLXF(0.0625f, 0.0f),
    CMPLXF(576.0f, 0.0f),    CMPLXF(0.002f, 0.0f),     CMPLXF(8.0f, 0.0f),       CMPLXF(1.0f, 0.0f),      CMPLXF(-500.0f, 0.0f),
    CMPLXF(768.0f, 0.0f),    CMPLXF(400.0f, 0.0f),     CMPLXF(16.0f, 0.0f),      CMPLXF(1200.0f, 0.0f),   CMPLXF(3.0f, 0.0f),
};
static const f32 c11_s[] = {128.0f, 2.0f, 16.0f, 2.0f, 1.0f};

/* Case 12: N=5 extreme magnitudes (from cbal.in, differs from zbal.in) */
static const c64 c12_a[] = {
    CMPLXF(1.0f, 0.0f),     CMPLXF(1.0e15f, 0.0f),  CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),
    CMPLXF(1.0e-15f, 0.0f), CMPLXF(1.0f, 0.0f),     CMPLXF(1.0e15f, 0.0f),  CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(1.0e-15f, 0.0f), CMPLXF(1.0f, 0.0f),     CMPLXF(1.0e15f, 0.0f),  CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(1.0e-15f, 0.0f), CMPLXF(1.0f, 0.0f),     CMPLXF(1.0e15f, 0.0f),
    CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(0.0f, 0.0f),     CMPLXF(1.0e-15f, 0.0f), CMPLXF(1.0f, 0.0f),
};
static const c64 c12_ain[] = {
    CMPLXF(1.0000000e+00f, 0.0f), CMPLXF(7.1054273e+00f, 0.0f), CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),
    CMPLXF(1.4073749e-01f, 0.0f), CMPLXF(1.0000000e+00f, 0.0f), CMPLXF(3.5527136e+00f, 0.0f), CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),           CMPLXF(2.8147498e-01f, 0.0f), CMPLXF(1.0000000e+00f, 0.0f), CMPLXF(1.7763568e+00f, 0.0f), CMPLXF(0.0f, 0.0f),
    CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),           CMPLXF(5.6294996e-01f, 0.0f), CMPLXF(1.0000000e+00f, 0.0f), CMPLXF(8.8817841e-01f, 0.0f),
    CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),           CMPLXF(0.0f, 0.0f),           CMPLXF(1.1258999e+00f, 0.0f), CMPLXF(1.0000000e+00f, 0.0f),
};
static const f32 c12_s[] = {
    5.0706024e+30f, 3.6028797e+16f, 1.2800000e+02f, 2.2737368e-13f, 2.0194839e-28f
};

static const zbal_case_t cases[] = {
    { 5, 0, 0, c0_a, c0_ain, c0_s },
    { 5, 0, 0, c1_a, c1_ain, c1_s },
    { 5, 0, 0, c2_a, c2_ain, c2_s },
    { 4, 0, 3, c3_a, c3_ain, c3_s },
    { 6, 3, 5, c4_a, c4_ain, c4_s },
    { 5, 0, 4, c5_a, c5_ain, c5_s },
    { 4, 0, 3, c6_a, c6_ain, c6_s },
    { 4, 0, 3, c7_a, c7_ain, c7_s },
    { 5, 0, 4, c8_a, c8_ain, c8_s },
    { 6, 1, 4, c9_a, c9_ain, c9_s },
    { 7, 1, 4, c10_a, c10_ain, c10_s },
    { 5, 0, 4, c11_a, c11_ain, c11_s },
    { 5, 0, 4, c12_a, c12_ain, c12_s },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_zgebal(void** state)
{
    (void)state;

    f32 sfmin = slamch("S");

    c64 a[LDA * LDA], ain[LDA * LDA];
    f32 scale[LDA];
    INT ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax_info = 0, lmax_idx = 0, lmax_resid = 0;

    for (INT tc = 0; tc < NCASES; tc++) {
        const zbal_case_t* c = &cases[tc];
        INT n = c->n;
        INT iloin = c->iloin;
        INT ihiin = c->ihiin;

        rowmajor_to_colmajor(c->a_rm, a, n, LDA);
        rowmajor_to_colmajor(c->ain_rm, ain, n, LDA);

        knt++;

        cgebal("B", n, a, LDA, &ilo, &ihi, scale, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        if (ilo != iloin || ihi != ihiin) {
            ninfo++;
            lmax_idx = knt;
            fprintf(stderr, "Case %d: ilo/ihi mismatch: got (%d,%d) expected (%d,%d)\n",
                          tc, ilo, ihi, iloin, ihiin);
        }

        vmax = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                c64 aij = a[i + j * LDA];
                c64 ainij = ain[i + j * LDA];
                f32 temp = cabs1f(aij);
                if (cabs1f(ainij) > temp) temp = cabs1f(ainij);
                if (sfmin > temp) temp = sfmin;
                f32 diff = cabs1f(aij - ainij) / temp;
                if (diff > vmax) vmax = diff;
            }
        }

        for (INT i = 0; i < n; i++) {
            f32 si = scale[i];
            f32 ei = c->scalin[i];
            f32 temp = fabsf(si);
            if (fabsf(ei) > temp) temp = fabsf(ei);
            if (sfmin > temp) temp = sfmin;
            f32 diff = fabsf(si - ei) / temp;
            if (diff > vmax) vmax = diff;
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    fprintf(stderr, "CGEBAL: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax_resid);
    if (ninfo > 0)
        fprintf(stderr, "  INFO/index errors: %d (info case %d, idx case %d)\n",
                      ninfo, lmax_info, lmax_idx);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zgebal),
    };
    (void)cmocka_run_group_tests_name("zchkbl", tests, NULL, NULL);
    return 0;
}
