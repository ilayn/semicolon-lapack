/**
 * @file test_zchkbl.c
 * @brief Tests ZGEBAL, a routine for balancing a general complex matrix.
 *
 * Port of LAPACK TESTING/EIG/zchkbl.f with embedded test data from
 * TESTING/zbal.in. All test matrices and expected results are hardcoded
 * as static arrays.
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0

#define LDA 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const c128* rm, c128* cm, INT n, INT ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(c128));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/zbal.in ---------- */

typedef struct {
    INT n;
    INT iloin;
    INT ihiin;
    const c128* a_rm;
    const c128* ain_rm;
    const f64* scalin;
} zbal_case_t;

/* Case 0: N=5 diagonal */
static const c128 c0_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 5.0),
};
static const c128 c0_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(2.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(5.0, 5.0),
};
static const f64 c0_s[] = {1.0, 2.0, 3.0, 4.0, 5.0};

/* Case 1: N=5 lower triangular */
static const c128 c1_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(2.0, 2.0), CMPLX(3.0, 3.0), CMPLX(4.0, 4.0), CMPLX(5.0, 5.0),
};
static const c128 c1_ain[] = {
    CMPLX(5.0, 5.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(4.0, 4.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 3.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(2.0, 2.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c1_s[] = {1.0, 2.0, 3.0, 2.0, 1.0};

/* Case 2: N=5 sub-diagonal */
static const c128 c2_a[] = {
    CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(1.0, 1.0),
};
static const c128 c2_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 0.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c2_s[] = {1.0, 2.0, 3.0, 2.0, 1.0};

/* Case 3: N=4 */
static const c128 c3_a[] = {
    CMPLX(0.0, 0.0),   CMPLX(2.0, 0.0),   CMPLX(0.1, 0.0),   CMPLX(0.0, 0.0),
    CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.1, 0.0),
    CMPLX(100.0, 0.0), CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(2.0, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(100.0, 0.0), CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0),
};
static const c128 c3_ain[] = {
    CMPLX(0.0, 0.0),   CMPLX(2.0, 0.0),   CMPLX(3.2, 0.0),   CMPLX(0.0, 0.0),
    CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(3.2, 0.0),
    CMPLX(3.125, 0.0), CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(2.0, 0.0),
    CMPLX(0.0, 0.0),   CMPLX(3.125, 0.0), CMPLX(2.0, 0.0),   CMPLX(0.0, 0.0),
};
static const f64 c3_s[] = {0.0625, 0.0625, 2.0, 2.0};

/* Case 4: N=6 */
static const c128 c4_a[] = {
    CMPLX(1.0, 1.0),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(1024.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(128.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 1.0),      CMPLX(3000.0, 0.0),    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(2.0, 0.0),
    CMPLX(0.0, 128.0),   CMPLX(4.0, 0.0),      CMPLX(0.004, 0.0),     CMPLX(5.0, 0.0),     CMPLX(600.0, 0.0),     CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.002),     CMPLX(2.0, 0.0),
    CMPLX(8.0, 0.0),     CMPLX(0.0, 8192.0),   CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(2.0, 0.0),
};
static const c128 c4_ain[] = {
    CMPLX(5.0, 0.0),     CMPLX(0.004, 0.0),    CMPLX(600.0, 0.0),     CMPLX(0.0, 1024.0),  CMPLX(0.5, 0.0),       CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(3000.0, 0.0),   CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.25, 0.125),    CMPLX(2.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.002),     CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(2.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(1.0, 1.0),     CMPLX(0.0, 0.0),       CMPLX(128.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(1024.0, 0.0),
    CMPLX(64.0, 0.0),    CMPLX(0.0, 1024.0),   CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),       CMPLX(2.0, 0.0),
};
static const f64 c4_s[] = {4.0, 3.0, 5.0, 8.0, 0.125, 1.0};

/* Case 5: N=5 */
static const c128 c5_a[] = {
    CMPLX(1.0, 1.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 1.0),      CMPLX(8192.0, 0.0),    CMPLX(2.0, 0.0),     CMPLX(4.0, 0.0),
    CMPLX(2.5e-4, 0.0),  CMPLX(1.25e-4, 0.0),  CMPLX(4.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(64.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 0.0),      CMPLX(1024.0, 1.024),  CMPLX(4.0, 0.0),     CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 8192.0),   CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),     CMPLX(8.0, 0.0),
};
static const c128 c5_ain[] = {
    CMPLX(1.0, 1.0),     CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.25, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 1.0),       CMPLX(1024.0, 0.0),    CMPLX(16.0, 0.0),     CMPLX(16.0, 0.0),
    CMPLX(0.256, 0.0),   CMPLX(0.001, 0.0),     CMPLX(4.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(2048.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.25, 0.0),      CMPLX(16.0, 0.016),    CMPLX(4.0, 0.0),      CMPLX(4.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 2048.0),    CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(8.0, 0.0),
};
static const f64 c5_s[] = {64.0, 0.5, 0.0625, 4.0, 2.0};

/* Case 6: N=4 */
static const c128 c6_a[] = {
    CMPLX(1.0, 1.0),     CMPLX(1e6, 0.0),    CMPLX(1e6, 0.0),     CMPLX(1e6, 0.0),
    CMPLX(-2e6, 0.0),    CMPLX(3.0, 1.0),    CMPLX(2e-6, 0.0),    CMPLX(3e-6, 0.0),
    CMPLX(-3e6, 0.0),    CMPLX(0.0, 0.0),    CMPLX(1e-6, 1.0),    CMPLX(2.0, 0.0),
    CMPLX(1e6, 0.0),     CMPLX(0.0, 0.0),    CMPLX(3e-6, 0.0),    CMPLX(4e6, 1.0),
};
static const c128 c6_ain[] = {
    CMPLX(1.0, 1.0),     CMPLX(1e6, 0.0),    CMPLX(2e6, 0.0),     CMPLX(1e6, 0.0),
    CMPLX(-2e6, 0.0),    CMPLX(3.0, 1.0),    CMPLX(4e-6, 0.0),    CMPLX(3e-6, 0.0),
    CMPLX(-1.5e6, 0.0),  CMPLX(0.0, 0.0),    CMPLX(1e-6, 1.0),    CMPLX(1.0, 0.0),
    CMPLX(1e6, 0.0),     CMPLX(0.0, 0.0),    CMPLX(6e-6, 0.0),    CMPLX(4e6, 1.0),
};
static const f64 c6_s[] = {1.0, 1.0, 2.0, 1.0};

/* Case 7: N=4 */
static const c128 c7_a[] = {
    CMPLX(1.0, 0.0),     CMPLX(0.0, 1e4),    CMPLX(0.0, 1e4),     CMPLX(0.0, 1e4),
    CMPLX(-2e4, 0.0),    CMPLX(3.0, 0.0),    CMPLX(2e-3, 0.0),    CMPLX(3e-3, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 1.0),    CMPLX(0.0, 0.0),     CMPLX(-3e4, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),    CMPLX(1e4, 0.0),     CMPLX(0.0, 0.0),
};
static const c128 c7_ain[] = {
    CMPLX(1.0, 0.0),     CMPLX(0.0, 1e4),    CMPLX(0.0, 1e4),     CMPLX(0.0, 5e3),
    CMPLX(-2e4, 0.0),    CMPLX(3.0, 0.0),    CMPLX(2e-3, 0.0),    CMPLX(1.5e-3, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(2.0, 1.0),    CMPLX(0.0, 0.0),     CMPLX(-1.5e4, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),    CMPLX(2e4, 0.0),     CMPLX(0.0, 0.0),
};
static const f64 c7_s[] = {1.0, 1.0, 1.0, 0.5};

/* Case 8: N=5 */
static const c128 c8_a[] = {
    CMPLX(1.0, 0.0),     CMPLX(512.0, 0.0),    CMPLX(4096.0, 0.0),    CMPLX(32768.0, 0.0),  CMPLX(262144.0, 0.0),
    CMPLX(8.0, 8.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(8.0, 8.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(8.0, 8.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(8.0, 8.0),      CMPLX(0.0, 0.0),
};
static const c128 c8_ain[] = {
    CMPLX(1.0, 0.0),      CMPLX(64.0, 0.0),     CMPLX(64.0, 0.0),      CMPLX(64.0, 0.0),     CMPLX(64.0, 0.0),
    CMPLX(64.0, 64.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(64.0, 64.0),    CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(64.0, 64.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(64.0, 64.0),    CMPLX(0.0, 0.0),
};
static const f64 c8_s[] = {128.0, 16.0, 2.0, 0.25, 0.03125};

/* Case 9: N=6 with isolation */
static const c128 c9_a[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
};
static const c128 c9_ain[] = {
    CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0), CMPLX(1.0, 1.0),
    CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 1.0),
};
static const f64 c9_s[] = {3.0, 1.0, 1.0, 1.0, 1.0, 4.0};

/* Case 10: N=7 */
static const c128 c10_a[] = {
    CMPLX(6.0, 0.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(1.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(4.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(2.5e-4, 0.0),   CMPLX(0.0125, 0.0),   CMPLX(0.02, 0.0),     CMPLX(0.125, 0.0),
    CMPLX(1.0, 0.0),    CMPLX(128.0, 0.0),    CMPLX(64.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(-2.0, 0.0),     CMPLX(16.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(16384.0, 0.0),  CMPLX(0.0, 0.0),      CMPLX(1.0, 0.0),      CMPLX(-400.0, 0.0),   CMPLX(256.0, 0.0),    CMPLX(-4000.0, 0.0),
    CMPLX(-2.0, 0.0),   CMPLX(-256.0, 0.0),   CMPLX(0.0, 0.0),      CMPLX(0.0125, 0.0),   CMPLX(2.0, 0.0),      CMPLX(2.0, 0.0),      CMPLX(32.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),    CMPLX(8.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.004, 0.0),    CMPLX(0.125, 0.0),    CMPLX(-0.2, 0.0),     CMPLX(3.0, 0.0),
};
static const c128 c10_ain[] = {
    CMPLX(64.0, 0.0),    CMPLX(0.25, 0.0),     CMPLX(0.5, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(1.0, 0.0),      CMPLX(-2.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(4.0, 0.0),      CMPLX(2.0, 0.0),       CMPLX(4.096, 0.0),    CMPLX(1.6, 0.0),      CMPLX(0.0, 0.0),      CMPLX(10.24, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.5, 0.0),      CMPLX(3.0, 0.0),       CMPLX(4.096, 0.0),    CMPLX(1.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(-6.4, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(1.0, 0.0),      CMPLX(-3.90625, 0.0),  CMPLX(1.0, 0.0),      CMPLX(-3.125, 0.0),   CMPLX(0.0, 0.0),      CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(-2.0, 0.0),     CMPLX(4.0, 0.0),       CMPLX(1.6, 0.0),      CMPLX(2.0, 0.0),      CMPLX(-8.0, 0.0),     CMPLX(8.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(6.0, 0.0),      CMPLX(1.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
};
static const f64 c10_s[] = {3.0, 1.953125e-3, 0.03125, 32.0, 0.25, 1.0, 6.0};

/* Case 11: N=5 */
static const c128 c11_a[] = {
    CMPLX(1000.0, 0.0),  CMPLX(2.0, 0.0),      CMPLX(3.0, 0.0),      CMPLX(4.0, 0.0),      CMPLX(5e5, 0.0),
    CMPLX(9.0, 0.0),     CMPLX(0.0, 0.0),      CMPLX(2e-4, 0.0),     CMPLX(1.0, 0.0),      CMPLX(3.0, 0.0),
    CMPLX(0.0, 0.0),     CMPLX(-300.0, 0.0),   CMPLX(2.0, 0.0),      CMPLX(1.0, 0.0),      CMPLX(1.0, 0.0),
    CMPLX(9.0, 0.0),     CMPLX(0.002, 0.0),    CMPLX(1.0, 0.0),      CMPLX(1.0, 0.0),      CMPLX(-1000.0, 0.0),
    CMPLX(6.0, 0.0),     CMPLX(200.0, 0.0),    CMPLX(1.0, 0.0),      CMPLX(600.0, 0.0),    CMPLX(3.0, 0.0),
};
static const c128 c11_ain[] = {
    CMPLX(1000.0, 0.0),   CMPLX(0.03125, 0.0),   CMPLX(0.375, 0.0),     CMPLX(0.0625, 0.0),   CMPLX(3906.25, 0.0),
    CMPLX(576.0, 0.0),    CMPLX(0.0, 0.0),       CMPLX(0.0016, 0.0),    CMPLX(1.0, 0.0),      CMPLX(1.5, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(-37.5, 0.0),     CMPLX(2.0, 0.0),       CMPLX(0.125, 0.0),    CMPLX(0.0625, 0.0),
    CMPLX(576.0, 0.0),    CMPLX(0.002, 0.0),     CMPLX(8.0, 0.0),       CMPLX(1.0, 0.0),      CMPLX(-500.0, 0.0),
    CMPLX(768.0, 0.0),    CMPLX(400.0, 0.0),     CMPLX(16.0, 0.0),      CMPLX(1200.0, 0.0),   CMPLX(3.0, 0.0),
};
static const f64 c11_s[] = {128.0, 2.0, 16.0, 2.0, 1.0};

/* Case 12: N=6 extreme magnitudes */
static const c128 c12_a[] = {
    CMPLX(1.0, 0.0),      CMPLX(1e120, 0.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(1e-120, 0.0),   CMPLX(1.0, 0.0),      CMPLX(1e120, 0.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(1e-120, 0.0),   CMPLX(1.0, 0.0),      CMPLX(1e120, 0.0),    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(1e-120, 0.0),   CMPLX(1.0, 0.0),      CMPLX(1e120, 0.0),    CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(1e-120, 0.0),   CMPLX(1.0, 0.0),      CMPLX(1e120, 0.0),
    CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(0.0, 0.0),      CMPLX(1e-120, 0.0),   CMPLX(1.0, 0.0),
};
static const c128 c12_ain[] = {
    CMPLX(1.0, 0.0),                       CMPLX(6.344854593289122931e3, 0.0),  CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),
    CMPLX(1.576080247855779135e-4, 0.0),    CMPLX(1.0, 0.0),                    CMPLX(6.344854593289122931e3, 0.0),     CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),                       CMPLX(1.576080247855779135e-4, 0.0), CMPLX(1.0, 0.0),                       CMPLX(3.172427296644561466e3, 0.0),     CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                    CMPLX(3.152160495711558270e-4, 0.0),    CMPLX(1.0, 0.0),                       CMPLX(1.586213648322280733e3, 0.0),     CMPLX(0.0, 0.0),
    CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                    CMPLX(0.0, 0.0),                       CMPLX(6.304320991423116539e-4, 0.0),    CMPLX(1.0, 0.0),                       CMPLX(1.586213648322280733e3, 0.0),
    CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                    CMPLX(0.0, 0.0),                       CMPLX(0.0, 0.0),                       CMPLX(6.304320991423116539e-4, 0.0),    CMPLX(1.0, 0.0),
};
static const f64 c12_s[] = {
    2.494800386918399765e291,
    1.582914569427869018e175,
    1.004336277661868922e59,
    3.186183822264904554e-58,
    5.053968264940243633e-175,
    8.016673440035891112e-292,
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
    { 6, 0, 5, c12_a, c12_ain, c12_s },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_zgebal(void** state)
{
    (void)state;

    f64 sfmin = dlamch("S");

    c128 a[LDA * LDA], ain[LDA * LDA];
    f64 scale[LDA];
    INT ilo, ihi, info;
    f64 rmax = 0.0, vmax;
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

        zgebal("B", n, a, LDA, &ilo, &ihi, scale, &info);

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
                c128 aij = a[i + j * LDA];
                c128 ainij = ain[i + j * LDA];
                f64 temp = cabs1(aij);
                if (cabs1(ainij) > temp) temp = cabs1(ainij);
                if (sfmin > temp) temp = sfmin;
                f64 diff = cabs1(aij - ainij) / temp;
                if (diff > vmax) vmax = diff;
            }
        }

        for (INT i = 0; i < n; i++) {
            f64 si = scale[i];
            f64 ei = c->scalin[i];
            f64 temp = fabs(si);
            if (fabs(ei) > temp) temp = fabs(ei);
            if (sfmin > temp) temp = sfmin;
            f64 diff = fabs(si - ei) / temp;
            if (diff > vmax) vmax = diff;
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("ZGEBAL: %d cases, max residual = %.3e (case %d)\n",
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
        cmocka_unit_test(test_zgebal),
    };
    return cmocka_run_group_tests_name("zchkbl", tests, NULL, NULL);
}
