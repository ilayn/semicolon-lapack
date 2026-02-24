/**
 * @file sget38.c
 * @brief SGET38 tests STRSEN, a routine for estimating condition numbers of a
 *        cluster of eigenvalues and/or its associated right invariant subspace.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define LDT 20
#define LWORK (2 * LDT * (10 + LDT))
#define LIWORK (LDT * LDT)
#define NCASES38 26

typedef struct {
    INT n;
    INT ndim;
    INT iselec[LDT];  /* 1-based indices from Fortran */
    f32 sin_val;
    f32 sepin;
} dget38_meta_t;

static const dget38_meta_t dget38_meta[NCASES38] = {
    /* Case 0 */ {1, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 0.0f},
    /* Case 1 */ {1, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.0f},
    /* Case 2 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 4.43734e-31f},
    /* Case 3 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.19209e-07f},
    /* Case 4 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.01235e-36f, 3.20988e-36f},
    /* Case 5 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.01235e-36f, 3.20988e-36f},
    /* Case 6 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.0f},
    /* Case 7 */ {2, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.70711f, 2.0f},
    /* Case 8 */ {4, 2, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.72220f, 0.46394f},
    /* Case 9 */ {7, 6, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.94322f, 3.20530f},
    /* Case 10 */ {4, 2, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.52287f, 0.54553f},
    /* Case 11 */ {7, 5, {1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.60473f, 0.90039f},
    /* Case 12 */ {6, 4, {3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.89525e-05f, 4.56492e-05f},
    /* Case 13 */ {8, 4, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.56158e-05f, 4.14317e-05f},
    /* Case 14 */ {9, 3, {1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 5.55801e-07f},
    /* Case 15 */ {10, 4, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0f, 1.16972e-10f},
    /* Case 16 */ {12, 6, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.85655e-10f, 2.20147e-16f},
    /* Case 17 */ {12, 7, {6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 6.92558e-05f, 5.52606e-05f},
    /* Case 18 */ {3, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.73030f, 4.0f},
    /* Case 19 */ {5, 1, {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.99999e-12f, 3.99201e-12f},
    /* Case 20 */ {6, 4, {1, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.29329f, 0.16345f},
    /* Case 21 */ {6, 2, {3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.39736f, 0.35829f},
    /* Case 22 */ {6, 3, {3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.72893f, 0.01246f},
    /* Case 23 */ {5, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.21768f, 0.52263f},
    /* Case 24 */ {6, 2, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.06789f, 0.04220f},
    /* Case 25 */ {10, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.03604f, 0.07961f},
};

static const f32 dget38_data[] = {
    /* Case 0: N=1 */
    0.0f,

    /* Case 1: N=1 */
    1.0f,

    /* Case 2: N=6 */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

    /* Case 3: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,

    /* Case 4: N=6 */
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,

    /* Case 5: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,

    /* Case 6: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f,

    /* Case 7: N=2 */
    1.0f, 2.0f,
    0.0f, 3.0f,

    /* Case 8: N=4 */
    0.85240f, 0.56110f, 0.70430f, 0.95400f,
    0.27980f, 0.72160f, 0.96130f, 0.35820f,
    0.70810f, 0.40940f, 0.22500f, 0.95180f,
    0.55430f, 0.52200f, 0.68600f, 0.03070f,

    /* Case 9: N=7 */
    0.78180f, 0.56570f, 0.76210f, 0.74360f, 0.25530f, 0.41000f, 0.01340f,
    0.64580f, 0.26660f, 0.55100f, 0.83180f, 0.92710f, 0.62090f, 0.78390f,
    0.13160f, 0.49140f, 0.17710f, 0.19640f, 0.10850f, 0.92700f, 0.22470f,
    0.64100f, 0.46890f, 0.96590f, 0.88840f, 0.37690f, 0.96730f, 0.61830f,
    0.83820f, 0.87430f, 0.45070f, 0.94420f, 0.77550f, 0.96760f, 0.78310f,
    0.32590f, 0.73890f, 0.83020f, 0.45210f, 0.30150f, 0.21330f, 0.84340f,
    0.52440f, 0.50160f, 0.75290f, 0.38380f, 0.84790f, 0.91280f, 0.57700f,

    /* Case 10: N=4 */
    -0.98590f, 1.47840f, -0.13360f, -2.95970f,
    -0.43370f, -0.65400f, -0.71550f, 1.23760f,
    -0.73630f, -1.97680f, -0.19510f, 0.34320f,
    0.64140f, -1.40880f, 0.63940f, 0.08580f,

    /* Case 11: N=7 */
    2.72840f, 0.21520f, -1.05200f, -0.24460f, -0.06530f, 0.39050f, 1.40980f,
    0.97530f, 0.65150f, -0.47620f, 0.54210f, 0.62090f, 0.47590f, -1.44930f,
    -0.90520f, 0.17900f, -0.70860f, 0.46210f, 1.05800f, 2.24260f, 1.58260f,
    -0.71790f, -0.25340f, -0.47390f, -1.08100f, 0.41380f, -0.09500f, 0.14530f,
    -1.37990f, -1.06490f, 1.25580f, 0.78010f, -0.64050f, -0.08610f, 0.08300f,
    0.28490f, -0.12990f, 0.04800f, -0.25860f, 0.41890f, 1.37680f, 0.82080f,
    -0.54420f, 0.97490f, 0.95580f, 0.12370f, 1.09020f, -0.14060f, 1.90960f,

    /* Case 12: N=6 */
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    1.00000e-06f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.50000f,

    /* Case 13: N=8 */
    1.0f, -1.0f, 0.0f, 0.0f, 10.0f, 0.0f, 10.0f, 0.0f,
    0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 10.0f, 10.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 10.0f, 10.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 10.0f, 0.0f, 10.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.50000f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.50000f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.50000f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.50000f,

    /* Case 14: N=9 */
    1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.75000f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.75000f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.75000f,

    /* Case 15: N=10 */
    1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.87500f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.87500f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.87500f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.87500f,

    /* Case 16: N=12 */
    1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.93750f,

    /* Case 17: N=12 */
    12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    11.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 10.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 9.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,

    /* Case 18: N=3 */
    2.00000e-06f, 1.0f, -2.0f,
    1.00000e-06f, -2.0f, 4.0f,
    0.0f, 1.0f, -2.0f,

    /* Case 19: N=5 */
    0.0020000f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0010000f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.0010000f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, -0.0020000f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

    /* Case 20: N=6 */
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

    /* Case 21: N=6 */
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 22: N=6 */
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.50000f, 0.33330f, 0.25000f, 0.20000f, 0.16670f, 0.14290f,
    0.33330f, 0.25000f, 0.20000f, 0.16670f, 0.14290f, 0.12500f,
    0.25000f, 0.20000f, 0.16670f, 0.14290f, 0.12500f, 0.11110f,
    0.20000f, 0.16670f, 0.14290f, 0.12500f, 0.11110f, 0.10000f,
    0.16670f, 0.14290f, 0.12500f, 0.11110f, 0.10000f, 0.09090f,

    /* Case 23: N=5 */
    15.0f, 11.0f, 6.0f, -9.0f, -15.0f,
    1.0f, 3.0f, 9.0f, -3.0f, -8.0f,
    7.0f, 6.0f, 6.0f, -3.0f, -11.0f,
    7.0f, 7.0f, 5.0f, -3.0f, -11.0f,
    17.0f, 12.0f, 5.0f, -10.0f, -16.0f,

    /* Case 24: N=6 */
    -9.0f, 21.0f, -15.0f, 4.0f, 2.0f, 0.0f,
    -10.0f, 21.0f, -14.0f, 4.0f, 2.0f, 0.0f,
    -8.0f, 16.0f, -11.0f, 4.0f, 2.0f, 0.0f,
    -6.0f, 12.0f, -9.0f, 3.0f, 3.0f, 0.0f,
    -4.0f, 8.0f, -6.0f, 0.0f, 5.0f, 0.0f,
    -2.0f, 4.0f, -3.0f, 0.0f, 1.0f, 3.0f,

    /* Case 25: N=10 */
    1.0f, 1.0f, 1.0f, -2.0f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -3.0f,
    -1.0f, 2.0f, 3.0f, -4.0f, 2.0f, -2.0f, 4.0f, -4.0f, 8.0f, -6.0f,
    -1.0f, 0.0f, 5.0f, -5.0f, 3.0f, -3.0f, 6.0f, -6.0f, 12.0f, -9.0f,
    -1.0f, 0.0f, 3.0f, -4.0f, 4.0f, -4.0f, 8.0f, -8.0f, 16.0f, -12.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 5.0f, -4.0f, 10.0f, -10.0f, 20.0f, -15.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -2.0f, 12.0f, -12.0f, 24.0f, -18.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 15.0f, -13.0f, 28.0f, -21.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -11.0f, 32.0f, -24.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -14.0f, 37.0f, -26.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -14.0f, 36.0f, -25.0f,
};

static void rowmajor_to_colmajor(const f32* rows, f32* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(f32));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

void sget38(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 EPSIN = 5.9605e-8f;

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    eps = fmaxf(eps, EPSIN);
    rmax[0] = ZERO;
    rmax[1] = ZERO;
    rmax[2] = ZERO;
    lmax[0] = 0;
    lmax[1] = 0;
    lmax[2] = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    ninfo[2] = 0;

    f32 val[3];
    val[0] = sqrtf(smlnum);
    val[1] = ONE;
    val[2] = sqrtf(sqrtf(bignum));

    INT select[LDT], ipnt[LDT], iwork[LIWORK];
    f32 q[LDT * LDT], qsav[LDT * LDT], qtmp[LDT * LDT];
    f32 t[LDT * LDT], tmp[LDT * LDT], tsav[LDT * LDT];
    f32 tsav1[LDT * LDT], ttmp[LDT * LDT];
    f32 wi[LDT], witmp[LDT], wr[LDT], wrtmp[LDT];
    f32 work[LWORK], result[2];

    const f32* dptr = dget38_data;

    for (INT ic = 0; ic < NCASES38; ic++) {
        INT n = dget38_meta[ic].n;
        INT ndim = dget38_meta[ic].ndim;
        f32 sin_val = dget38_meta[ic].sin_val;
        f32 sepin = dget38_meta[ic].sepin;

        rowmajor_to_colmajor(dptr, tmp, n, LDT);
        dptr += n * n;

        f32 tnrm = slange("M", n, n, tmp, LDT, work);

        for (INT iscl = 0; iscl < 3; iscl++) {

            (*knt)++;
            slacpy("F", n, n, tmp, LDT, t, LDT);
            f32 vmul = val[iscl];
            for (INT i = 0; i < n; i++)
                cblas_sscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;
            slacpy("F", n, n, t, LDT, tsav, LDT);

            INT info;
            sgehrd(n, 0, n - 1, t, LDT, work, &work[n], LWORK - n, &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0]++;
                continue;
            }

            slacpy("L", n, n, t, LDT, q, LDT);
            sorghr(n, 0, n - 1, q, LDT, work, &work[n], LWORK - n, &info);

            shseqr("S", "V", n, 0, n - 1, t, LDT, wr, wi, q, LDT,
                   work, LWORK, &info);
            if (info != 0) {
                lmax[1] = *knt;
                ninfo[1]++;
                continue;
            }

            for (INT i = 0; i < n; i++) {
                ipnt[i] = i;
                select[i] = 0;
            }
            cblas_scopy(n, wr, 1, wrtmp, 1);
            cblas_scopy(n, wi, 1, witmp, 1);
            for (INT i = 0; i < n - 1; i++) {
                INT kmin = i;
                f32 vrmin = wrtmp[i];
                f32 vimin = witmp[i];
                for (INT j = i + 1; j < n; j++) {
                    if (wrtmp[j] < vrmin) {
                        kmin = j;
                        vrmin = wrtmp[j];
                        vimin = witmp[j];
                    }
                }
                wrtmp[kmin] = wrtmp[i];
                witmp[kmin] = witmp[i];
                wrtmp[i] = vrmin;
                witmp[i] = vimin;
                INT itmp = ipnt[i];
                ipnt[i] = ipnt[kmin];
                ipnt[kmin] = itmp;
            }
            for (INT i = 0; i < ndim; i++)
                select[ipnt[dget38_meta[ic].iselec[i] - 1]] = 1;

            slacpy("F", n, n, q, LDT, qsav, LDT);
            slacpy("F", n, n, t, LDT, tsav1, LDT);
            INT m;
            f32 s, sep;
            strsen("B", "V", select, n, t, LDT, q, LDT, wrtmp, witmp,
                   &m, &s, &sep, work, LWORK, iwork, LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            f32 septmp = sep / vmul;
            f32 stmp = s;

            shst01(n, 0, n - 1, tsav, LDT, t, LDT, q, LDT, work, LWORK,
                   result);
            f32 vmax = fmaxf(result[0], result[1]);
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            f32 v = fmaxf(TWO * (f32)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            f32 tol, tolin;
            if (v > septmp)
                tol = ONE;
            else
                tol = v / septmp;
            if (v > sepin)
                tolin = ONE;
            else
                tolin = v / sepin;
            tol = fmaxf(tol, smlnum / eps);
            tolin = fmaxf(tolin, smlnum / eps);
            if (eps * (sin_val - tolin) > stmp + tol)
                vmax = ONE / eps;
            else if (sin_val - tolin > stmp + tol)
                vmax = (sin_val - tolin) / (stmp + tol);
            else if (sin_val + tolin < eps * (stmp - tol))
                vmax = ONE / eps;
            else if (sin_val + tolin < stmp - tol)
                vmax = (stmp - tol) / (sin_val + tolin);
            else
                vmax = ONE;
            if (vmax > rmax[1]) {
                rmax[1] = vmax;
                if (ninfo[1] == 0)
                    lmax[1] = *knt;
            }

            if (v > septmp * stmp)
                tol = septmp;
            else
                tol = v / stmp;
            if (v > sepin * sin_val)
                tolin = sepin;
            else
                tolin = v / sin_val;
            tol = fmaxf(tol, smlnum / eps);
            tolin = fmaxf(tolin, smlnum / eps);
            if (eps * (sepin - tolin) > septmp + tol)
                vmax = ONE / eps;
            else if (sepin - tolin > septmp + tol)
                vmax = (sepin - tolin) / (septmp + tol);
            else if (sepin + tolin < eps * (septmp - tol))
                vmax = ONE / eps;
            else if (sepin + tolin < septmp - tol)
                vmax = (septmp - tol) / (sepin + tolin);
            else
                vmax = ONE;
            if (vmax > rmax[1]) {
                rmax[1] = vmax;
                if (ninfo[1] == 0)
                    lmax[1] = *knt;
            }

            if (sin_val <= (f32)(2 * n) * eps && stmp <= (f32)(2 * n) * eps)
                vmax = ONE;
            else if (eps * sin_val > stmp)
                vmax = ONE / eps;
            else if (sin_val > stmp)
                vmax = sin_val / stmp;
            else if (sin_val < eps * stmp)
                vmax = ONE / eps;
            else if (sin_val < stmp)
                vmax = stmp / sin_val;
            else
                vmax = ONE;
            if (vmax > rmax[2]) {
                rmax[2] = vmax;
                if (ninfo[2] == 0)
                    lmax[2] = *knt;
            }

            if (sepin <= v && septmp <= v)
                vmax = ONE;
            else if (eps * sepin > septmp)
                vmax = ONE / eps;
            else if (sepin > septmp)
                vmax = sepin / septmp;
            else if (sepin < eps * septmp)
                vmax = ONE / eps;
            else if (sepin < septmp)
                vmax = septmp / sepin;
            else
                vmax = ONE;
            if (vmax > rmax[2]) {
                rmax[2] = vmax;
                if (ninfo[2] == 0)
                    lmax[2] = *knt;
            }

            vmax = ZERO;
            slacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            slacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            strsen("E", "V", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
                   witmp, &m, &stmp, &septmp, work, LWORK, iwork,
                   LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (s != stmp)
                vmax = ONE / eps;
            if (-ONE != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (ttmp[i + j * LDT] != t[i + j * LDT])
                        vmax = ONE / eps;
                    if (qtmp[i + j * LDT] != q[i + j * LDT])
                        vmax = ONE / eps;
                }

            slacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            slacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            strsen("V", "V", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
                   witmp, &m, &stmp, &septmp, work, LWORK, iwork,
                   LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (-ONE != stmp)
                vmax = ONE / eps;
            if (sep != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (ttmp[i + j * LDT] != t[i + j * LDT])
                        vmax = ONE / eps;
                    if (qtmp[i + j * LDT] != q[i + j * LDT])
                        vmax = ONE / eps;
                }

            slacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            slacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            strsen("E", "N", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
                   witmp, &m, &stmp, &septmp, work, LWORK, iwork,
                   LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (s != stmp)
                vmax = ONE / eps;
            if (-ONE != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (ttmp[i + j * LDT] != t[i + j * LDT])
                        vmax = ONE / eps;
                    if (qtmp[i + j * LDT] != qsav[i + j * LDT])
                        vmax = ONE / eps;
                }

            slacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            slacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            strsen("V", "N", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
                   witmp, &m, &stmp, &septmp, work, LWORK, iwork,
                   LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            if (-ONE != stmp)
                vmax = ONE / eps;
            if (sep != septmp)
                vmax = ONE / eps;
            for (INT i = 0; i < n; i++)
                for (INT j = 0; j < n; j++) {
                    if (ttmp[i + j * LDT] != t[i + j * LDT])
                        vmax = ONE / eps;
                    if (qtmp[i + j * LDT] != qsav[i + j * LDT])
                        vmax = ONE / eps;
                }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }
        }
    }
}
