/**
 * @file dget38.c
 * @brief DGET38 tests DTRSEN, a routine for estimating condition numbers of a
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
    f64 sin_val;
    f64 sepin;
} dget38_meta_t;

static const dget38_meta_t dget38_meta[NCASES38] = {
    /* Case 0 */ {1, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 0.0},
    /* Case 1 */ {1, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.0},
    /* Case 2 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 4.43734e-31},
    /* Case 3 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.19209e-07},
    /* Case 4 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.01235e-36, 3.20988e-36},
    /* Case 5 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.01235e-36, 3.20988e-36},
    /* Case 6 */ {6, 3, {4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.0},
    /* Case 7 */ {2, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.70711, 2.0},
    /* Case 8 */ {4, 2, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.72220, 0.46394},
    /* Case 9 */ {7, 6, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.94322, 3.20530},
    /* Case 10 */ {4, 2, {2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.52287, 0.54553},
    /* Case 11 */ {7, 5, {1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.60473, 0.90039},
    /* Case 12 */ {6, 4, {3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4.89525e-05, 4.56492e-05},
    /* Case 13 */ {8, 4, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 9.56158e-05, 4.14317e-05},
    /* Case 14 */ {9, 3, {1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 5.55801e-07},
    /* Case 15 */ {10, 4, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.0, 1.16972e-10},
    /* Case 16 */ {12, 6, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1.85655e-10, 2.20147e-16},
    /* Case 17 */ {12, 7, {6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 6.92558e-05, 5.52606e-05},
    /* Case 18 */ {3, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.73030, 4.0},
    /* Case 19 */ {5, 1, {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 3.99999e-12, 3.99201e-12},
    /* Case 20 */ {6, 4, {1, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.29329, 0.16345},
    /* Case 21 */ {6, 2, {3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.39736, 0.35829},
    /* Case 22 */ {6, 3, {3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.72893, 0.01246},
    /* Case 23 */ {5, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.21768, 0.52263},
    /* Case 24 */ {6, 2, {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.06789, 0.04220},
    /* Case 25 */ {10, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0.03604, 0.07961},
};

static const f64 dget38_data[] = {
    /* Case 0: N=1 */
    0.0,

    /* Case 1: N=1 */
    1.0,

    /* Case 2: N=6 */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* Case 3: N=6 */
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,

    /* Case 4: N=6 */
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,

    /* Case 5: N=6 */
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0,

    /* Case 6: N=6 */
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 6.0,

    /* Case 7: N=2 */
    1.0, 2.0,
    0.0, 3.0,

    /* Case 8: N=4 */
    0.85240, 0.56110, 0.70430, 0.95400,
    0.27980, 0.72160, 0.96130, 0.35820,
    0.70810, 0.40940, 0.22500, 0.95180,
    0.55430, 0.52200, 0.68600, 0.03070,

    /* Case 9: N=7 */
    0.78180, 0.56570, 0.76210, 0.74360, 0.25530, 0.41000, 0.01340,
    0.64580, 0.26660, 0.55100, 0.83180, 0.92710, 0.62090, 0.78390,
    0.13160, 0.49140, 0.17710, 0.19640, 0.10850, 0.92700, 0.22470,
    0.64100, 0.46890, 0.96590, 0.88840, 0.37690, 0.96730, 0.61830,
    0.83820, 0.87430, 0.45070, 0.94420, 0.77550, 0.96760, 0.78310,
    0.32590, 0.73890, 0.83020, 0.45210, 0.30150, 0.21330, 0.84340,
    0.52440, 0.50160, 0.75290, 0.38380, 0.84790, 0.91280, 0.57700,

    /* Case 10: N=4 */
    -0.98590, 1.47840, -0.13360, -2.95970,
    -0.43370, -0.65400, -0.71550, 1.23760,
    -0.73630, -1.97680, -0.19510, 0.34320,
    0.64140, -1.40880, 0.63940, 0.08580,

    /* Case 11: N=7 */
    2.72840, 0.21520, -1.05200, -0.24460, -0.06530, 0.39050, 1.40980,
    0.97530, 0.65150, -0.47620, 0.54210, 0.62090, 0.47590, -1.44930,
    -0.90520, 0.17900, -0.70860, 0.46210, 1.05800, 2.24260, 1.58260,
    -0.71790, -0.25340, -0.47390, -1.08100, 0.41380, -0.09500, 0.14530,
    -1.37990, -1.06490, 1.25580, 0.78010, -0.64050, -0.08610, 0.08300,
    0.28490, -0.12990, 0.04800, -0.25860, 0.41890, 1.37680, 0.82080,
    -0.54420, 0.97490, 0.95580, 0.12370, 1.09020, -0.14060, 1.90960,

    /* Case 12: N=6 */
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    1.00000e-06, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.50000,

    /* Case 13: N=8 */
    1.0, -1.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0,
    0.0, 1.0, -1.0, 0.0, 0.0, 10.0, 10.0, 0.0,
    0.0, 0.0, 1.0, -1.0, 0.0, 10.0, 10.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 10.0,
    0.0, 0.0, 0.0, 0.0, 0.50000, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.50000, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50000, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50000,

    /* Case 14: N=9 */
    1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75000, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75000, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75000,

    /* Case 15: N=10 */
    1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87500, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87500, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87500, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87500,

    /* Case 16: N=12 */
    1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.93750,

    /* Case 17: N=12 */
    12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    11.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,

    /* Case 18: N=3 */
    2.00000e-06, 1.0, -2.0,
    1.00000e-06, -2.0, 4.0,
    0.0, 1.0, -2.0,

    /* Case 19: N=5 */
    0.0020000, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0010000, 1.0, 0.0, 0.0,
    0.0, 0.0, -0.0010000, 1.0, 0.0,
    0.0, 0.0, 0.0, -0.0020000, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0,

    /* Case 20: N=6 */
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

    /* Case 21: N=6 */
    0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,

    /* Case 22: N=6 */
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.50000, 0.33330, 0.25000, 0.20000, 0.16670, 0.14290,
    0.33330, 0.25000, 0.20000, 0.16670, 0.14290, 0.12500,
    0.25000, 0.20000, 0.16670, 0.14290, 0.12500, 0.11110,
    0.20000, 0.16670, 0.14290, 0.12500, 0.11110, 0.10000,
    0.16670, 0.14290, 0.12500, 0.11110, 0.10000, 0.09090,

    /* Case 23: N=5 */
    15.0, 11.0, 6.0, -9.0, -15.0,
    1.0, 3.0, 9.0, -3.0, -8.0,
    7.0, 6.0, 6.0, -3.0, -11.0,
    7.0, 7.0, 5.0, -3.0, -11.0,
    17.0, 12.0, 5.0, -10.0, -16.0,

    /* Case 24: N=6 */
    -9.0, 21.0, -15.0, 4.0, 2.0, 0.0,
    -10.0, 21.0, -14.0, 4.0, 2.0, 0.0,
    -8.0, 16.0, -11.0, 4.0, 2.0, 0.0,
    -6.0, 12.0, -9.0, 3.0, 3.0, 0.0,
    -4.0, 8.0, -6.0, 0.0, 5.0, 0.0,
    -2.0, 4.0, -3.0, 0.0, 1.0, 3.0,

    /* Case 25: N=10 */
    1.0, 1.0, 1.0, -2.0, 1.0, -1.0, 2.0, -2.0, 4.0, -3.0,
    -1.0, 2.0, 3.0, -4.0, 2.0, -2.0, 4.0, -4.0, 8.0, -6.0,
    -1.0, 0.0, 5.0, -5.0, 3.0, -3.0, 6.0, -6.0, 12.0, -9.0,
    -1.0, 0.0, 3.0, -4.0, 4.0, -4.0, 8.0, -8.0, 16.0, -12.0,
    -1.0, 0.0, 3.0, -6.0, 5.0, -4.0, 10.0, -10.0, 20.0, -15.0,
    -1.0, 0.0, 3.0, -6.0, 2.0, -2.0, 12.0, -12.0, 24.0, -18.0,
    -1.0, 0.0, 3.0, -6.0, 2.0, -5.0, 15.0, -13.0, 28.0, -21.0,
    -1.0, 0.0, 3.0, -6.0, 2.0, -5.0, 12.0, -11.0, 32.0, -24.0,
    -1.0, 0.0, 3.0, -6.0, 2.0, -5.0, 12.0, -14.0, 37.0, -26.0,
    -1.0, 0.0, 3.0, -6.0, 2.0, -5.0, 12.0, -14.0, 36.0, -25.0,
};

static void rowmajor_to_colmajor(const f64* rows, f64* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(f64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

void dget38(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 EPSIN = 5.9605e-8;

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    eps = fmax(eps, EPSIN);
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

    f64 val[3];
    val[0] = sqrt(smlnum);
    val[1] = ONE;
    val[2] = sqrt(sqrt(bignum));

    INT select[LDT], ipnt[LDT], iwork[LIWORK];
    f64 q[LDT * LDT], qsav[LDT * LDT], qtmp[LDT * LDT];
    f64 t[LDT * LDT], tmp[LDT * LDT], tsav[LDT * LDT];
    f64 tsav1[LDT * LDT], ttmp[LDT * LDT];
    f64 wi[LDT], witmp[LDT], wr[LDT], wrtmp[LDT];
    f64 work[LWORK], result[2];

    const f64* dptr = dget38_data;

    for (INT ic = 0; ic < NCASES38; ic++) {
        INT n = dget38_meta[ic].n;
        INT ndim = dget38_meta[ic].ndim;
        f64 sin_val = dget38_meta[ic].sin_val;
        f64 sepin = dget38_meta[ic].sepin;

        rowmajor_to_colmajor(dptr, tmp, n, LDT);
        dptr += n * n;

        f64 tnrm = dlange("M", n, n, tmp, LDT, work);

        for (INT iscl = 0; iscl < 3; iscl++) {

            (*knt)++;
            dlacpy("F", n, n, tmp, LDT, t, LDT);
            f64 vmul = val[iscl];
            for (INT i = 0; i < n; i++)
                cblas_dscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;
            dlacpy("F", n, n, t, LDT, tsav, LDT);

            INT info;
            dgehrd(n, 0, n - 1, t, LDT, work, &work[n], LWORK - n, &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0]++;
                continue;
            }

            dlacpy("L", n, n, t, LDT, q, LDT);
            dorghr(n, 0, n - 1, q, LDT, work, &work[n], LWORK - n, &info);

            dhseqr("S", "V", n, 0, n - 1, t, LDT, wr, wi, q, LDT,
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
            cblas_dcopy(n, wr, 1, wrtmp, 1);
            cblas_dcopy(n, wi, 1, witmp, 1);
            for (INT i = 0; i < n - 1; i++) {
                INT kmin = i;
                f64 vrmin = wrtmp[i];
                f64 vimin = witmp[i];
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

            dlacpy("F", n, n, q, LDT, qsav, LDT);
            dlacpy("F", n, n, t, LDT, tsav1, LDT);
            INT m;
            f64 s, sep;
            dtrsen("B", "V", select, n, t, LDT, q, LDT, wrtmp, witmp,
                   &m, &s, &sep, work, LWORK, iwork, LIWORK, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2]++;
                continue;
            }
            f64 septmp = sep / vmul;
            f64 stmp = s;

            dhst01(n, 0, n - 1, tsav, LDT, t, LDT, q, LDT, work, LWORK,
                   result);
            f64 vmax = fmax(result[0], result[1]);
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            f64 v = fmax(TWO * (f64)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            f64 tol, tolin;
            if (v > septmp)
                tol = ONE;
            else
                tol = v / septmp;
            if (v > sepin)
                tolin = ONE;
            else
                tolin = v / sepin;
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
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
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
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

            if (sin_val <= (f64)(2 * n) * eps && stmp <= (f64)(2 * n) * eps)
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
            dlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            dlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            dtrsen("E", "V", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
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

            dlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            dlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            dtrsen("V", "V", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
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

            dlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            dlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            dtrsen("E", "N", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
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

            dlacpy("F", n, n, tsav1, LDT, ttmp, LDT);
            dlacpy("F", n, n, qsav, LDT, qtmp, LDT);
            septmp = -ONE;
            stmp = -ONE;
            dtrsen("V", "N", select, n, ttmp, LDT, qtmp, LDT, wrtmp,
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
