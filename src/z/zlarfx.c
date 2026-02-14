/**
 * @file zlarfx.c
 * @brief ZLARFX applies an elementary reflector to a general rectangular
 *        matrix, with loop unrolling when the reflector has order <= 10.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFX applies a complex elementary reflector H to a complex m by n
 * matrix C, from either the left or the right. H is represented in the
 * form
 *
 *       H = I - tau * v * v**H
 *
 * where tau is a complex scalar and v is a complex vector.
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * This version uses inline code if H has order < 11.
 *
 * @param[in]     side   'L': form H * C; 'R': form C * H
 * @param[in]     m      The number of rows of the matrix C.
 * @param[in]     n      The number of columns of the matrix C.
 * @param[in]     v      The vector v in the representation of H.
 *                       Dimension (m) if side = "L", or (n) if side = 'R'.
 * @param[in]     tau    The value tau in the representation of H.
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                       On entry, the m by n matrix C.
 *                       On exit, C is overwritten by H * C if side = "L",
 *                       or C * H if side = 'R'.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Complex array, dimension
 *                       (n) if side = "L", or (m) if side = 'R'.
 *                       Not referenced if H has order < 11.
 */
void zlarfx(const char* side, const int m, const int n,
            const double complex* const restrict v, const double complex tau,
            double complex* const restrict C, const int ldc,
            double complex* restrict work)
{
    int j;
    double complex sum, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    double complex v1, v2, v3, v4, v5, v6, v7, v8, v9, v10;

    if (tau == 0.0) {
        return;
    }

    if (side[0] == 'L' || side[0] == 'l') {

        /* Form  H * C, where H has order m. */

        switch (m) {
        case 1:
            /* Special code for 1 x 1 Householder */
            t1 = 1.0 - tau * v[0] * conj(v[0]);
            for (j = 0; j < n; j++) {
                C[0 + j * ldc] = t1 * C[0 + j * ldc];
            }
            break;

        case 2:
            /* Special code for 2 x 2 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
            }
            break;

        case 3:
            /* Special code for 3 x 3 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
            }
            break;

        case 4:
            /* Special code for 4 x 4 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
            }
            break;

        case 5:
            /* Special code for 5 x 5 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
            }
            break;

        case 6:
            /* Special code for 6 x 6 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            v6 = conj(v[5]);
            t6 = tau * conj(v6);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc] + v6 * C[5 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
                C[5 + j * ldc] -= sum * t6;
            }
            break;

        case 7:
            /* Special code for 7 x 7 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            v6 = conj(v[5]);
            t6 = tau * conj(v6);
            v7 = conj(v[6]);
            t7 = tau * conj(v7);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc] + v6 * C[5 + j * ldc]
                    + v7 * C[6 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
                C[5 + j * ldc] -= sum * t6;
                C[6 + j * ldc] -= sum * t7;
            }
            break;

        case 8:
            /* Special code for 8 x 8 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            v6 = conj(v[5]);
            t6 = tau * conj(v6);
            v7 = conj(v[6]);
            t7 = tau * conj(v7);
            v8 = conj(v[7]);
            t8 = tau * conj(v8);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc] + v6 * C[5 + j * ldc]
                    + v7 * C[6 + j * ldc] + v8 * C[7 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
                C[5 + j * ldc] -= sum * t6;
                C[6 + j * ldc] -= sum * t7;
                C[7 + j * ldc] -= sum * t8;
            }
            break;

        case 9:
            /* Special code for 9 x 9 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            v6 = conj(v[5]);
            t6 = tau * conj(v6);
            v7 = conj(v[6]);
            t7 = tau * conj(v7);
            v8 = conj(v[7]);
            t8 = tau * conj(v8);
            v9 = conj(v[8]);
            t9 = tau * conj(v9);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc] + v6 * C[5 + j * ldc]
                    + v7 * C[6 + j * ldc] + v8 * C[7 + j * ldc]
                    + v9 * C[8 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
                C[5 + j * ldc] -= sum * t6;
                C[6 + j * ldc] -= sum * t7;
                C[7 + j * ldc] -= sum * t8;
                C[8 + j * ldc] -= sum * t9;
            }
            break;

        case 10:
            /* Special code for 10 x 10 Householder */
            v1 = conj(v[0]);
            t1 = tau * conj(v1);
            v2 = conj(v[1]);
            t2 = tau * conj(v2);
            v3 = conj(v[2]);
            t3 = tau * conj(v3);
            v4 = conj(v[3]);
            t4 = tau * conj(v4);
            v5 = conj(v[4]);
            t5 = tau * conj(v5);
            v6 = conj(v[5]);
            t6 = tau * conj(v6);
            v7 = conj(v[6]);
            t7 = tau * conj(v7);
            v8 = conj(v[7]);
            t8 = tau * conj(v8);
            v9 = conj(v[8]);
            t9 = tau * conj(v9);
            v10 = conj(v[9]);
            t10 = tau * conj(v10);
            for (j = 0; j < n; j++) {
                sum = v1 * C[0 + j * ldc] + v2 * C[1 + j * ldc]
                    + v3 * C[2 + j * ldc] + v4 * C[3 + j * ldc]
                    + v5 * C[4 + j * ldc] + v6 * C[5 + j * ldc]
                    + v7 * C[6 + j * ldc] + v8 * C[7 + j * ldc]
                    + v9 * C[8 + j * ldc] + v10 * C[9 + j * ldc];
                C[0 + j * ldc] -= sum * t1;
                C[1 + j * ldc] -= sum * t2;
                C[2 + j * ldc] -= sum * t3;
                C[3 + j * ldc] -= sum * t4;
                C[4 + j * ldc] -= sum * t5;
                C[5 + j * ldc] -= sum * t6;
                C[6 + j * ldc] -= sum * t7;
                C[7 + j * ldc] -= sum * t8;
                C[8 + j * ldc] -= sum * t9;
                C[9 + j * ldc] -= sum * t10;
            }
            break;

        default:
            /* Code for general m */
            zlarf("L", m, n, v, 1, tau, C, ldc, work);
            break;
        }

    } else {

        /* Form  C * H, where H has order n. */

        switch (n) {
        case 1:
            /* Special code for 1 x 1 Householder */
            t1 = 1.0 - tau * v[0] * conj(v[0]);
            for (j = 0; j < m; j++) {
                C[j + 0 * ldc] = t1 * C[j + 0 * ldc];
            }
            break;

        case 2:
            /* Special code for 2 x 2 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
            }
            break;

        case 3:
            /* Special code for 3 x 3 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
            }
            break;

        case 4:
            /* Special code for 4 x 4 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
            }
            break;

        case 5:
            /* Special code for 5 x 5 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
            }
            break;

        case 6:
            /* Special code for 6 x 6 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            v6 = v[5];
            t6 = tau * conj(v6);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc] + v6 * C[j + 5 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
                C[j + 5 * ldc] -= sum * t6;
            }
            break;

        case 7:
            /* Special code for 7 x 7 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            v6 = v[5];
            t6 = tau * conj(v6);
            v7 = v[6];
            t7 = tau * conj(v7);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc] + v6 * C[j + 5 * ldc]
                    + v7 * C[j + 6 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
                C[j + 5 * ldc] -= sum * t6;
                C[j + 6 * ldc] -= sum * t7;
            }
            break;

        case 8:
            /* Special code for 8 x 8 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            v6 = v[5];
            t6 = tau * conj(v6);
            v7 = v[6];
            t7 = tau * conj(v7);
            v8 = v[7];
            t8 = tau * conj(v8);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc] + v6 * C[j + 5 * ldc]
                    + v7 * C[j + 6 * ldc] + v8 * C[j + 7 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
                C[j + 5 * ldc] -= sum * t6;
                C[j + 6 * ldc] -= sum * t7;
                C[j + 7 * ldc] -= sum * t8;
            }
            break;

        case 9:
            /* Special code for 9 x 9 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            v6 = v[5];
            t6 = tau * conj(v6);
            v7 = v[6];
            t7 = tau * conj(v7);
            v8 = v[7];
            t8 = tau * conj(v8);
            v9 = v[8];
            t9 = tau * conj(v9);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc] + v6 * C[j + 5 * ldc]
                    + v7 * C[j + 6 * ldc] + v8 * C[j + 7 * ldc]
                    + v9 * C[j + 8 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
                C[j + 5 * ldc] -= sum * t6;
                C[j + 6 * ldc] -= sum * t7;
                C[j + 7 * ldc] -= sum * t8;
                C[j + 8 * ldc] -= sum * t9;
            }
            break;

        case 10:
            /* Special code for 10 x 10 Householder */
            v1 = v[0];
            t1 = tau * conj(v1);
            v2 = v[1];
            t2 = tau * conj(v2);
            v3 = v[2];
            t3 = tau * conj(v3);
            v4 = v[3];
            t4 = tau * conj(v4);
            v5 = v[4];
            t5 = tau * conj(v5);
            v6 = v[5];
            t6 = tau * conj(v6);
            v7 = v[6];
            t7 = tau * conj(v7);
            v8 = v[7];
            t8 = tau * conj(v8);
            v9 = v[8];
            t9 = tau * conj(v9);
            v10 = v[9];
            t10 = tau * conj(v10);
            for (j = 0; j < m; j++) {
                sum = v1 * C[j + 0 * ldc] + v2 * C[j + 1 * ldc]
                    + v3 * C[j + 2 * ldc] + v4 * C[j + 3 * ldc]
                    + v5 * C[j + 4 * ldc] + v6 * C[j + 5 * ldc]
                    + v7 * C[j + 6 * ldc] + v8 * C[j + 7 * ldc]
                    + v9 * C[j + 8 * ldc] + v10 * C[j + 9 * ldc];
                C[j + 0 * ldc] -= sum * t1;
                C[j + 1 * ldc] -= sum * t2;
                C[j + 2 * ldc] -= sum * t3;
                C[j + 3 * ldc] -= sum * t4;
                C[j + 4 * ldc] -= sum * t5;
                C[j + 5 * ldc] -= sum * t6;
                C[j + 6 * ldc] -= sum * t7;
                C[j + 7 * ldc] -= sum * t8;
                C[j + 8 * ldc] -= sum * t9;
                C[j + 9 * ldc] -= sum * t10;
            }
            break;

        default:
            /* Code for general n */
            zlarf("R", m, n, v, 1, tau, C, ldc, work);
            break;
        }
    }
}
