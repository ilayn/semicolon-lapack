/**
 * @file clatsy.c
 * @brief CLATSY generates a special test matrix for the complex symmetric
 *        indefinite factorization.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void clatsy(const char* uplo, const INT n, c64* X, const INT ldx,
            uint64_t state[static 4])
{
    const c64 EYE = CMPLXF(0.0f, 1.0f);

    INT i, j, n5;
    f32 alpha, alpha3, beta;
    c64 a, b, c, r;

    /*     Initialize constants */

    alpha = (1.0f + sqrtf(17.0f)) / 8.0f;
    beta = alpha - 1.0f / 1000.0f;
    alpha3 = alpha * alpha * alpha;

    /*     UPLO = 'U':  Upper triangular storage */

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /*        Fill the upper triangle of the matrix with zeros. */

        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                X[i + j * ldx] = CMPLXF(0.0f, 0.0f);
            }
        }
        n5 = n / 5;
        n5 = n - 5 * n5 + 1;

        for (i = n - 1; i >= n5 - 1; i -= 5) {
            a = alpha3 * clarnd_rng(5, state);
            b = clarnd_rng(5, state) / alpha;
            c = a - 2.0f * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i - 2) + i * ldx] = b;
            X[(i - 2) + (i - 1) * ldx] = r;
            X[(i - 2) + (i - 2) * ldx] = c;
            X[(i - 1) + (i - 1) * ldx] = clarnd_rng(2, state);
            X[(i - 3) + (i - 3) * ldx] = clarnd_rng(2, state);
            X[(i - 4) + (i - 4) * ldx] = clarnd_rng(2, state);
            if (cabsf(X[(i - 3) + (i - 3) * ldx]) > cabsf(X[(i - 4) + (i - 4) * ldx])) {
                X[(i - 4) + (i - 3) * ldx] = 2.0f * X[(i - 3) + (i - 3) * ldx];
            } else {
                X[(i - 4) + (i - 3) * ldx] = 2.0f * X[(i - 4) + (i - 4) * ldx];
            }
        }

        /*        Clean-up for N not a multiple of 5. */

        i = n5 - 2;
        if (i > 1) {
            a = alpha3 * clarnd_rng(5, state);
            b = clarnd_rng(5, state) / alpha;
            c = a - 2.0f * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i - 2) + i * ldx] = b;
            X[(i - 2) + (i - 1) * ldx] = r;
            X[(i - 2) + (i - 2) * ldx] = c;
            X[(i - 1) + (i - 1) * ldx] = clarnd_rng(2, state);
            i = i - 3;
        }
        if (i > 0) {
            X[i + i * ldx] = clarnd_rng(2, state);
            X[(i - 1) + (i - 1) * ldx] = clarnd_rng(2, state);
            if (cabsf(X[i + i * ldx]) > cabsf(X[(i - 1) + (i - 1) * ldx])) {
                X[(i - 1) + i * ldx] = 2.0f * X[i + i * ldx];
            } else {
                X[(i - 1) + i * ldx] = 2.0f * X[(i - 1) + (i - 1) * ldx];
            }
            i = i - 2;
        } else if (i == 0) {
            X[i + i * ldx] = clarnd_rng(2, state);
            i = i - 1;
        }

    /*     UPLO = 'L':  Lower triangular storage */

    } else {

        /*        Fill the lower triangle of the matrix with zeros. */

        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                X[i + j * ldx] = CMPLXF(0.0f, 0.0f);
            }
        }
        n5 = n / 5;
        n5 = n5 * 5;

        for (i = 0; i < n5; i += 5) {
            a = alpha3 * clarnd_rng(5, state);
            b = clarnd_rng(5, state) / alpha;
            c = a - 2.0f * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i + 2) + i * ldx] = b;
            X[(i + 2) + (i + 1) * ldx] = r;
            X[(i + 2) + (i + 2) * ldx] = c;
            X[(i + 1) + (i + 1) * ldx] = clarnd_rng(2, state);
            X[(i + 3) + (i + 3) * ldx] = clarnd_rng(2, state);
            X[(i + 4) + (i + 4) * ldx] = clarnd_rng(2, state);
            if (cabsf(X[(i + 3) + (i + 3) * ldx]) > cabsf(X[(i + 4) + (i + 4) * ldx])) {
                X[(i + 4) + (i + 3) * ldx] = 2.0f * X[(i + 3) + (i + 3) * ldx];
            } else {
                X[(i + 4) + (i + 3) * ldx] = 2.0f * X[(i + 4) + (i + 4) * ldx];
            }
        }

        /*        Clean-up for N not a multiple of 5. */

        i = n5;
        if (i < n - 2) {
            a = alpha3 * clarnd_rng(5, state);
            b = clarnd_rng(5, state) / alpha;
            c = a - 2.0f * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i + 2) + i * ldx] = b;
            X[(i + 2) + (i + 1) * ldx] = r;
            X[(i + 2) + (i + 2) * ldx] = c;
            X[(i + 1) + (i + 1) * ldx] = clarnd_rng(2, state);
            i = i + 3;
        }
        if (i < n - 1) {
            X[i + i * ldx] = clarnd_rng(2, state);
            X[(i + 1) + (i + 1) * ldx] = clarnd_rng(2, state);
            if (cabsf(X[i + i * ldx]) > cabsf(X[(i + 1) + (i + 1) * ldx])) {
                X[(i + 1) + i * ldx] = 2.0f * X[i + i * ldx];
            } else {
                X[(i + 1) + i * ldx] = 2.0f * X[(i + 1) + (i + 1) * ldx];
            }
            i = i + 2;
        } else if (i == n - 1) {
            X[i + i * ldx] = clarnd_rng(2, state);
            i = i + 1;
        }
    }
}
