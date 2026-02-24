/**
 * @file zlatsy.c
 * @brief ZLATSY generates a special test matrix for the complex symmetric
 *        indefinite factorization.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void zlatsy(const char* uplo, const INT n, c128* X, const INT ldx,
            uint64_t state[static 4])
{
    const c128 EYE = CMPLX(0.0, 1.0);

    INT i, j, n5;
    f64 alpha, alpha3, beta;
    c128 a, b, c, r;

    /*     Initialize constants */

    alpha = (1.0 + sqrt(17.0)) / 8.0;
    beta = alpha - 1.0 / 1000.0;
    alpha3 = alpha * alpha * alpha;

    /*     UPLO = 'U':  Upper triangular storage */

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /*        Fill the upper triangle of the matrix with zeros. */

        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                X[i + j * ldx] = CMPLX(0.0, 0.0);
            }
        }
        n5 = n / 5;
        n5 = n - 5 * n5 + 1;

        for (i = n - 1; i >= n5 - 1; i -= 5) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i - 2) + i * ldx] = b;
            X[(i - 2) + (i - 1) * ldx] = r;
            X[(i - 2) + (i - 2) * ldx] = c;
            X[(i - 1) + (i - 1) * ldx] = zlarnd_rng(2, state);
            X[(i - 3) + (i - 3) * ldx] = zlarnd_rng(2, state);
            X[(i - 4) + (i - 4) * ldx] = zlarnd_rng(2, state);
            if (cabs(X[(i - 3) + (i - 3) * ldx]) > cabs(X[(i - 4) + (i - 4) * ldx])) {
                X[(i - 4) + (i - 3) * ldx] = 2.0 * X[(i - 3) + (i - 3) * ldx];
            } else {
                X[(i - 4) + (i - 3) * ldx] = 2.0 * X[(i - 4) + (i - 4) * ldx];
            }
        }

        /*        Clean-up for N not a multiple of 5. */

        i = n5 - 2;
        if (i > 1) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i - 2) + i * ldx] = b;
            X[(i - 2) + (i - 1) * ldx] = r;
            X[(i - 2) + (i - 2) * ldx] = c;
            X[(i - 1) + (i - 1) * ldx] = zlarnd_rng(2, state);
            i = i - 3;
        }
        if (i > 0) {
            X[i + i * ldx] = zlarnd_rng(2, state);
            X[(i - 1) + (i - 1) * ldx] = zlarnd_rng(2, state);
            if (cabs(X[i + i * ldx]) > cabs(X[(i - 1) + (i - 1) * ldx])) {
                X[(i - 1) + i * ldx] = 2.0 * X[i + i * ldx];
            } else {
                X[(i - 1) + i * ldx] = 2.0 * X[(i - 1) + (i - 1) * ldx];
            }
            i = i - 2;
        } else if (i == 0) {
            X[i + i * ldx] = zlarnd_rng(2, state);
            i = i - 1;
        }

    /*     UPLO = 'L':  Lower triangular storage */

    } else {

        /*        Fill the lower triangle of the matrix with zeros. */

        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                X[i + j * ldx] = CMPLX(0.0, 0.0);
            }
        }
        n5 = n / 5;
        n5 = n5 * 5;

        for (i = 0; i < n5; i += 5) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i + 2) + i * ldx] = b;
            X[(i + 2) + (i + 1) * ldx] = r;
            X[(i + 2) + (i + 2) * ldx] = c;
            X[(i + 1) + (i + 1) * ldx] = zlarnd_rng(2, state);
            X[(i + 3) + (i + 3) * ldx] = zlarnd_rng(2, state);
            X[(i + 4) + (i + 4) * ldx] = zlarnd_rng(2, state);
            if (cabs(X[(i + 3) + (i + 3) * ldx]) > cabs(X[(i + 4) + (i + 4) * ldx])) {
                X[(i + 4) + (i + 3) * ldx] = 2.0 * X[(i + 3) + (i + 3) * ldx];
            } else {
                X[(i + 4) + (i + 3) * ldx] = 2.0 * X[(i + 4) + (i + 4) * ldx];
            }
        }

        /*        Clean-up for N not a multiple of 5. */

        i = n5;
        if (i < n - 2) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[i + i * ldx] = a;
            X[(i + 2) + i * ldx] = b;
            X[(i + 2) + (i + 1) * ldx] = r;
            X[(i + 2) + (i + 2) * ldx] = c;
            X[(i + 1) + (i + 1) * ldx] = zlarnd_rng(2, state);
            i = i + 3;
        }
        if (i < n - 1) {
            X[i + i * ldx] = zlarnd_rng(2, state);
            X[(i + 1) + (i + 1) * ldx] = zlarnd_rng(2, state);
            if (cabs(X[i + i * ldx]) > cabs(X[(i + 1) + (i + 1) * ldx])) {
                X[(i + 1) + i * ldx] = 2.0 * X[i + i * ldx];
            } else {
                X[(i + 1) + i * ldx] = 2.0 * X[(i + 1) + (i + 1) * ldx];
            }
            i = i + 2;
        } else if (i == n - 1) {
            X[i + i * ldx] = zlarnd_rng(2, state);
            i = i + 1;
        }
    }
}
