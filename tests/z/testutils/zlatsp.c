/**
 * @file zlatsp.c
 * @brief ZLATSP generates a special test matrix for the complex symmetric
 *        indefinite factorization.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void zlatsp(const char* uplo, const INT n, c128* X,
            uint64_t state[static 4])
{
    const c128 EYE = CMPLX(0.0, 1.0);

    INT j, jj, n5;
    f64 alpha, alpha3, beta;
    c128 a, b, c, r;

    /*     Initialize constants */

    alpha = (1.0 + sqrt(17.0)) / 8.0;
    beta = alpha - 1.0 / 1000.0;
    alpha3 = alpha * alpha * alpha;

    /*     Fill the matrix with zeros. */

    for (j = 0; j < n * (n + 1) / 2; j++) {
        X[j] = 0.0;
    }

    /*     UPLO = 'U':  Upper triangular storage */

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        n5 = n / 5;
        n5 = n - 5 * n5 + 1;

        jj = n * (n + 1) / 2 - 1;
        for (j = n - 1; j >= n5 - 1; j -= 5) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[jj] = a;
            X[jj - 2] = b;
            jj = jj - (j + 1);
            X[jj] = zlarnd_rng(2, state);
            X[jj - 1] = r;
            jj = jj - j;
            X[jj] = c;
            jj = jj - (j - 1);
            X[jj] = zlarnd_rng(2, state);
            jj = jj - (j - 2);
            X[jj] = zlarnd_rng(2, state);
            if (cabs(X[jj + (j - 2)]) > cabs(X[jj])) {
                X[jj + (j - 3)] = 2.0 * X[jj + (j - 2)];
            } else {
                X[jj + (j - 3)] = 2.0 * X[jj];
            }
            jj = jj - (j - 3);
        }

        /*        Clean-up for N not a multiple of 5. */

        j = n5 - 2;
        if (j > 1) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[jj] = a;
            X[jj - 2] = b;
            jj = jj - (j + 1);
            X[jj] = zlarnd_rng(2, state);
            X[jj - 1] = r;
            jj = jj - j;
            X[jj] = c;
            jj = jj - (j - 1);
            j = j - 3;
        }
        if (j > 0) {
            X[jj] = zlarnd_rng(2, state);
            X[jj - (j + 1)] = zlarnd_rng(2, state);
            if (cabs(X[jj]) > cabs(X[jj - (j + 1)])) {
                X[jj - 1] = 2.0 * X[jj];
            } else {
                X[jj - 1] = 2.0 * X[jj - (j + 1)];
            }
            jj = jj - (j + 1) - j;
            j = j - 2;
        } else if (j == 0) {
            X[jj] = zlarnd_rng(2, state);
            j = j - 1;
        }

    /*     UPLO = 'L':  Lower triangular storage */

    } else {
        n5 = n / 5;
        n5 = n5 * 5;

        jj = 0;
        for (j = 0; j < n5; j += 5) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[jj] = a;
            X[jj + 2] = b;
            jj = jj + (n - j);
            X[jj] = zlarnd_rng(2, state);
            X[jj + 1] = r;
            jj = jj + (n - j - 1);
            X[jj] = c;
            jj = jj + (n - j - 2);
            X[jj] = zlarnd_rng(2, state);
            jj = jj + (n - j - 3);
            X[jj] = zlarnd_rng(2, state);
            if (cabs(X[jj - (n - j - 3)]) > cabs(X[jj])) {
                X[jj - (n - j - 3) + 1] = 2.0 * X[jj - (n - j - 3)];
            } else {
                X[jj - (n - j - 3) + 1] = 2.0 * X[jj];
            }
            jj = jj + (n - j - 4);
        }

        /*        Clean-up for N not a multiple of 5. */

        j = n5;
        if (j < n - 2) {
            a = alpha3 * zlarnd_rng(5, state);
            b = zlarnd_rng(5, state) / alpha;
            c = a - 2.0 * b * EYE;
            r = c / beta;
            X[jj] = a;
            X[jj + 2] = b;
            jj = jj + (n - j);
            X[jj] = zlarnd_rng(2, state);
            X[jj + 1] = r;
            jj = jj + (n - j - 1);
            X[jj] = c;
            jj = jj + (n - j - 2);
            j = j + 3;
        }
        if (j < n - 1) {
            X[jj] = zlarnd_rng(2, state);
            X[jj + (n - j)] = zlarnd_rng(2, state);
            if (cabs(X[jj]) > cabs(X[jj + (n - j)])) {
                X[jj + 1] = 2.0 * X[jj];
            } else {
                X[jj + 1] = 2.0 * X[jj + (n - j)];
            }
            jj = jj + (n - j) + (n - j - 1);
            j = j + 2;
        } else if (j == n - 1) {
            X[jj] = zlarnd_rng(2, state);
            jj = jj + (n - j);
            j = j + 1;
        }
    }
}
