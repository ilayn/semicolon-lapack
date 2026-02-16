/**
 * @file dlasd8.c
 * @brief DLASD8 finds the square roots of the roots of the secular equation,
 *        and stores, for each element in D, the distance to its two nearest poles.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/** @cond */
static inline f64 dlamc3(f64 a, f64 b)
{
    volatile f64 result = a + b;
    return result;
}
/** @endcond */

/**
 * DLASD8 finds the square roots of the roots of the secular equation,
 * as defined by the values in DSIGMA and Z. It makes the appropriate
 * calls to DLASD4, and stores, for each element in D, the distance
 * to its two nearest poles (elements in DSIGMA). It also updates
 * the arrays VF and VL, the first and last components of all the
 * right singular vectors of the original bidiagonal matrix.
 *
 * DLASD8 is called from DLASD6.
 */
void dlasd8(const int icompq, const int k,
            f64* restrict D, f64* restrict Z,
            f64* restrict VF, f64* restrict VL,
            f64* restrict DIFL, f64* restrict DIFR,
            const int lddifr, const f64* restrict DSIGMA,
            f64* restrict work, int* info)
{
    const f64 ONE = 1.0;

    int i, j, iwk1, iwk2, iwk3;
    f64 diflj, difrj = 0.0, dj, dsigj, dsigjp = 0.0, rho, temp;

    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (k < 1) {
        *info = -2;
    } else if (lddifr < k) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DLASD8", -(*info));
        return;
    }

    /* Quick return if possible */
    if (k == 1) {
        /* D(1) = ABS(Z(1)) -> D[0] = fabs(Z[0]) */
        D[0] = fabs(Z[0]);
        DIFL[0] = D[0];
        if (icompq == 1) {
            /* DIFL(2) = ONE -> DIFL[1] = 1.0 */
            DIFL[1] = ONE;
            /* DIFR(1, 2) = ONE -> DIFR[0 + 1*lddifr] */
            DIFR[0 + 1 * lddifr] = ONE;
        }
        return;
    }

    /* Book keeping.
       Fortran: IWK1 = 1, IWK2 = IWK1 + K, IWK3 = IWK2 + K
       C: iwk1 = 0, iwk2 = k, iwk3 = 2*k */
    iwk1 = 0;
    iwk2 = iwk1 + k;
    iwk3 = iwk2 + k;

    /* Normalize Z. */
    rho = cblas_dnrm2(k, Z, 1);
    dlascl("G", 0, 0, rho, ONE, k, 1, Z, k, info);
    rho = rho * rho;

    /* Initialize WORK(IWK3) -> work[iwk3..iwk3+k-1] */
    dlaset("A", k, 1, ONE, ONE, &work[iwk3], k);

    /* Compute the updated singular values, the arrays DIFL, DIFR,
       and the updated Z.
       DO 40 J = 1, K -> for j = 0 to k-1 (0-based) */
    for (j = 0; j < k; j++) {
        /* CALL DLASD4(K, J, DSIGMA, Z, WORK(IWK1), RHO, D(J), WORK(IWK2), INFO)
           Note: DLASD4 expects J as 1-based index */
        dlasd4(k, j + 1, DSIGMA, Z, &work[iwk1], rho, &D[j], &work[iwk2], info);

        if (*info != 0) {
            return;
        }

        /* WORK(IWK3I+J) = WORK(IWK3I+J) * WORK(J) * WORK(IWK2I+J)
           Fortran: IWK3I+J for J=1..K gives IWK3I+1..IWK3I+K = IWK3..IWK3+K-1
           C: iwk3i + (j+1) for j=0..k-1 gives iwk3i+1..iwk3i+k = iwk3..iwk3+k-1
           So: work[iwk3 + j] = work[iwk3 + j] * work[j] * work[iwk2i + j + 1]
           where iwk2i + j + 1 = iwk2 - 1 + j + 1 = iwk2 + j */
        work[iwk3 + j] = work[iwk3 + j] * work[j] * work[iwk2 + j];

        /* DIFL(J) = -WORK(J) -> DIFL[j] = -work[j] */
        DIFL[j] = -work[j];

        /* DIFR(J, 1) = -WORK(J+1) -> DIFR[j + 0*lddifr] = -work[j+1] */
        DIFR[j + 0 * lddifr] = -work[j + 1];

        /* DO 20 I = 1, J - 1 -> for i = 0 to j-1 (0-based) */
        for (i = 0; i < j; i++) {
            work[iwk3 + i] = work[iwk3 + i] * work[i] *
                             work[iwk2 + i] / (DSIGMA[i] - DSIGMA[j]) /
                             (DSIGMA[i] + DSIGMA[j]);
        }

        /* DO 30 I = J + 1, K -> for i = j+1 to k-1 (0-based) */
        for (i = j + 1; i < k; i++) {
            work[iwk3 + i] = work[iwk3 + i] * work[i] *
                             work[iwk2 + i] / (DSIGMA[i] - DSIGMA[j]) /
                             (DSIGMA[i] + DSIGMA[j]);
        }
    }

    /* Compute updated Z.
       DO 50 I = 1, K -> for i = 0 to k-1 (0-based) */
    for (i = 0; i < k; i++) {
        Z[i] = copysign(sqrt(fabs(work[iwk3 + i])), Z[i]);
    }

    /* Update VF and VL.
       DO 80 J = 1, K -> for j = 0 to k-1 (0-based) */
    for (j = 0; j < k; j++) {
        diflj = DIFL[j];
        dj = D[j];
        dsigj = -DSIGMA[j];
        if (j < k - 1) {
            /* DIFRJ = -DIFR(J, 1) -> DIFR[j + 0*lddifr] */
            difrj = -DIFR[j + 0 * lddifr];
            dsigjp = -DSIGMA[j + 1];
        }

        /* WORK(J) = -Z(J) / DIFLJ / (DSIGMA(J)+DJ) */
        work[j] = -Z[j] / diflj / (DSIGMA[j] + dj);

        /* DO 60 I = 1, J - 1 -> for i = 0 to j-1 (0-based) */
        for (i = 0; i < j; i++) {
            work[i] = Z[i] / (dlamc3(DSIGMA[i], dsigj) - diflj) /
                      (DSIGMA[i] + dj);
        }

        /* DO 70 I = J + 1, K -> for i = j+1 to k-1 (0-based) */
        for (i = j + 1; i < k; i++) {
            work[i] = Z[i] / (dlamc3(DSIGMA[i], dsigjp) + difrj) /
                      (DSIGMA[i] + dj);
        }

        temp = cblas_dnrm2(k, work, 1);
        /* WORK(IWK2I+J) = DDOT(K, WORK, 1, VF, 1) / TEMP
           IWK2I + J for J=1..K gives IWK2I+1..IWK2I+K = IWK2..IWK2+K-1
           C: work[iwk2 + j] */
        work[iwk2 + j] = cblas_ddot(k, work, 1, VF, 1) / temp;
        work[iwk3 + j] = cblas_ddot(k, work, 1, VL, 1) / temp;
        if (icompq == 1) {
            /* DIFR(J, 2) = TEMP -> DIFR[j + 1*lddifr] */
            DIFR[j + 1 * lddifr] = temp;
        }
    }

    /* CALL DCOPY(K, WORK(IWK2), 1, VF, 1) */
    cblas_dcopy(k, &work[iwk2], 1, VF, 1);
    cblas_dcopy(k, &work[iwk3], 1, VL, 1);
}
