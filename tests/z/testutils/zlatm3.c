/**
 * @file zlatm3.c
 * @brief ZLATM3 returns the (ISUB,JSUB) entry of a random matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatm3.f
 */

#include <complex.h>
#include "verify.h"
#include "test_rng.h"

/**
 * ZLATM3 returns the (ISUB,JSUB) entry of a random matrix of
 * dimension (M, N) described by the other parameters. (ISUB,JSUB)
 * is the final position of the (I,J) entry after pivoting
 * according to IPVTNG and IWORK. ZLATM3 is called by the
 * ZLATMR routine in order to build random test matrices. No error
 * checking on parameters is done, because this routine is called in
 * a tight loop by ZLATMR which has already checked the parameters.
 *
 * Use of ZLATM3 differs from ZLATM2 in the order in which the random
 * number generator is called to fill in random matrix entries.
 * With ZLATM2, the generator is called to fill in the pivoted matrix
 * columnwise. With ZLATM3, the generator is called to fill in the
 * matrix columnwise, after which it is pivoted. Thus, ZLATM3 can
 * be used to construct random matrices which differ only in their
 * order of rows and/or columns. ZLATM2 is used to construct band
 * matrices while avoiding calling the random number generator for
 * entries outside the band (and therefore generating random numbers
 * in different orders for different pivot orders).
 *
 * @param[in] m       Number of rows of matrix.
 * @param[in] n       Number of columns of matrix.
 * @param[in] i       Row of unpivoted entry to be returned (0-based).
 * @param[in] j       Column of unpivoted entry to be returned (0-based).
 * @param[out] isub   Row of pivoted entry (0-based).
 * @param[out] jsub   Column of pivoted entry (0-based).
 * @param[in] kl      Lower bandwidth.
 * @param[in] ku      Upper bandwidth.
 * @param[in] idist   Distribution type: 1=U(0,1), 2=U(-1,1), 3=N(0,1), 4=disk.
 * @param[in] d       Complex diagonal entries of matrix.
 * @param[in] igrade  Grading type (0-6).
 * @param[in] dl      Complex left scale factors for grading.
 * @param[in] dr      Complex right scale factors for grading.
 * @param[in] ipvtng  Pivoting type: 0=none, 1=row, 2=column, 3=both.
 * @param[in] iwork   Permutation array.
 * @param[in] sparse  Sparsity parameter (0 to 1).
 *
 * @return The (isub,jsub) entry of the matrix.
 */
c128 zlatm3(
    const INT m,
    const INT n,
    const INT i,
    const INT j,
    INT* isub,
    INT* jsub,
    const INT kl,
    const INT ku,
    const INT idist,
    const c128* d,
    const INT igrade,
    const c128* dl,
    const c128* dr,
    const INT ipvtng,
    const INT* iwork,
    const f64 sparse,
    uint64_t state[static 4])
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const f64 ZERO = 0.0;

    c128 ctemp;

    if (i < 0 || i >= m || j < 0 || j >= n) {
        *isub = i;
        *jsub = j;
        return CZERO;
    }

    if (ipvtng == 0) {
        *isub = i;
        *jsub = j;
    } else if (ipvtng == 1) {
        *isub = iwork[i];
        *jsub = j;
    } else if (ipvtng == 2) {
        *isub = i;
        *jsub = iwork[j];
    } else if (ipvtng == 3) {
        *isub = iwork[i];
        *jsub = iwork[j];
    } else {
        *isub = i;
        *jsub = j;
    }

    if (*jsub > *isub + ku || *jsub < *isub - kl) {
        return CZERO;
    }

    if (sparse > ZERO) {
        if (rng_uniform(state) < sparse) {
            return CZERO;
        }
    }

    if (i == j) {
        ctemp = d[i];
    } else {
        ctemp = zlarnd_rng(idist, state);
    }

    if (igrade == 1) {
        ctemp = ctemp * dl[i];
    } else if (igrade == 2) {
        ctemp = ctemp * dr[j];
    } else if (igrade == 3) {
        ctemp = ctemp * dl[i] * dr[j];
    } else if (igrade == 4 && i != j) {
        ctemp = ctemp * dl[i] / dl[j];
    } else if (igrade == 5) {
        ctemp = ctemp * dl[i] * conj(dl[j]);
    } else if (igrade == 6) {
        ctemp = ctemp * dl[i] * dl[j];
    }

    return ctemp;
}
