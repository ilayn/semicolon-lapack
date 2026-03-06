/**
 * @file clatm2.c
 * @brief CLATM2 returns the (I,J) entry of a random matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/clatm2.f
 */

#include <complex.h>
#include "verify.h"
#include "test_rng.h"

/**
 * CLATM2 returns the (I,J) entry of a random matrix of dimension
 * (M, N) described by the other parameters. It is called by the
 * CLATMR routine in order to build random test matrices. No error
 * checking on parameters is done, because this routine is called in
 * a tight loop by CLATMR which has already checked the parameters.
 *
 * Use of CLATM2 differs from CLATM3 in the order in which the random
 * number generator is called to fill in random matrix entries.
 * With CLATM2, the generator is called to fill in the pivoted matrix
 * columnwise. With CLATM3, the generator is called to fill in the
 * matrix columnwise, after which it is pivoted. Thus, CLATM3 can
 * be used to construct random matrices which differ only in their
 * order of rows and/or columns. CLATM2 is used to construct band
 * matrices while avoiding calling the random number generator for
 * entries outside the band (and therefore generating random numbers
 * in different orders for different pivot orders).
 *
 * @param[in] m       Number of rows of matrix.
 * @param[in] n       Number of columns of matrix.
 * @param[in] i       Row of entry to be returned (0-based).
 * @param[in] j       Column of entry to be returned (0-based).
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
 * @return The (i,j) entry of the matrix.
 */
c64 clatm2(
    const INT m,
    const INT n,
    const INT i,
    const INT j,
    const INT kl,
    const INT ku,
    const INT idist,
    const c64* d,
    const INT igrade,
    const c64* dl,
    const c64* dr,
    const INT ipvtng,
    const INT* iwork,
    const f32 sparse,
    uint64_t state[static 4])
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const f32 ZERO = 0.0f;

    INT isub, jsub;
    c64 ctemp;

    if (i < 0 || i >= m || j < 0 || j >= n) {
        return CZERO;
    }

    if (j > i + ku || j < i - kl) {
        return CZERO;
    }

    if (sparse > ZERO) {
        if (rng_uniform_f32(state) < sparse) {
            return CZERO;
        }
    }

    if (ipvtng == 0) {
        isub = i;
        jsub = j;
    } else if (ipvtng == 1) {
        isub = iwork[i];
        jsub = j;
    } else if (ipvtng == 2) {
        isub = i;
        jsub = iwork[j];
    } else if (ipvtng == 3) {
        isub = iwork[i];
        jsub = iwork[j];
    } else {
        isub = i;
        jsub = j;
    }

    if (isub == jsub) {
        ctemp = d[isub];
    } else {
        ctemp = clarnd_rng(idist, state);
    }

    if (igrade == 1) {
        ctemp = ctemp * dl[isub];
    } else if (igrade == 2) {
        ctemp = ctemp * dr[jsub];
    } else if (igrade == 3) {
        ctemp = ctemp * dl[isub] * dr[jsub];
    } else if (igrade == 4 && isub != jsub) {
        ctemp = ctemp * dl[isub] / dl[jsub];
    } else if (igrade == 5) {
        ctemp = ctemp * dl[isub] * conjf(dl[jsub]);
    } else if (igrade == 6) {
        ctemp = ctemp * dl[isub] * dl[jsub];
    }

    return ctemp;
}
