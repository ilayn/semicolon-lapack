/**
 * @file dlatm3.c
 * @brief DLATM3 returns the (ISUB,JSUB) entry of a random matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlatm3.f
 */

#include "verify.h"
#include "test_rng.h"

/**
 * DLATM3 returns the (ISUB,JSUB) entry of a random matrix of
 * dimension (M, N) described by the other parameters. (ISUB,JSUB)
 * is the final position of the (I,J) entry after pivoting
 * according to IPVTNG and IWORK. DLATM3 is called by the
 * DLATMR routine in order to build random test matrices. No error
 * checking on parameters is done, because this routine is called in
 * a tight loop by DLATMR which has already checked the parameters.
 *
 * Use of DLATM3 differs from DLATM2 in the order in which the random
 * number generator is called to fill in random matrix entries.
 * With DLATM2, the generator is called to fill in the pivoted matrix
 * columnwise. With DLATM3, the generator is called to fill in the
 * matrix columnwise, after which it is pivoted. Thus, DLATM3 can
 * be used to construct random matrices which differ only in their
 * order of rows and/or columns. DLATM2 is used to construct band
 * matrices while avoiding calling the random number generator for
 * entries outside the band (and therefore generating random numbers
 * in different orders for different pivot orders).
 *
 * @param[in] m       Number of rows of matrix.
 * @param[in] n       Number of columns of matrix.
 * @param[in] i       Row of unpivoted entry to be returned (1-based).
 * @param[in] j       Column of unpivoted entry to be returned (1-based).
 * @param[out] isub   Row of pivoted entry (1-based).
 * @param[out] jsub   Column of pivoted entry (1-based).
 * @param[in] kl      Lower bandwidth.
 * @param[in] ku      Upper bandwidth.
 * @param[in] idist   Distribution type: 1=U(0,1), 2=U(-1,1), 3=N(0,1).
 * @param[in] d       Diagonal entries of matrix.
 * @param[in] igrade  Grading type (0-5).
 * @param[in] dl      Left scale factors for grading.
 * @param[in] dr      Right scale factors for grading.
 * @param[in] ipvtng  Pivoting type: 0=none, 1=row, 2=column, 3=both.
 * @param[in] iwork   Permutation array.
 * @param[in] sparse  Sparsity parameter (0 to 1).
 *
 * @return The (isub,jsub) entry of the matrix.
 */
double dlatm3(
    const int m,
    const int n,
    const int i,
    const int j,
    int* isub,
    int* jsub,
    const int kl,
    const int ku,
    const int idist,
    const double* d,
    const int igrade,
    const double* dl,
    const double* dr,
    const int ipvtng,
    const int* iwork,
    const double sparse,
    uint64_t state[static 4])
{
    const double ZERO = 0.0;

    double temp;

    if (i < 1 || i > m || j < 1 || j > n) {
        *isub = i;
        *jsub = j;
        return ZERO;
    }

    if (ipvtng == 0) {
        *isub = i;
        *jsub = j;
    } else if (ipvtng == 1) {
        *isub = iwork[i - 1];
        *jsub = j;
    } else if (ipvtng == 2) {
        *isub = i;
        *jsub = iwork[j - 1];
    } else if (ipvtng == 3) {
        *isub = iwork[i - 1];
        *jsub = iwork[j - 1];
    } else {
        *isub = i;
        *jsub = j;
    }

    if (*jsub > *isub + ku || *jsub < *isub - kl) {
        return ZERO;
    }

    if (sparse > ZERO) {
        if (rng_uniform(state) < sparse) {
            return ZERO;
        }
    }

    if (i == j) {
        temp = d[i - 1];
    } else {
        temp = rng_dist(state, idist);
    }

    if (igrade == 1) {
        temp = temp * dl[i - 1];
    } else if (igrade == 2) {
        temp = temp * dr[j - 1];
    } else if (igrade == 3) {
        temp = temp * dl[i - 1] * dr[j - 1];
    } else if (igrade == 4 && i != j) {
        temp = temp * dl[i - 1] / dl[j - 1];
    } else if (igrade == 5) {
        temp = temp * dl[i - 1] * dl[j - 1];
    }

    return temp;
}
