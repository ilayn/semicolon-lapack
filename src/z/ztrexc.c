/**
 * @file ztrexc.c
 * @brief ZTREXC reorders the Schur factorization of a complex matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZTREXC reorders the Schur factorization of a complex matrix
 * A = Q*T*Q**H, so that the diagonal element of T with row index IFST
 * is moved to row ILST.
 *
 * The Schur form T is reordered by a unitary similarity transformation
 * Z**H*T*Z, and optionally the matrix Q of Schur vectors is updated by
 * postmultiplying it with Z.
 *
 * @param[in]     compq  'V': update the matrix Q of Schur vectors.
 *                       'N': do not update Q.
 * @param[in]     n      The order of the matrix T. n >= 0.
 *                       If n == 0, arguments ilst and ifst may be any value.
 * @param[in,out] T      Complex array, dimension (ldt, n).
 *                       On entry, the upper triangular matrix T.
 *                       On exit, the reordered upper triangular matrix.
 * @param[in]     ldt    The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] Q      Complex array, dimension (ldq, n).
 *                       On entry, if compq = 'V', the matrix Q of Schur vectors.
 *                       On exit, if compq = 'V', Q has been postmultiplied by
 *                       the unitary transformation matrix Z which reorders T.
 *                       If compq = 'N', Q is not referenced.
 * @param[in]     ldq    The leading dimension of Q. ldq >= 1, and if
 *                       compq = 'V', ldq >= max(1, n).
 * @param[in]     ifst   Specify the reordering of the diagonal elements of T:
 *                       The element with row index IFST is moved to row ILST
 *                       by a sequence of transpositions between adjacent elements.
 *                       0 <= ifst < n (0-based).
 * @param[in]     ilst   0 <= ilst < n (0-based).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztrexc(const char* compq, const int n, double complex* T, const int ldt,
            double complex* Q, const int ldq, const int ifst, const int ilst,
            int* info)
{
    int wantq;
    int k, m1, m2, m3;
    double cs;
    double complex sn, t11, t22, temp;

    /* Decode and test the input parameters. */
    *info = 0;
    wantq = (compq[0] == 'V' || compq[0] == 'v');
    if (!wantq && compq[0] != 'N' && compq[0] != 'n') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldq < 1 || (wantq && ldq < (n > 1 ? n : 1))) {
        *info = -6;
    } else if ((ifst < 0 || ifst >= n) && (n > 0)) {
        *info = -7;
    } else if ((ilst < 0 || ilst >= n) && (n > 0)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("ZTREXC", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 1 || ifst == ilst)
        return;

    if (ifst < ilst) {

        /* Move the IFST-th diagonal element forward down the diagonal. */
        m1 = 0;
        m2 = -1;
        m3 = 1;
    } else {

        /* Move the IFST-th diagonal element backward up the diagonal. */
        m1 = -1;
        m2 = 0;
        m3 = -1;
    }

    for (k = ifst + m1; m3 > 0 ? k <= ilst + m2 : k >= ilst + m2; k += m3) {

        /* Interchange the k-th and (k+1)-th diagonal elements. */
        t11 = T[k + k * ldt];
        t22 = T[(k + 1) + (k + 1) * ldt];

        /* Determine the transformation to perform the interchange. */
        zlartg(T[k + (k + 1) * ldt], t22 - t11, &cs, &sn, &temp);

        /* Apply transformation to the matrix T. */
        if (k + 2 < n)
            zrot(n - k - 2, &T[k + (k + 2) * ldt], ldt,
                 &T[(k + 1) + (k + 2) * ldt], ldt, cs, sn);
        zrot(k, &T[k * ldt], 1, &T[(k + 1) * ldt], 1, cs, conj(sn));

        T[k + k * ldt] = t22;
        T[(k + 1) + (k + 1) * ldt] = t11;

        if (wantq) {

            /* Accumulate transformation in the matrix Q. */
            zrot(n, &Q[k * ldq], 1, &Q[(k + 1) * ldq], 1, cs, conj(sn));
        }
    }
}
