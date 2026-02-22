/**
 * @file cpteqr.c
 * @brief CPTEQR computes all eigenvalues and, optionally, eigenvectors of a
 *        symmetric positive definite tridiagonal matrix.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPTEQR computes all eigenvalues and, optionally, eigenvectors of a
 * symmetric positive definite tridiagonal matrix by first factoring the
 * matrix using SPTTRF, and then calling CBDSQR to compute the singular
 * values of the bidiagonal factor.
 *
 * This routine computes the eigenvalues of the positive definite
 * tridiagonal matrix to high relative accuracy. This means that if the
 * eigenvalues range over many orders of magnitude in size, then the
 * small eigenvalues and corresponding eigenvectors will be computed
 * more accurately than, for example, with the standard QR method.
 *
 * The eigenvectors of a full or band positive definite Hermitian matrix
 * can also be found if CHETRD, CHPTRD, or CHBTRD has been used to
 * reduce this matrix to tridiagonal form. (The reduction to tridiagonal
 * form, however, may preclude the possibility of obtaining high
 * relative accuracy in the small eigenvalues of the original matrix, if
 * these eigenvalues range over many orders of magnitude.)
 *
 * @param[in]     compz Specifies whether the eigenvectors are to be computed.
 *                      = 'N': Compute eigenvalues only.
 *                      = 'V': Compute eigenvectors of original Hermitian
 *                             matrix also. Array Z contains the unitary
 *                             matrix used to reduce the original matrix to
 *                             tridiagonal form.
 *                      = 'I': Compute eigenvectors of tridiagonal matrix also.
 * @param[in]     n     The order of the matrix. n >= 0.
 * @param[in,out] D     Single precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal
 *                      matrix.
 *                      On normal exit, D contains the eigenvalues, in descending
 *                      order.
 * @param[in,out] E     Single precision array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the tridiagonal
 *                      matrix.
 *                      On exit, E has been destroyed.
 * @param[in,out] Z     Complex*16 array, dimension (ldz, n).
 *                      On entry, if compz = 'V', the unitary matrix used in the
 *                      reduction to tridiagonal form.
 *                      On exit, if compz = 'V', the orthonormal eigenvectors of the
 *                      original Hermitian matrix;
 *                      if compz = 'I', the orthonormal eigenvectors of the
 *                      tridiagonal matrix.
 *                      If info > 0 on exit, Z contains the eigenvectors associated
 *                      with only the stored eigenvalues.
 *                      If compz = 'N', then Z is not referenced.
 * @param[in]     ldz   The leading dimension of the array Z. ldz >= 1, and if
 *                      compz = 'V' or 'I', ldz >= max(1, n).
 * @param[out]    work  Single precision array, dimension (4*n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit.
 *                           - < 0: if info = -i, the i-th argument had an illegal value.
 *                           - > 0: if info = i, and i is:
 *                         - <= n  the Cholesky factorization of the matrix could
 *                           not be performed because the leading principal
 *                           minor of order i was not positive.
 *                         - > n   the SVD algorithm failed to converge;
 *                           if info = n+i, i off-diagonal elements of the
 *                           bidiagonal factor did not converge to zero.
 */
void cpteqr(
    const char* compz,
    const INT n,
    f32* restrict D,
    f32* restrict E,
    c64* restrict Z,
    const INT ldz,
    f32* restrict work,
    INT* info)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    INT i, icompz, nru;

    // Dummy arrays for cbdsqr (ncvt=0 and ncc=0)
    c64 VT_dummy[1];
    c64 C_dummy[1];

    // Test the input parameters
    *info = 0;

    if (compz[0] == 'N' || compz[0] == 'n') {
        icompz = 0;
    } else if (compz[0] == 'V' || compz[0] == 'v') {
        icompz = 1;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        icompz = 2;
    } else {
        icompz = -1;
    }

    if (icompz < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldz < 1 || (icompz > 0 && ldz < (n > 1 ? n : 1))) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CPTEQR", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0)
        return;

    if (n == 1) {
        if (icompz > 0) {
            Z[0] = CONE;
        }
        return;
    }

    // Initialize Z to the identity matrix if requested
    if (icompz == 2) {
        claset("F", n, n, CZERO, CONE, Z, ldz);
    }

    // Call SPTTRF to factor the matrix.
    spttrf(n, D, E, info);
    if (*info != 0)
        return;

    // D[i] = sqrt(D[i])
    for (i = 0; i < n; i++) {
        D[i] = sqrtf(D[i]);
    }

    // E[i] = E[i] * D[i] (form the bidiagonal matrix)
    for (i = 0; i < n - 1; i++) {
        E[i] = E[i] * D[i];
    }

    // Call CBDSQR to compute the singular values/vectors of the
    // bidiagonal factor.
    if (icompz > 0) {
        nru = n;
    } else {
        nru = 0;
    }

    cbdsqr("L", n, 0, nru, 0, D, E, VT_dummy, 1, Z, ldz, C_dummy, 1, work, info);

    // Square the singular values to get eigenvalues
    if (*info == 0) {
        for (i = 0; i < n; i++) {
            D[i] = D[i] * D[i];
        }
    } else {
        *info = n + *info;
    }
}
