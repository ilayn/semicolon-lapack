/**
 * @file zhegvx.c
 * @brief ZHEGVX computes selected eigenvalues and optionally eigenvectors of a
 *        complex generalized Hermitian-definite eigenproblem.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHEGVX computes selected eigenvalues, and optionally, eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x. Here A and
 * B are assumed to be Hermitian and B is also positive definite.
 * Eigenvalues and eigenvectors can be selected by specifying either a
 * range of values or a range of indices for the desired eigenvalues.
 *
 * @param[in]     itype   = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     range   = 'A': all; = 'V': in (vl,vu]; = 'I': il-th through iu-th
 * @param[in]     uplo    = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n       The order of matrices A and B. n >= 0.
 * @param[in,out] A       On entry, Hermitian matrix A. On exit, destroyed.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       On entry, Hermitian positive definite B. On exit, Cholesky factor.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1, n).
 * @param[in]     vl      Lower bound of interval (if range='V').
 * @param[in]     vu      Upper bound of interval (if range='V').
 * @param[in]     il      Index of smallest eigenvalue (if range='I', 1-based).
 * @param[in]     iu      Index of largest eigenvalue (if range='I', 1-based).
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Total number of eigenvalues found.
 * @param[out]    W       Selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, and if jobz='V', ldz >= n.
 * @param[out]    work    Complex workspace array, dimension (max(1, lwork)).
 * @param[in]     lwork   Length of work. lwork >= max(1, 2*n). If -1, workspace query.
 * @param[out]    rwork   Real workspace array, dimension (7*n).
 * @param[out]    iwork   Integer workspace array, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: ZPOTRF/ZHEEVX error.
 */
void zhegvx(
    const int itype,
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    c128* restrict B,
    const int ldb,
    const f64 vl,
    const f64 vu,
    const int il,
    const int iu,
    const f64 abstol,
    int* m,
    f64* restrict W,
    c128* restrict Z,
    const int ldz,
    c128* restrict work,
    const int lwork,
    f64* restrict rwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    int wantz, upper, alleig, valeig, indeig, lquery;
    int lwkopt, nb;
    char trans;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lquery = (lwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!(alleig || valeig || indeig)) {
        *info = -3;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 1 || il > (n > 1 ? n : 1)) {
                *info = -12;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -13;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -18;
        }
    }

    if (*info == 0) {
        nb = lapack_get_nb("HETRD");
        lwkopt = (nb + 1) * n;
        if (lwkopt < 1) {
            lwkopt = 1;
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);

        if (lwork < (2 * n > 1 ? 2 * n : 1) && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        xerbla("ZHEGVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0) {
        return;
    }

    /* Form a Cholesky factorization of B */
    zpotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem and solve */
    zhegst(itype, uplo, n, A, lda, B, ldb, info);
    zheevx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol,
           m, W, Z, ldz, work, lwork, rwork, iwork, ifail, info);

    if (wantz) {
        /* Backtransform eigenvectors to the original problem */
        if (*info > 0) {
            *m = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            /* x = inv(L)**H*y or inv(U)*y */
            if (upper) {
                trans = 'N';
            } else {
                trans = 'C';
            }

            cblas_ztrsm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'N' ? CblasNoTrans : CblasConjTrans,
                        CblasNonUnit, n, *m, &CONE, B, ldb, Z, ldz);

        } else if (itype == 3) {
            /* x = L*y or U**H*y */
            if (upper) {
                trans = 'C';
            } else {
                trans = 'N';
            }

            cblas_ztrmm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'C' ? CblasConjTrans : CblasNoTrans,
                        CblasNonUnit, n, *m, &CONE, B, ldb, Z, ldz);
        }
    }

    /* Set WORK(1) to optimal complex workspace size */
    work[0] = CMPLX((f64)lwkopt, 0.0);
}
