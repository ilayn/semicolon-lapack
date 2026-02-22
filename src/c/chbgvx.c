/**
 * @file chbgvx.c
 * @brief CHBGVX computes selected eigenvalues of a complex generalized
 *        Hermitian-definite banded eigenproblem.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHBGVX computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite banded eigenproblem, of
 * the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
 * and banded, and B is also positive definite. Eigenvalues and
 * eigenvectors can be selected by specifying either all eigenvalues,
 * a range of values or a range of indices for the desired eigenvalues.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     range  = 'A': all eigenvalues will be found
 *                        = 'V': all eigenvalues in (vl,vu] will be found
 *                        = 'I': the il-th through iu-th eigenvalues will be found
 * @param[in]     uplo   = 'U': Upper triangles of A and B are stored
 *                        = 'L': Lower triangles of A and B are stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in]     ka     The number of super-/sub-diagonals of A. ka >= 0.
 * @param[in]     kb     The number of super-/sub-diagonals of B. kb >= 0.
 * @param[in,out] AB     Complex array, dimension (ldab, n).
 *                        On entry, the Hermitian band matrix A.
 *                        On exit, contents are destroyed.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in,out] BB     Complex array, dimension (ldbb, n).
 *                        On entry, the Hermitian band matrix B.
 *                        On exit, the split Cholesky factor S from cpbstf.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    Q      Complex array, dimension (ldq, n). If jobz='V', the
 *                        transformation matrix.
 * @param[in]     ldq    The leading dimension of Q.
 *                        ldq >= 1, and if jobz='V', ldq >= max(1,n).
 * @param[in]     vl     If range='V', the lower bound of the interval.
 * @param[in]     vu     If range='V', the upper bound of the interval. vl < vu.
 * @param[in]     il     If range='I', the index of the smallest eigenvalue.
 * @param[in]     iu     If range='I', the index of the largest eigenvalue.
 * @param[in]     abstol The absolute error tolerance for eigenvalues.
 * @param[out]    m      The total number of eigenvalues found.
 * @param[out]    W      The eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, n). If jobz='V',
 *                        the eigenvectors.
 * @param[in]     ldz    The leading dimension of Z.
 *                        ldz >= 1, and if jobz='V', ldz >= max(1,n).
 * @param[out]    work   Complex workspace array of dimension (n).
 * @param[out]    rwork  Single precision workspace array of dimension (7*n).
 * @param[out]    iwork  Integer workspace array of dimension (5*n).
 * @param[out]    ifail  If jobz='V', indices of eigenvectors that failed
 *                        to converge. Array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - <= n: i eigenvectors failed to converge
 *                         - > n: cpbstf returned info = i (B not positive definite)
 */
void chbgvx(
    const char* jobz,
    const char* range,
    const char* uplo,
    const INT n,
    const INT ka,
    const INT kb,
    c64* restrict AB,
    const INT ldab,
    c64* restrict BB,
    const INT ldbb,
    c64* restrict Q,
    const INT ldq,
    const f32 vl,
    const f32 vu,
    const INT il,
    const INT iu,
    const f32 abstol,
    INT* m,
    f32* restrict W,
    c64* restrict Z,
    const INT ldz,
    c64* restrict work,
    f32* restrict rwork,
    INT* restrict iwork,
    INT* restrict ifail,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT alleig, indeig, test, upper, valeig, wantz;
    INT i, iinfo, indd, inde, indee, indisp, indiwk, indrwk, indwrk, itmp1, j, jj, nsplit;
    f32 tmp1;
    char order, vect;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ka < 0) {
        *info = -5;
    } else if (kb < 0 || kb > ka) {
        *info = -6;
    } else if (ldab < ka + 1) {
        *info = -8;
    } else if (ldbb < kb + 1) {
        *info = -10;
    } else if (ldq < 1 || (wantz && ldq < n)) {
        *info = -12;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -14;
        } else if (indeig) {
            if (il < 0 || il > ((0 > n - 1) ? 0 : n - 1)) {
                *info = -15;
            } else if (iu < ((n - 1 < il) ? n - 1 : il) || iu > n - 1) {
                *info = -16;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -21;
        }
    }

    if (*info != 0) {
        xerbla("CHBGVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    cpbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    chbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Q, ldq,
           work, rwork, &iinfo);

    indd = 0;
    inde = indd + n;
    indrwk = inde + n;
    indwrk = 0;
    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    chbtrd(&vect, uplo, n, ka, AB, ldab, &rwork[indd],
           &rwork[inde], Q, ldq, &work[indwrk], &iinfo);

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_scopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        cblas_scopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
        if (!wantz) {
            ssterf(n, W, &rwork[indee], info);
        } else {
            clacpy("A", n, n, Q, ldq, Z, ldz);
            csteqr(jobz, n, W, &rwork[indee], Z, ldz, &rwork[indrwk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indisp = n;
    indiwk = indisp + n;
    sstebz(range, &order, n, vl, vu, il, iu, abstol,
           &rwork[indd], &rwork[inde], m, &nsplit, W,
           &iwork[0], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        cstein(n, &rwork[indd], &rwork[inde], *m, W,
               &iwork[0], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwk], ifail, info);

        for (j = 0; j < *m; j++) {
            cblas_ccopy(n, &Z[j * ldz], 1, work, 1);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, Q, ldq,
                        work, 1, &CZERO, &Z[j * ldz], 1);
        }
    }

L30:
    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = 0;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i != 0) {
                itmp1 = iwork[i];
                W[i] = W[j];
                iwork[i] = iwork[j];
                W[j] = tmp1;
                iwork[j] = itmp1;
                cblas_cswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
