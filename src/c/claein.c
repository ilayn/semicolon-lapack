/**
 * @file claein.c
 * @brief CLAEIN computes a specified right or left eigenvector of an upper
 *        Hessenberg matrix by inverse iteration.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAEIN uses inverse iteration to find a right or left eigenvector
 * corresponding to the eigenvalue W of a complex upper Hessenberg
 * matrix H.
 *
 * @param[in]     rightv  If nonzero, compute right eigenvector;
 *                        if zero, compute left eigenvector.
 * @param[in]     noinit  If nonzero, no initial vector supplied in V;
 *                        if zero, initial vector supplied in V.
 * @param[in]     n       The order of the matrix H.  N >= 0.
 * @param[in]     H       Complex array, dimension (ldh, n).
 *                        The upper Hessenberg matrix H.
 * @param[in]     ldh     The leading dimension of H.  ldh >= max(1,n).
 * @param[in]     w       The eigenvalue of H whose corresponding right or left
 *                        eigenvector is to be computed.
 * @param[in,out] V       Complex array, dimension (n).
 *                        On entry, if noinit = 0, V must contain a starting
 *                        vector for inverse iteration; otherwise V need not be set.
 *                        On exit, V contains the computed eigenvector, normalized so
 *                        that the component of largest magnitude has magnitude 1; here
 *                        the magnitude of a complex number (x,y) is taken to be
 *                        |x| + |y|.
 * @param[out]    B       Complex array, dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of B.  ldb >= max(1,n).
 * @param[out]    rwork   Single precision array, dimension (n).
 * @param[in]     eps3    A small machine-dependent value which is used to perturb
 *                        close eigenvalues, and to replace zero pivots.
 * @param[in]     smlnum  A machine-dependent value close to the underflow threshold.
 * @param[out]    info    = 0: successful exit
 *                        = 1: inverse iteration did not converge; V is set to the
 *                             last iterate.
 */
void claein(
    const int rightv,
    const int noinit,
    const int n,
    const c64* restrict H,
    const int ldh,
    const c64 w,
    c64* restrict V,
    c64* restrict B,
    const int ldb,
    f32* restrict rwork,
    const f32 eps3,
    const f32 smlnum,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 TENTH = 0.1f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    int i, ierr, its, j;
    f32 growto, nrmsml, rootn, rtemp, scale, vnorm;
    c64 ei, ej, temp, x;

    *info = 0;

    /*     GROWTO is the threshold used in the acceptance test for an
     *     eigenvector. */
    rootn = sqrtf((f32)n);
    growto = TENTH / rootn;
    nrmsml = (ONE > eps3 * rootn ? ONE : eps3 * rootn) * smlnum;

    /*     Form B = H - W*I (except that the subdiagonal elements are not
     *     stored). */
    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++) {
            B[i + j * ldb] = H[i + j * ldh];
        }
        B[j + j * ldb] = H[j + j * ldh] - w;
    }

    if (noinit) {
        /*        Initialize V. */
        for (i = 0; i < n; i++) {
            V[i] = CMPLXF(eps3, 0.0f);
        }
    } else {
        /*        Scale supplied initial vector. */
        vnorm = cblas_scnrm2(n, V, 1);
        f32 denom = vnorm > nrmsml ? vnorm : nrmsml;
        cblas_csscal(n, (eps3 * rootn) / denom, V, 1);
    }

    if (rightv) {
        /*        LU decomposition with partial pivoting of B, replacing zero
         *        pivots by EPS3. */
        for (i = 0; i < n - 1; i++) {
            ei = H[i + 1 + i * ldh];
            if (cabs1f(B[i + i * ldb]) < cabs1f(ei)) {
                /*              Interchange rows and eliminate. */
                x = cladiv(B[i + i * ldb], ei);
                B[i + i * ldb] = ei;
                for (j = i + 1; j < n; j++) {
                    temp = B[i + 1 + j * ldb];
                    B[i + 1 + j * ldb] = B[i + j * ldb] - x * temp;
                    B[i + j * ldb] = temp;
                }
            } else {
                /*              Eliminate without interchange. */
                if (B[i + i * ldb] == CZERO) {
                    B[i + i * ldb] = CMPLXF(eps3, 0.0f);
                }
                x = cladiv(ei, B[i + i * ldb]);
                if (x != CZERO) {
                    for (j = i + 1; j < n; j++) {
                        B[i + 1 + j * ldb] = B[i + 1 + j * ldb] - x * B[i + j * ldb];
                    }
                }
            }
        }
        if (B[n - 1 + (n - 1) * ldb] == CZERO) {
            B[n - 1 + (n - 1) * ldb] = CMPLXF(eps3, 0.0f);
        }
    } else {
        /*        UL decomposition with partial pivoting of B, replacing zero
         *        pivots by EPS3. */
        for (j = n - 1; j >= 1; j--) {
            ej = H[j + (j - 1) * ldh];
            if (cabs1f(B[j + j * ldb]) < cabs1f(ej)) {
                /*              Interchange columns and eliminate. */
                x = cladiv(B[j + j * ldb], ej);
                B[j + j * ldb] = ej;
                for (i = 0; i < j; i++) {
                    temp = B[i + (j - 1) * ldb];
                    B[i + (j - 1) * ldb] = B[i + j * ldb] - x * temp;
                    B[i + j * ldb] = temp;
                }
            } else {
                /*              Eliminate without interchange. */
                if (B[j + j * ldb] == CZERO) {
                    B[j + j * ldb] = CMPLXF(eps3, 0.0f);
                }
                x = cladiv(ej, B[j + j * ldb]);
                if (x != CZERO) {
                    for (i = 0; i < j; i++) {
                        B[i + (j - 1) * ldb] = B[i + (j - 1) * ldb] - x * B[i + j * ldb];
                    }
                }
            }
        }
        if (B[0] == CZERO) {
            B[0] = CMPLXF(eps3, 0.0f);
        }
    }

    char normin = 'N';
    char trans = rightv ? 'N' : 'C';
    for (its = 0; its < n; its++) {
        /*        Solve U*x = scale*v for a right eigenvector
         *          or U**H *x = scale*v for a left eigenvector,
         *        overwriting x on v. */
        clatrs("U", &trans, "N", &normin, n, B, ldb, V, &scale, rwork, &ierr);
        normin = 'Y';

        /*        Test for sufficient growth in the norm of v. */
        vnorm = cblas_scasum(n, V, 1);
        if (vnorm >= growto * scale) {
            goto L120;
        }

        /*        Choose new orthogonal starting vector and try again. */
        rtemp = eps3 / (rootn + ONE);
        V[0] = CMPLXF(eps3, 0.0f);
        for (i = 1; i < n; i++) {
            V[i] = CMPLXF(rtemp, 0.0f);
        }
        V[n - its - 1] = V[n - its - 1] - CMPLXF(eps3 * rootn, 0.0f);
    }

    /*     Failure to find eigenvector in N iterations. */
    *info = 1;

L120:
    /*     Normalize eigenvector. */
    i = cblas_icamax(n, V, 1);
    cblas_csscal(n, ONE / cabs1f(V[i]), V, 1);
}
