/**
 * @file sstein.c
 * @brief SSTEIN computes eigenvectors of a real symmetric tridiagonal matrix
 *        T corresponding to specified eigenvalues, using inverse iteration.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSTEIN computes the eigenvectors of a real symmetric tridiagonal
 * matrix T corresponding to specified eigenvalues, using inverse
 * iteration.
 *
 * The maximum number of iterations allowed for each eigenvector is
 * specified by an internal parameter MAXITS (currently set to 5).
 *
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[in]     D       Double precision array, dimension (n).
 *                        The n diagonal elements of the tridiagonal matrix T.
 * @param[in]     E       Double precision array, dimension (n-1).
 *                        The (n-1) subdiagonal elements of the tridiagonal
 *                        matrix T.
 * @param[in]     m       The number of eigenvectors to be found. 0 <= m <= n.
 * @param[in]     W       Double precision array, dimension (m).
 *                        The eigenvalues for which eigenvectors are to be
 *                        computed. The eigenvalues should be grouped by
 *                        split-off block and ordered from smallest to largest
 *                        within the block. (The output array W from SSTEBZ
 *                        with ORDER = 'B' is expected here.)
 * @param[in]     iblock  Integer array, dimension (m).
 *                        The submatrix indices associated with the
 *                        corresponding eigenvalues in W; iblock[i]=0 if
 *                        eigenvalue W[i] belongs to the first submatrix from
 *                        the top, =1 if W[i] belongs to the second submatrix,
 *                        etc. (0-based block numbers.) (The output array
 *                        iblock from SSTEBZ is expected here.)
 *                        A negative entry -(b+2), with which SSTEBZ flags an
 *                        eigenvalue of block b that did not converge, is
 *                        treated as block b.
 * @param[in]     isplit  Integer array, dimension (nsplit).
 *                        The splitting points, at which T breaks up into
 *                        submatrices. The first submatrix consists of
 *                        rows/columns 0 to isplit[0], the second of
 *                        rows/columns isplit[0]+1 through isplit[1], etc.
 *                        (0-based endpoint indices.)
 * @param[out]    Z       Double precision array, dimension (ldz, m).
 *                        The computed eigenvectors. The eigenvector associated
 *                        with the eigenvalue W[i] is stored in the i-th column
 *                        of Z. Any vector which fails to converge is set to its
 *                        current iterate after MAXITS iterations.
 * @param[in]     ldz     The leading dimension of the array Z. ldz >= max(1,n).
 * @param[out]    work    Double precision array, dimension (5*n).
 * @param[out]    iwork   Integer array, dimension (n).
 * @param[out]    ifail   Integer array, dimension (m).
 *                        On normal exit, all elements of ifail are -1.
 *                        If one or more eigenvectors fail to converge after
 *                        MAXITS iterations, then their indices are stored in
 *                        array ifail (0-based indices).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value.
 *                         - > 0: if info = i, then i eigenvectors failed to
 *                           converge in MAXITS iterations. Their indices are
 *                           stored in array ifail.
 */
void sstein(
    const INT n,
    const f32* restrict D,
    const f32* restrict E,
    const INT m,
    const f32* restrict W,
    const INT* restrict iblock,
    const INT* restrict isplit,
    f32* restrict Z,
    const INT ldz,
    f32* restrict work,
    INT* restrict iwork,
    INT* restrict ifail,
    INT* info)
{
    const INT MAXITS = 5;
    const INT EXTRA = 2;
    const f32 ODM3 = 1.0e-3f;
    const f32 ODM1 = 1.0e-1f;

    INT b1, blksiz, bn, gpind = 0, i, iinfo, its, j, j1, jblk, jmax, nblk, nrmchk;
    f32 dtpcrt, eps, eps1, nrm, onenrm, ortol, pertol, scl, sep, tol, xj, xjm = 0.0f, ztr;
    INT iseed[4];

    /* Initialize pointers. */
    const INT indrv1 = 0;
    const INT indrv2 = n;
    const INT indrv3 = 2 * n;
    const INT indrv4 = 3 * n;
    const INT indrv5 = 4 * n;

    /* Test the input parameters. */
    *info = 0;
    for (i = 0; i < m; i++) {
        ifail[i] = -1;
    }

    if (n < 0) {
        *info = -1;
    } else if (m < 0 || m > n) {
        *info = -4;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -9;
    } else {
        for (j = 1; j < m; j++) {
            INT ibj = (iblock[j] >= 0) ? iblock[j] : -iblock[j] - 2;
            INT ibjm1 = (iblock[j - 1] >= 0) ? iblock[j - 1] : -iblock[j - 1] - 2;
            if (ibj < ibjm1) {
                *info = -6;
                break;
            }
            if (ibj == ibjm1 && W[j] < W[j - 1]) {
                *info = -5;
                break;
            }
        }
    }

    if (*info != 0) {
        xerbla("SSTEIN", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0) {
        return;
    } else if (n == 1) {
        Z[0] = 1.0f;
        return;
    }

    /* Get machine constants. */
    eps = slamch("P");

    /* Initialize seed for random number generator SLARNV. */
    iseed[0] = 1;
    iseed[1] = 1;
    iseed[2] = 1;
    iseed[3] = 1;

    /* Compute eigenvectors of matrix blocks. */
    j1 = 0;
    for (nblk = 0; nblk <= ((iblock[m - 1] >= 0) ? iblock[m - 1] : -iblock[m - 1] - 2); nblk++) {

        /* Find starting and ending indices of block nblk. */
        if (nblk == 0) {
            b1 = 0;
        } else {
            b1 = isplit[nblk - 1] + 1;
        }
        bn = isplit[nblk];
        blksiz = bn - b1 + 1;

        if (blksiz != 1) {
            gpind = j1;

            /*
             * Compute reorthogonalization criterion and stopping criterion.
             */
            onenrm = fabsf(D[b1]) + fabsf(E[b1]);
            onenrm = (onenrm > fabsf(D[bn]) + fabsf(E[bn - 1]))
                ? onenrm : fabsf(D[bn]) + fabsf(E[bn - 1]);
            for (i = b1 + 1; i <= bn - 1; i++) {
                onenrm = (onenrm > fabsf(D[i]) + fabsf(E[i - 1]) + fabsf(E[i]))
                    ? onenrm : fabsf(D[i]) + fabsf(E[i - 1]) + fabsf(E[i]);
            }
            ortol = ODM3 * onenrm;

            dtpcrt = sqrtf(ODM1 / blksiz);
        }

        /* Loop through eigenvalues of block nblk. */
        jblk = 0;
        for (j = j1; j < m; j++) {
            if (((iblock[j] >= 0) ? iblock[j] : -iblock[j] - 2) != nblk) {
                j1 = j;
                break;
            }
            jblk++;
            xj = W[j];

            /* Skip all the work if the block size is one. */
            if (blksiz == 1) {
                work[indrv1] = 1.0f;
                goto store_eigenvector;
            }

            /*
             * If eigenvalues j and j-1 are too close, add a relatively
             * small perturbation.
             */
            if (jblk > 1) {
                eps1 = fabsf(eps * xj);
                pertol = 10.0f * eps1;
                sep = xj - xjm;
                if (sep < pertol) {
                    xj = xjm + pertol;
                }
            }

            its = 0;
            nrmchk = 0;

            /* Get random starting vector. */
            slarnv(2, iseed, blksiz, &work[indrv1]);

            /* Copy the matrix T so it won't be destroyed in factorization. */
            cblas_scopy(blksiz, &D[b1], 1, &work[indrv4], 1);
            cblas_scopy(blksiz - 1, &E[b1], 1, &work[indrv2 + 1], 1);
            cblas_scopy(blksiz - 1, &E[b1], 1, &work[indrv3], 1);

            /* Compute LU factors with partial pivoting  ( PT = LU ) */
            tol = 0.0f;
            slagtf(blksiz, &work[indrv4], xj,
                   &work[indrv2 + 1],
                   &work[indrv3], tol, &work[indrv5], iwork,
                   &iinfo);

            /* Update iteration count. */
            for (;;) {
                its++;
                if (its > MAXITS) {
                    goto convergence_failure;
                }

                /* Normalize and scale the righthand side vector Pb. */
                jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
                scl = (f32)blksiz * onenrm *
                      ((eps > fabsf(work[indrv4 + blksiz - 1])) ? eps : fabsf(work[indrv4 + blksiz - 1])) /
                      fabsf(work[indrv1 + jmax]);
                cblas_sscal(blksiz, scl, &work[indrv1], 1);

                /* Solve the system LU = Pb. */
                slagts(-1, blksiz, &work[indrv4],
                       &work[indrv2 + 1],
                       &work[indrv3], &work[indrv5], iwork,
                       &work[indrv1], &tol, &iinfo);

                /*
                 * Reorthogonalize by modified Gram-Schmidt if eigenvalues are
                 * close enough.
                 */
                if (jblk == 1) {
                    goto check_norm;
                }
                if (fabsf(xj - xjm) > ortol) {
                    gpind = j;
                }
                if (gpind != j) {
                    for (i = gpind; i < j; i++) {
                        ztr = -cblas_sdot(blksiz, &work[indrv1], 1,
                                          &Z[b1 + i * ldz], 1);
                        cblas_saxpy(blksiz, ztr, &Z[b1 + i * ldz], 1,
                                    &work[indrv1], 1);
                    }
                }

check_norm:
                /* Check the infinity norm of the iterate. */
                jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
                nrm = fabsf(work[indrv1 + jmax]);

                /*
                 * Continue for additional iterations after norm reaches
                 * stopping criterion.
                 */
                if (nrm < dtpcrt) {
                    continue;
                }
                nrmchk++;
                if (nrmchk < EXTRA + 1) {
                    continue;
                }

                goto accept_eigenvector;
            }

convergence_failure:
            /*
             * If stopping criterion was not satisfied, update info and
             * store eigenvector number in array ifail.
             */
            (*info)++;
            ifail[(*info) - 1] = j;

accept_eigenvector:
            /* Accept iterate as jth eigenvector. */
            scl = 1.0f / cblas_snrm2(blksiz, &work[indrv1], 1);
            jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
            if (work[indrv1 + jmax] < 0.0f) {
                scl = -scl;
            }
            cblas_sscal(blksiz, scl, &work[indrv1], 1);

store_eigenvector:
            for (i = 0; i < n; i++) {
                Z[i + j * ldz] = 0.0f;
            }
            for (i = 0; i < blksiz; i++) {
                Z[b1 + i + j * ldz] = work[indrv1 + i];
            }

            /* Save the shift to check eigenvalue spacing at next iteration. */
            xjm = xj;
        }
    }
}
