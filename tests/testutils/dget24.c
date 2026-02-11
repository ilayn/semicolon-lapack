/**
 * @file dget24.c
 * @brief DGET24 checks the nonsymmetric eigenvalue (Schur form) problem
 *        expert driver DGEESX.
 *
 * Port of LAPACK's TESTING/EIG/dget24.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern double dlapy2(const double x, const double y);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);

/* Forward declaration - defined in semicolon_lapack_double.h */
typedef int (*dselect2_t_local)(const double* wr, const double* wi);
extern void dgeesx(const char* jobvs, const char* sort, dselect2_t_local select,
                   const char* sense, const int n, double* A, const int lda,
                   int* sdim, double* wr, double* wi, double* VS, const int ldvs,
                   double* rconde, double* rcondv, double* work, const int lwork,
                   int* iwork, const int liwork, int* bwork, int* info);

/* File-static globals for SSLCT COMMON block */
static int g_selopt;
static int g_seldim;
static int g_selval[20];
static double g_selwr[20];
static double g_selwi[20];

/**
 * DSLECT returns .TRUE. if the eigenvalue ZR+sqrt(-1)*ZI is to be
 * selected, and otherwise it returns .FALSE.
 * It is used by DGEESX to test whether the j-th eigenvalue is to be
 * reordered to the top left corner of the Schur form.
 */
static int dslect(const double* zr, const double* zi)
{
    int i;
    double rmin, x;

    if (g_selopt == 0) {
        return (*zr < 0.0);
    } else {
        rmin = dlapy2(*zr - g_selwr[0], *zi - g_selwi[0]);
        int val = g_selval[0];
        for (i = 1; i < g_seldim; i++) {
            x = dlapy2(*zr - g_selwr[i], *zi - g_selwi[i]);
            if (x <= rmin) {
                rmin = x;
                val = g_selval[i];
            }
        }
        return val;
    }
}

/**
 * DGET24 checks the nonsymmetric eigenvalue (Schur form) problem
 * expert driver DGEESX.
 *
 * If COMP = 0, the first 13 of the following tests will be performed
 * on the input matrix A, and also tests 14 and 15 if LWORK is
 * sufficiently large.
 * If COMP = 1, all 17 tests will be performed.
 *
 *    (1)     0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *    (2)     | A - VS T VS' | / ( n |A| ulp ) (no sorting)
 *    (3)     | I - VS VS' | / ( n ulp ) (no sorting)
 *    (4)     0 if WR+sqrt(-1)*WI are eigenvalues of T, 1/ulp otherwise
 *    (5)     0 if T(with VS) = T(without VS), 1/ulp otherwise
 *    (6)     0 if eigenvalues(with VS) = eigenvalues(without VS)
 *    (7)     0 if T is in Schur form (with sorting), 1/ulp otherwise
 *    (8)     | A - VS T VS' | / ( n |A| ulp ) (with sorting)
 *    (9)     | I - VS VS' | / ( n ulp ) (with sorting)
 *   (10)     0 if WR+sqrt(-1)*WI are eigenvalues of T (with sorting)
 *   (11)     0 if T(with VS) = T(without VS) (with sorting)
 *   (12)     0 if eigenvalues(with VS) = eigenvalues(without VS)
 *   (13)     if sorting worked and SDIM is the number of eigenvalues selected
 *   (14)     if RCONDE the same no matter if VS and/or RCONDV computed
 *   (15)     if RCONDV the same no matter if VS and/or RCONDE computed
 *   (16)     |RCONDE - RCDEIN| / cond(RCONDE)
 *   (17)     |RCONDV - RCDVIN| / cond(RCONDV)
 */
void dget24(const int comp, const int jtype, const double thresh,
            const int n, double* A, const int lda,
            double* H, double* HT,
            double* wr, double* wi, double* wrt, double* wit,
            double* wrtmp, double* witmp,
            double* VS, const int ldvs, double* VS1,
            const double rcdein, const double rcdvin,
            const int nslct, const int* islct,
            double* result, double* work, const int lwork,
            int* iwork, int* bwork, int* info)
{
    (void)jtype;
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double EPSIN = 5.9605e-8;

    int i, j, kmin, knteig, liwork, rsub, sdim, sdim1;
    int iinfo, isort, itmp;
    double anorm, eps, rcnde1, rcndv1, rconde, rcondv;
    double smlnum, tmp, tol, tolin, ulp, ulpinv, v;
    double vimin, vrmin, wnorm;
    int ipnt[20];

    /* Check for errors */
    *info = 0;
    if (thresh < ZERO) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < 1 || lda < n) {
        *info = -6;
    } else if (ldvs < 1 || ldvs < n) {
        *info = -15;
    } else if (lwork < 3 * n) {
        *info = -22;
    }

    if (*info != 0)
        return;

    /* Quick return if nothing to do */
    for (i = 0; i < 17; i++)
        result[i] = -ONE;

    if (n == 0)
        return;

    /* Important constants */
    smlnum = dlamch("S");
    ulp = dlamch("P");
    ulpinv = ONE / ulp;

    /* Perform tests (1)-(13) */
    g_selopt = 0;
    liwork = n * n;
    for (isort = 0; isort <= 1; isort++) {
        if (isort == 0) {
            rsub = 0;
        } else {
            rsub = 6;
        }

        /* Compute Schur form and Schur vectors, and test them */
        dlacpy("F", n, n, A, lda, H, lda);
        dgeesx("V", isort == 0 ? "N" : "S", dslect, "N", n, H, lda, &sdim,
               wr, wi, VS, ldvs, &rconde, &rcondv, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[0 + rsub] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            return;
        }
        if (isort == 0) {
            cblas_dcopy(n, wr, 1, wrtmp, 1);
            cblas_dcopy(n, wi, 1, witmp, 1);
        }

        /* Do Test (1) or Test (7) */
        result[0 + rsub] = ZERO;
        for (j = 0; j < n - 2; j++) {
            for (i = j + 2; i < n; i++) {
                if (H[i + j * lda] != ZERO)
                    result[0 + rsub] = ulpinv;
            }
        }
        for (i = 0; i < n - 2; i++) {
            if (H[(i + 1) + i * lda] != ZERO && H[(i + 2) + (i + 1) * lda] != ZERO)
                result[0 + rsub] = ulpinv;
        }
        for (i = 0; i < n - 1; i++) {
            if (H[(i + 1) + i * lda] != ZERO) {
                if (H[i + i * lda] != H[(i + 1) + (i + 1) * lda] ||
                    H[i + (i + 1) * lda] == ZERO ||
                    (H[(i + 1) + i * lda] > ZERO) == (H[i + (i + 1) * lda] > ZERO))
                    result[0 + rsub] = ulpinv;
            }
        }

        /* Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP) */

        /* Copy A to VS1, used as workspace */
        dlacpy(" ", n, n, A, lda, VS1, ldvs);

        /* Compute Q*H and store in HT */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, ONE, VS, ldvs, H, lda, ZERO, HT, lda);

        /* Compute A - Q*H*Q' */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    n, n, n, -ONE, HT, lda, VS, ldvs, ONE, VS1, ldvs);

        anorm = fmax(dlange("1", n, n, A, lda, work), smlnum);
        wnorm = dlange("1", n, n, VS1, ldvs, work);

        if (anorm > wnorm) {
            result[1 + rsub] = (wnorm / anorm) / (n * ulp);
        } else {
            if (anorm < ONE) {
                result[1 + rsub] = (fmin(wnorm, n * anorm) / anorm) / (n * ulp);
            } else {
                result[1 + rsub] = fmin(wnorm / anorm, (double)n) / (n * ulp);
            }
        }

        /* Test (3) or (9):  Compute norm( I - Q'*Q ) / ( N * ULP ) */
        dort01("Columns", n, n, VS, ldvs, work, lwork, &result[2 + rsub]);

        /* Do Test (4) or Test (10) */
        result[3 + rsub] = ZERO;
        for (i = 0; i < n; i++) {
            if (H[i + i * lda] != wr[i])
                result[3 + rsub] = ulpinv;
        }
        if (n > 1) {
            if (H[1 + 0 * lda] == ZERO && wi[0] != ZERO)
                result[3 + rsub] = ulpinv;
            if (H[(n - 1) + (n - 2) * lda] == ZERO && wi[n - 1] != ZERO)
                result[3 + rsub] = ulpinv;
        }
        for (i = 0; i < n - 1; i++) {
            if (H[(i + 1) + i * lda] != ZERO) {
                tmp = sqrt(fabs(H[(i + 1) + i * lda])) *
                      sqrt(fabs(H[i + (i + 1) * lda]));
                result[3 + rsub] = fmax(result[3 + rsub],
                    fabs(wi[i] - tmp) / fmax(ulp * tmp, smlnum));
                result[3 + rsub] = fmax(result[3 + rsub],
                    fabs(wi[i + 1] + tmp) / fmax(ulp * tmp, smlnum));
            } else if (i > 0) {
                if (H[(i + 1) + i * lda] == ZERO && H[i + (i - 1) * lda] == ZERO &&
                    wi[i] != ZERO)
                    result[3 + rsub] = ulpinv;
            }
        }

        /* Do Test (5) or Test (11) */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("N", isort == 0 ? "N" : "S", dslect, "N", n, HT, lda, &sdim,
               wrt, wit, VS, ldvs, &rconde, &rcondv, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[4 + rsub] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        result[4 + rsub] = ZERO;
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[4 + rsub] = ulpinv;
            }
        }

        /* Do Test (6) or Test (12) */
        result[5 + rsub] = ZERO;
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[5 + rsub] = ulpinv;
        }

        /* Do Test (13) */
        if (isort == 1) {
            result[12] = ZERO;
            knteig = 0;
            for (i = 0; i < n; i++) {
                if (dslect(&wr[i], &wi[i]) || dslect(&wr[i], &(double){-wi[i]}))
                    knteig = knteig + 1;
                if (i < n - 1) {
                    if ((dslect(&wr[i + 1], &wi[i + 1]) ||
                         dslect(&wr[i + 1], &(double){-wi[i + 1]})) &&
                        (!(dslect(&wr[i], &wi[i]) ||
                           dslect(&wr[i], &(double){-wi[i]}))) &&
                        iinfo != n + 2)
                        result[12] = ulpinv;
                }
            }
            if (sdim != knteig)
                result[12] = ulpinv;
        }
    }

    /* If there is enough workspace, perform tests (14) and (15)
     * as well as (10) through (13) */
    if (lwork >= n + (n * n) / 2) {

        /* Compute both RCONDE and RCONDV with VS */
        result[13] = ZERO;
        result[14] = ZERO;
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("V", "S", dslect, "B", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rconde, &rcondv, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute both RCONDE and RCONDV without VS, and compare */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("N", "S", dslect, "B", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rcnde1, &rcndv1, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform tests (14) and (15) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDE with VS, and compare */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("V", "S", dslect, "E", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rcnde1, &rcndv1, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform test (14) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDE without VS, and compare */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("N", "S", dslect, "E", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rcnde1, &rcndv1, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform test (14) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDV with VS, and compare */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("V", "S", dslect, "V", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rcnde1, &rcndv1, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform test (15) */
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDV without VS, and compare */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("N", "S", dslect, "V", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rcnde1, &rcndv1, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Perform test (15) */
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (wr[i] != wrt[i] || wi[i] != wit[i])
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (H[i + j * lda] != HT[i + j * lda])
                    result[10] = ulpinv;
                if (VS[i + j * ldvs] != VS1[i + j * ldvs])
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;
    }

label_250:

    /* If there are precomputed reciprocal condition numbers, compare
     * computed values with them. */
    if (comp) {

        /* First set up SELOPT, SELDIM, SELVAL, SELWR, and SELWI so that
         * the logical function DSLECT selects the eigenvalues specified
         * by NSLCT and ISLCT. */
        g_seldim = n;
        g_selopt = 1;
        eps = fmax(ulp, EPSIN);
        for (i = 0; i < n; i++) {
            ipnt[i] = i;
            g_selval[i] = 0;
            g_selwr[i] = wrtmp[i];
            g_selwi[i] = witmp[i];
        }
        for (i = 0; i < n - 1; i++) {
            kmin = i;
            vrmin = wrtmp[i];
            vimin = witmp[i];
            for (j = i + 1; j < n; j++) {
                if (wrtmp[j] < vrmin) {
                    kmin = j;
                    vrmin = wrtmp[j];
                    vimin = witmp[j];
                }
            }
            wrtmp[kmin] = wrtmp[i];
            witmp[kmin] = witmp[i];
            wrtmp[i] = vrmin;
            witmp[i] = vimin;
            itmp = ipnt[i];
            ipnt[i] = ipnt[kmin];
            ipnt[kmin] = itmp;
        }
        for (i = 0; i < nslct; i++) {
            g_selval[ipnt[islct[i]]] = 1;
        }

        /* Compute condition numbers */
        dlacpy("F", n, n, A, lda, HT, lda);
        dgeesx("N", "S", dslect, "B", n, HT, lda, &sdim1, wrt,
               wit, VS1, ldvs, &rconde, &rcondv, work, lwork,
               iwork, liwork, bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[15] = ulpinv;
            result[16] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_300;
        }

        /* Compare condition number for average of selected eigenvalues
         * taking its condition number into account */
        anorm = dlange("1", n, n, A, lda, work);
        v = fmax((double)n * eps * anorm, smlnum);
        if (anorm == ZERO)
            v = ONE;
        if (v > rcondv) {
            tol = ONE;
        } else {
            tol = v / rcondv;
        }
        if (v > rcdvin) {
            tolin = ONE;
        } else {
            tolin = v / rcdvin;
        }
        tol = fmax(tol, smlnum / eps);
        tolin = fmax(tolin, smlnum / eps);
        if (eps * (rcdein - tolin) > rconde + tol) {
            result[15] = ulpinv;
        } else if (rcdein - tolin > rconde + tol) {
            result[15] = (rcdein - tolin) / (rconde + tol);
        } else if (rcdein + tolin < eps * (rconde - tol)) {
            result[15] = ulpinv;
        } else if (rcdein + tolin < rconde - tol) {
            result[15] = (rconde - tol) / (rcdein + tolin);
        } else {
            result[15] = ONE;
        }

        /* Compare condition numbers for right invariant subspace
         * taking its condition number into account */
        if (v > rcondv * rconde) {
            tol = rcondv;
        } else {
            tol = v / rconde;
        }
        if (v > rcdvin * rcdein) {
            tolin = rcdvin;
        } else {
            tolin = v / rcdein;
        }
        tol = fmax(tol, smlnum / eps);
        tolin = fmax(tolin, smlnum / eps);
        if (eps * (rcdvin - tolin) > rcondv + tol) {
            result[16] = ulpinv;
        } else if (rcdvin - tolin > rcondv + tol) {
            result[16] = (rcdvin - tolin) / (rcondv + tol);
        } else if (rcdvin + tolin < eps * (rcondv - tol)) {
            result[16] = ulpinv;
        } else if (rcdvin + tolin < rcondv - tol) {
            result[16] = (rcondv - tol) / (rcdvin + tolin);
        } else {
            result[16] = ONE;
        }

label_300:
        ;
    }
}
