/**
 * @file test_dstevx_scaling.c
 * @brief Regression tests for the scaling block of the DSTEBZ-based drivers.
 *
 * test_inf_range_v is a port of the infinity block that LAPACK added to
 * the xERRST error-exit tests: DSTEVX and DSYEVX with RANGE = 'V' on a
 * matrix holding an infinity must return without reaching XERBLA.
 * test_inf_w_bound runs the scaling drivers with RANGE = 'I' on the same
 * kind of matrix and checks that nothing is written past W(N).
 * test_abstol_scaled runs DSTEVX and DSTEVR with a positive ABSTOL on the
 * xDRVST type 19 matrix, which the drivers scale down, and compares the
 * eigenvalues with DSTEV as xDRVST does.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0
#define NMAX 3
#define LIW (12 * NMAX)
#define LW (20 * NMAX)
#define NGUARD 8
#define CANARY 777.0

static void test_inf_range_v(void** state)
{
    (void)state;
    INT iw[LIW], i3[NMAX];
    INT m, info, lerr_stevx, lerr_syevx;
    f64 A[NMAX * NMAX], D[NMAX], E[NMAX], W[LW], X[NMAX], Z[NMAX * NMAX];
    f64 rzero, rinf;

    rzero = 0.0;
    rinf = INFINITY;
    for (INT j = 0; j < NMAX; j++) {
        D[j] = (f64)(j + 1);
        E[j] = 1.0 / (f64)(j + 2);
        for (INT i = 0; i < NMAX; i++) {
            A[i + j * NMAX] = rzero;
        }
        A[j + j * NMAX] = D[j];
    }
    D[0] = rinf;
    A[0 + 0 * NMAX] = rinf;
    E[NMAX - 1] = rzero;
    A[0 + 1 * NMAX] = E[0];
    A[1 + 2 * NMAX] = E[1];
    strcpy(xerbla_srnamt, "DSTEVX");
    xerbla_infot = 0;
    xerbla_lerr = 0;
    dstevx("V", "V", NMAX, D, E, rzero, 1.0, 0, NMAX - 1,
           rzero, &m, X, Z, NMAX, W, iw, i3, &info);
    lerr_stevx = xerbla_lerr;
    D[0] = rinf;
    E[0] = 1.0 / (f64)2;
    E[1] = 1.0 / (f64)3;
    E[NMAX - 1] = rzero;
    strcpy(xerbla_srnamt, "DSYEVX");
    xerbla_infot = 0;
    xerbla_lerr = 0;
    dsyevx("V", "V", "U", NMAX, A, NMAX, rzero, 1.0, 0,
           NMAX - 1, rzero, &m, X, Z, NMAX, W, LW, iw, i3, &info);
    lerr_syevx = xerbla_lerr;
    xerbla_srnamt[0] = '\0';
    xerbla_lerr = 0;
    xerbla_ok = 1;

    if (lerr_stevx) {
        fail_msg("%s called XERBLA for a matrix with an infinity", "DSTEVX");
    }
    if (lerr_syevx) {
        fail_msg("%s called XERBLA for a matrix with an infinity", "DSYEVX");
    }
}

static INT count_hit(const f64* W, const INT n)
{
    INT hit = 0;
    for (INT i = n; i < n + NGUARD; i++) {
        if (!(W[i] == CANARY)) {
            hit++;
        }
    }
    return hit;
}

/* Same kind of matrix as test_inf_range_v at order N = 6.  RANGE = 'I'
   with ABSTOL = 0 takes the DSTEBZ path, and a failure in DSTEBZ must
   not make the rescale of W run past W(N). */
static void test_inf_w_bound(void** state)
{
    (void)state;
    const INT n = 6;
    const INT il = 1, iu = 3;
    const char* names[6] = {"DSTEVX", "DSYEVX", "DSPEVX", "DSBEVX",
                            "DSYEVX_2STAGE", "DSBEVX_2STAGE"};
    f64 D[6], E[6], A[36], AP[21], AB[12], Q[36], Z[36];
    f64 W[6 + NGUARD];
    f64 wq;
    INT iwork[30], ifail[6];
    INT m, info, nfail = 0;
    char msg[512] = "";

    memset(A, 0, sizeof(A));
    memset(AB, 0, sizeof(AB));
    INT lwork = 8 * n;
    dsyevx_2stage("N", "I", "U", n, A, n, 0.0, 0.0, il, iu, 0.0, &m,
                  W, Z, n, &wq, -1, iwork, ifail, &info);
    assert_info_success(info);
    if ((INT)wq > lwork) {
        lwork = (INT)wq;
    }
    dsbevx_2stage("N", "I", "U", n, 1, AB, 2, Q, n, 0.0, 0.0, il, iu,
                  0.0, &m, W, Z, n, &wq, -1, iwork, ifail, &info);
    assert_info_success(info);
    if ((INT)wq > lwork) {
        lwork = (INT)wq;
    }
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(work);

    for (INT k = 0; k < 6; k++) {
        for (INT j = 0; j < n; j++) {
            D[j] = (f64)(j + 1);
            E[j] = 1.0 / (f64)(j + 2);
        }
        D[0] = INFINITY;
        E[n - 1] = 0.0;
        memset(A, 0, sizeof(A));
        for (INT j = 0; j < n; j++) {
            A[j + j * n] = D[j];
            if (j + 1 < n) {
                A[j + (j + 1) * n] = E[j];
            }
        }
        INT p = 0;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i <= j; i++) {
                AP[p++] = A[i + j * n];
            }
        }
        for (INT j = 0; j < n; j++) {
            AB[1 + j * 2] = D[j];
            AB[0 + j * 2] = (j > 0) ? E[j - 1] : 0.0;
        }
        for (INT i = 0; i < n + NGUARD; i++) {
            W[i] = CANARY;
        }
        m = -1;
        info = 0;

        switch (k) {
        case 0:
            dstevx("N", "I", n, D, E, 0.0, 0.0, il, iu, 0.0, &m, W, Z, n,
                   work, iwork, ifail, &info);
            break;
        case 1:
            dsyevx("N", "I", "U", n, A, n, 0.0, 0.0, il, iu, 0.0, &m, W, Z, n,
                   work, lwork, iwork, ifail, &info);
            break;
        case 2:
            dspevx("N", "I", "U", n, AP, 0.0, 0.0, il, iu, 0.0, &m, W, Z, n,
                   work, iwork, ifail, &info);
            break;
        case 3:
            dsbevx("N", "I", "U", n, 1, AB, 2, Q, n, 0.0, 0.0, il, iu, 0.0,
                   &m, W, Z, n, work, iwork, ifail, &info);
            break;
        case 4:
            dsyevx_2stage("N", "I", "U", n, A, n, 0.0, 0.0, il, iu, 0.0, &m,
                          W, Z, n, work, lwork, iwork, ifail, &info);
            break;
        default:
            dsbevx_2stage("N", "I", "U", n, 1, AB, 2, Q, n, 0.0, 0.0, il, iu,
                          0.0, &m, W, Z, n, work, lwork, iwork, ifail, &info);
            break;
        }

        INT hit = count_hit(W, n);
        if (hit > 0 || m < 0 || m > n) {
            nfail++;
            size_t len = strlen(msg);
            snprintf(msg + len, sizeof(msg) - len,
                     " %s: INFO = %lld, M = %lld, %lld of %d guard entries after W(N) overwritten;",
                     names[k], (long long)info, (long long)m, (long long)hit,
                     NGUARD);
        }
    }
    free(work);

    if (nfail > 0) {
        fail_msg("%s", msg);
    }
}

/* xDRVST type 19: tridiagonal with ANORM on the diagonal and ANORM/2 off
   it, ANORM at the "large" magnitude, ABSTOL = TEN*ULP*ANORM. */
static void test_abstol_scaled(void** state)
{
    (void)state;
    const INT n = 5;
    f64 D1[5], E1[5], D2[5], E2[5], W[5], Z[25], work[100];
    INT iwork[50], ifail[5], isuppz[10];
    INT m, info;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = dlamch("O");
    f64 rtovfl = sqrt(ovfl);
    f64 aninv = 1.0 / (f64)n;
    f64 anorm = (rtovfl * ulp) * aninv;
    f64 abstol = 10.0 * ulp * anorm;

    for (INT i = 0; i < n; i++) {
        D1[i] = anorm;
        E1[i] = (i < n - 1) ? anorm / 2.0 : 0.0;
    }
    dstev("N", n, D1, E1, Z, n, work, &info);
    assert_info_success(info);

    for (INT r = 0; r < 2; r++) {
        for (INT i = 0; i < n; i++) {
            D2[i] = anorm;
            E2[i] = (i < n - 1) ? anorm / 2.0 : 0.0;
        }
        if (r == 0) {
            set_test_context("DSTEVX('N','A') xDRVST type 19");
            dstevx("N", "A", n, D2, E2, 0.0, 0.0, 0, 0, abstol, &m, W, Z, n,
                   work, iwork, ifail, &info);
        } else {
            set_test_context("DSTEVR('N','V') xDRVST type 19");
            dstevr("N", "V", n, D2, E2, -anorm, 3.0 * anorm, 0, 0, abstol,
                   &m, W, Z, n, isuppz, work, 20 * n, iwork, 10 * n, &info);
        }
        assert_info_success(info);
        assert_int_equal(m, n);

        f64 temp1 = 0.0, temp2 = 0.0, temp3 = 0.0;
        for (INT j = 0; j < n; j++) {
            temp1 = fmax(temp1, fabs(D1[j]));
            temp2 = fmax(temp2, fabs(W[j]));
            temp3 = fmax(temp3, fabs(D1[j] - W[j]));
        }
        f64 result = temp3 / fmax(unfl, ulp * fmax(temp1, temp2));
        assert_residual_ok(result);
    }
    clear_test_context();
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_inf_range_v),
        cmocka_unit_test(test_inf_w_bound),
        cmocka_unit_test(test_abstol_scaled),
    };
    (void)cmocka_run_group_tests_name("dstevx_scaling", tests, NULL, NULL);
    return 0;
}
