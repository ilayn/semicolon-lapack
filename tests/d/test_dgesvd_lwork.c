/**
 * @file test_dgesvd_lwork.c
 * @brief DGESVD regression test for the workspace layouts chosen from LWORK.
 *
 * In the paths for M much larger than N with JOBU = 'O' (and their
 * transposed counterparts with JOBVT = 'O'), DGESVD chooses WORK(IU) and
 * WORK(IR) as LDA by N and LDA by N, LDA by N and N by N, or LDWRKU by N
 * and N by N, with LDWRKU = (LWORK-N*N-N)/N, depending on LWORK. Each case
 * calls DGESVD for every JOBU and JOBVT, with 13 LWORK values from the
 * minimum to the optimum when JOBU or JOBVT is 'O' and with the minimum and
 * the optimum otherwise, checks that nothing after LWORK is written,
 * compares the singular values with those computed with the optimal LWORK,
 * and checks the residual and the orthogonality of the computed singular
 * vectors.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0
#define NGUARD 64
#define NSTEP 12

static const f64 CANARY = -98765.0;

static void run_shape(const INT m, const INT n)
{
    const char* jobs[4] = {"N", "O", "S", "A"};
    const INT lda = m, ldu = m, ldvt = n;
    const INT mn = (m < n) ? m : n;
    const INT mx = (m > n) ? m : n;
    const f64 eps = dlamch("P");
    INT info;
    f64 wq[1];
    uint64_t rng_state[4];

    f64* A0 = malloc((size_t)lda * n * sizeof(f64));
    f64* A = malloc((size_t)lda * n * sizeof(f64));
    f64* U = malloc((size_t)ldu * m * sizeof(f64));
    f64* VT = malloc((size_t)ldvt * n * sizeof(f64));
    f64* Uf = malloc((size_t)m * mn * sizeof(f64));
    f64* VTf = malloc((size_t)mn * n * sizeof(f64));
    f64* S = malloc(mn * sizeof(f64));
    f64* Sref = malloc(mn * sizeof(f64));
    f64* E = calloc(mn, sizeof(f64));
    const INT lrw = (m + n) + mx * (mx + 1);
    f64* rw = malloc((size_t)lrw * sizeof(f64));
    assert_non_null(A0); assert_non_null(A); assert_non_null(U);
    assert_non_null(VT); assert_non_null(Uf); assert_non_null(VTf);
    assert_non_null(S); assert_non_null(Sref); assert_non_null(E);
    assert_non_null(rw);

    rng_seed(rng_state, 0x6E5DULL + (uint64_t)(m * 1000 + n));
    for (INT i = 0; i < lda * n; i++) A0[i] = rng_uniform_symmetric(rng_state);

    memcpy(A, A0, (size_t)lda * n * sizeof(f64));
    dgesvd("N", "N", m, n, A, lda, Sref, U, ldu, VT, ldvt, wq, -1, &info);
    assert_info_success(info);
    {
        INT lw = (INT)wq[0];
        f64* w = malloc((size_t)lw * sizeof(f64));
        assert_non_null(w);
        dgesvd("N", "N", m, n, A, lda, Sref, U, ldu, VT, ldvt, w, lw, &info);
        assert_info_success(info);
        free(w);
    }

    for (INT ju = 0; ju < 4; ju++) {
        for (INT jv = 0; jv < 4; jv++) {
            const char* jobu = jobs[ju];
            const char* jobvt = jobs[jv];
            if (jobu[0] == 'O' && jobvt[0] == 'O') continue;

            memcpy(A, A0, (size_t)lda * n * sizeof(f64));
            dgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, wq, -1, &info);
            assert_info_success(info);
            const INT lwopt = (INT)wq[0];
            INT lwmin = 3 * mn + mx;
            if (lwmin < 5 * mn) lwmin = 5 * mn;
            if (lwmin < 1) lwmin = 1;
            f64* work = malloc((size_t)(lwopt + NGUARD) * sizeof(f64));
            assert_non_null(work);

            const INT nstep = (jobu[0] == 'O' || jobvt[0] == 'O') ? NSTEP : 1;
            for (INT k = 0; k <= nstep; k++) {
                const INT lwork = lwmin + (INT)(((int64_t)(lwopt - lwmin) * k) / nstep);
                f64 resid, serr;

                memcpy(A, A0, (size_t)lda * n * sizeof(f64));
                for (INT i = 0; i < lwork + NGUARD; i++) work[i] = CANARY;
                dgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
                       work, lwork, &info);
                assert_info_success(info);
                for (INT i = lwork; i < lwork + NGUARD; i++) {
                    if (work[i] != CANARY) {
                        fail_msg("JOBU=%s JOBVT=%s M=%lld N=%lld LWORK=%lld: "
                                 "WORK(%lld) written", jobu, jobvt,
                                 (long long)m, (long long)n,
                                 (long long)lwork, (long long)(i + 1));
                    }
                }

                serr = 0.0;
                for (INT i = 0; i < mn; i++) {
                    f64 d = fabs(S[i] - Sref[i]);
                    if (d > serr) serr = d;
                }
                serr = serr / ((f64)mn * eps * Sref[0]);
                assert_residual_ok(serr);

                INT haveu = (jobu[0] != 'N');
                INT havev = (jobvt[0] != 'N');
                if (haveu) {
                    const f64* src = (jobu[0] == 'O') ? A : U;
                    const INT ld = (jobu[0] == 'O') ? lda : ldu;
                    for (INT j = 0; j < mn; j++)
                        for (INT i = 0; i < m; i++)
                            Uf[i + j * m] = src[i + j * ld];
                    dort01("Columns", m, mn, Uf, m, rw, lrw, &resid);
                    assert_residual_ok(resid);
                }
                if (havev) {
                    const f64* src = (jobvt[0] == 'O') ? A : VT;
                    const INT ld = (jobvt[0] == 'O') ? lda : ldvt;
                    for (INT j = 0; j < n; j++)
                        for (INT i = 0; i < mn; i++)
                            VTf[i + j * mn] = src[i + j * ld];
                    dort01("Rows", mn, n, VTf, mn, rw, lrw, &resid);
                    assert_residual_ok(resid);
                }
                if (haveu && havev) {
                    dbdt01(m, n, 0, A0, lda, Uf, m, S, E, VTf, mn, rw, &resid);
                    assert_residual_ok(resid);
                }
            }
            free(work);
        }
    }

    free(A0); free(A); free(U); free(VT); free(Uf); free(VTf);
    free(S); free(Sref); free(E); free(rw);
}

static void test_tall(void** state)
{
    (void)state;
    run_shape(200, 100);
}

static void test_wide(void** state)
{
    (void)state;
    run_shape(100, 200);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_tall),
        cmocka_unit_test(test_wide),
    };
    (void)cmocka_run_group_tests_name("dgesvd_lwork", tests, NULL, NULL);
    return 0;
}
