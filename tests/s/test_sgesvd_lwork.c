/**
 * @file test_sgesvd_lwork.c
 * @brief SGESVD regression test for the workspace layouts chosen from LWORK.
 *
 * In the paths for M much larger than N with JOBU = 'O' (and their
 * transposed counterparts with JOBVT = 'O'), SGESVD chooses WORK(IU) and
 * WORK(IR) as LDA by N and LDA by N, LDA by N and N by N, or LDWRKU by N
 * and N by N, with LDWRKU = (LWORK-N*N-N)/N, depending on LWORK. Each case
 * calls SGESVD for every JOBU and JOBVT with LWORK from the minimum to the
 * optimum, checks that nothing after LWORK is written, compares the
 * singular values with those computed with the optimal LWORK, and checks
 * the residual and the orthogonality of the computed singular vectors.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0f
#define NGUARD 64
#define NSTEP 24

static const f32 CANARY = -98765.0f;

static void run_shape(const INT m, const INT n)
{
    const char* jobs[4] = {"N", "O", "S", "A"};
    const INT lda = m, ldu = m, ldvt = n;
    const INT mn = (m < n) ? m : n;
    const INT mx = (m > n) ? m : n;
    const f32 eps = slamch("P");
    INT info;
    f32 wq[1];
    uint64_t rng_state[4];

    f32* A0 = malloc((size_t)lda * n * sizeof(f32));
    f32* A = malloc((size_t)lda * n * sizeof(f32));
    f32* U = malloc((size_t)ldu * m * sizeof(f32));
    f32* VT = malloc((size_t)ldvt * n * sizeof(f32));
    f32* Uf = malloc((size_t)m * mn * sizeof(f32));
    f32* VTf = malloc((size_t)mn * n * sizeof(f32));
    f32* S = malloc(mn * sizeof(f32));
    f32* Sref = malloc(mn * sizeof(f32));
    f32* E = calloc(mn, sizeof(f32));
    const INT lrw = (m + n) + mx * (mx + 1);
    f32* rw = malloc((size_t)lrw * sizeof(f32));
    assert_non_null(A0); assert_non_null(A); assert_non_null(U);
    assert_non_null(VT); assert_non_null(Uf); assert_non_null(VTf);
    assert_non_null(S); assert_non_null(Sref); assert_non_null(E);
    assert_non_null(rw);

    rng_seed(rng_state, 0x6E5DULL + (uint64_t)(m * 1000 + n));
    for (INT i = 0; i < lda * n; i++) A0[i] = rng_uniform_symmetric_f32(rng_state);

    memcpy(A, A0, (size_t)lda * n * sizeof(f32));
    sgesvd("N", "N", m, n, A, lda, Sref, U, ldu, VT, ldvt, wq, -1, &info);
    assert_info_success(info);
    {
        INT lw = (INT)wq[0];
        f32* w = malloc((size_t)lw * sizeof(f32));
        assert_non_null(w);
        sgesvd("N", "N", m, n, A, lda, Sref, U, ldu, VT, ldvt, w, lw, &info);
        assert_info_success(info);
        free(w);
    }

    for (INT ju = 0; ju < 4; ju++) {
        for (INT jv = 0; jv < 4; jv++) {
            const char* jobu = jobs[ju];
            const char* jobvt = jobs[jv];
            if (jobu[0] == 'O' && jobvt[0] == 'O') continue;

            memcpy(A, A0, (size_t)lda * n * sizeof(f32));
            sgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, wq, -1, &info);
            assert_info_success(info);
            const INT lwopt = (INT)wq[0];
            INT lwmin = 3 * mn + mx;
            if (lwmin < 5 * mn) lwmin = 5 * mn;
            if (lwmin < 1) lwmin = 1;
            f32* work = malloc((size_t)(lwopt + NGUARD) * sizeof(f32));
            assert_non_null(work);

            for (INT k = 0; k <= NSTEP; k++) {
                const INT lwork = lwmin + (INT)(((int64_t)(lwopt - lwmin) * k) / NSTEP);
                f32 resid, serr;

                memcpy(A, A0, (size_t)lda * n * sizeof(f32));
                for (INT i = 0; i < lwork + NGUARD; i++) work[i] = CANARY;
                sgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
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

                serr = 0.0f;
                for (INT i = 0; i < mn; i++) {
                    f32 d = fabsf(S[i] - Sref[i]);
                    if (d > serr) serr = d;
                }
                serr = serr / ((f32)mn * eps * Sref[0]);
                assert_residual_ok(serr);

                INT haveu = (jobu[0] != 'N');
                INT havev = (jobvt[0] != 'N');
                if (haveu) {
                    const f32* src = (jobu[0] == 'O') ? A : U;
                    const INT ld = (jobu[0] == 'O') ? lda : ldu;
                    for (INT j = 0; j < mn; j++)
                        for (INT i = 0; i < m; i++)
                            Uf[i + j * m] = src[i + j * ld];
                    sort01("Columns", m, mn, Uf, m, rw, lrw, &resid);
                    assert_residual_ok(resid);
                }
                if (havev) {
                    const f32* src = (jobvt[0] == 'O') ? A : VT;
                    const INT ld = (jobvt[0] == 'O') ? lda : ldvt;
                    for (INT j = 0; j < n; j++)
                        for (INT i = 0; i < mn; i++)
                            VTf[i + j * mn] = src[i + j * ld];
                    sort01("Rows", mn, n, VTf, mn, rw, lrw, &resid);
                    assert_residual_ok(resid);
                }
                if (haveu && havev) {
                    sbdt01(m, n, 0, A0, lda, Uf, m, S, E, VTf, mn, rw, &resid);
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
    (void)cmocka_run_group_tests_name("sgesvd_lwork", tests, NULL, NULL);
    return 0;
}
