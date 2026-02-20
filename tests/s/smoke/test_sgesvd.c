/**
 * @file test_sgesvd.c
 * @brief CMocka test suite for sgesvd (Singular Value Decomposition).
 *
 * Tests the SVD routine sgesvd using LAPACK's verification methodology
 * with normalized residuals.
 *
 * Verification:
 *   1. Singular values non-negative and sorted descending
 *   2. ||A - U*S*VT|| / (||A||*max(m,n)*eps) < threshold
 *   3. ||I - U'*U|| / (min(m,n)*eps) < threshold (orthogonality of U)
 *   4. ||I - VT*VT'|| / (min(m,n)*eps) < threshold (orthogonality of VT)
 *
 * Matrix types tested:
 *   1. Zero matrix
 *   2. Identity-like (diagonal of 1s)
 *   3. Random well-conditioned
 *   4. Random ill-conditioned
 *   5. Rank-deficient
 */

#include "test_harness.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold */
#define THRESH 30.0f

/* Routine under test */
extern void sgesvd(const char* jobu, const char* jobvt,
                   const int m, const int n,
                   f32* A, const int lda,
                   f32* S,
                   f32* U, const int ldu,
                   f32* VT, const int ldvt,
                   f32* work, const int lwork,
                   int* info);

/* Utilities */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int m, n;
    f32* A;          /* original matrix (preserved) */
    f32* Acopy;      /* working copy (overwritten by sgesvd) */
    f32* S;          /* singular values */
    f32* U;          /* left singular vectors */
    f32* VT;         /* right singular vectors (transposed) */
    f32* work;       /* workspace */
    int lwork;          /* workspace size */
    f32* temp;       /* temporary workspace for verification */
    uint64_t rng_state[4];
} dgesvd_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2024;

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgesvd_setup(void** state, int m, int n)
{
    dgesvd_fixture_t* fix = malloc(sizeof(dgesvd_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    rng_seed(fix->rng_state, g_seed++);

    int minmn = (m < n) ? m : n;
    int maxmn = (m > n) ? m : n;

    /* Query optimal workspace size */
    f32 work_query;
    int info;
    sgesvd("A", "A", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    fix->lwork = (int)work_query;
    if (fix->lwork < 1) fix->lwork = 5 * maxmn;

    /* Allocate arrays */
    fix->A = malloc(m * n * sizeof(f32));
    fix->Acopy = malloc(m * n * sizeof(f32));
    fix->S = malloc(minmn * sizeof(f32));
    fix->U = malloc(m * m * sizeof(f32));
    fix->VT = malloc(n * n * sizeof(f32));
    fix->work = malloc(fix->lwork * sizeof(f32));
    fix->temp = malloc(maxmn * maxmn * sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->Acopy);
    assert_non_null(fix->S);
    assert_non_null(fix->U);
    assert_non_null(fix->VT);
    assert_non_null(fix->work);
    assert_non_null(fix->temp);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgesvd_teardown(void** state)
{
    dgesvd_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->S);
        free(fix->U);
        free(fix->VT);
        free(fix->work);
        free(fix->temp);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_5x5(void** state) { return dgesvd_setup(state, 5, 5); }
static int setup_10x10(void** state) { return dgesvd_setup(state, 10, 10); }
static int setup_5x3(void** state) { return dgesvd_setup(state, 5, 3); }
static int setup_3x5(void** state) { return dgesvd_setup(state, 3, 5); }
static int setup_20x10(void** state) { return dgesvd_setup(state, 20, 10); }
static int setup_10x20(void** state) { return dgesvd_setup(state, 10, 20); }
static int setup_50x30(void** state) { return dgesvd_setup(state, 50, 30); }
static int setup_30x50(void** state) { return dgesvd_setup(state, 30, 50); }

/**
 * Helper: generate random m x n matrix
 */
static void generate_random_matrix(f32* A, int m, int n, uint64_t state[static 4])
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * m] = rng_uniform_symmetric_f32(state);
        }
    }
}

/**
 * Helper: generate random matrix with specified singular values
 */
static void generate_matrix_with_sv(f32* A, int m, int n, const f32* sv,
                                    uint64_t state[static 4])
{
    int minmn = (m < n) ? m : n;

    /* Start with diagonal of singular values */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (i == j && i < minmn) {
                A[i + j * m] = sv[i];
            } else {
                A[i + j * m] = 0.0f;
            }
        }
    }

    /* Apply random orthogonal transformations from left and right */
    /* For simplicity, just use Householder reflections */
    f32* v = malloc(((m > n) ? m : n) * sizeof(f32));

    /* Left transformations */
    for (int k = 0; k < minmn; k++) {
        /* Random Householder vector */
        f32 nrm = 0.0f;
        for (int i = 0; i < m; i++) {
            v[i] = rng_uniform_symmetric_f32(state);
            nrm += v[i] * v[i];
        }
        nrm = sqrtf(nrm);
        if (nrm > 0) {
            for (int i = 0; i < m; i++) v[i] /= nrm;
        }

        /* Apply H = I - 2*v*v' to A from left */
        for (int j = 0; j < n; j++) {
            f32 dot = 0.0f;
            for (int i = 0; i < m; i++) {
                dot += v[i] * A[i + j * m];
            }
            for (int i = 0; i < m; i++) {
                A[i + j * m] -= 2.0f * v[i] * dot;
            }
        }
    }

    /* Right transformations */
    for (int k = 0; k < minmn; k++) {
        /* Random Householder vector */
        f32 nrm = 0.0f;
        for (int j = 0; j < n; j++) {
            v[j] = rng_uniform_symmetric_f32(state);
            nrm += v[j] * v[j];
        }
        nrm = sqrtf(nrm);
        if (nrm > 0) {
            for (int j = 0; j < n; j++) v[j] /= nrm;
        }

        /* Apply H = I - 2*v*v' to A from right */
        for (int i = 0; i < m; i++) {
            f32 dot = 0.0f;
            for (int j = 0; j < n; j++) {
                dot += A[i + j * m] * v[j];
            }
            for (int j = 0; j < n; j++) {
                A[i + j * m] -= 2.0f * dot * v[j];
            }
        }
    }

    free(v);
}

/**
 * Helper: check singular values are non-negative and sorted descending
 */
static int check_sv_sorted(const f32* S, int n)
{
    for (int i = 0; i < n; i++) {
        if (S[i] < 0.0f) return 0;  /* negative singular value */
    }
    for (int i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1]) return 0;  /* not descending */
    }
    return 1;
}

/**
 * Helper: compute ||A - U*diag(S)*VT|| / (||A||*max(m,n)*eps)
 */
static f32 compute_svd_residual(const f32* A, int m, int n,
                                   const f32* U, int ldu,
                                   const f32* S,
                                   const f32* VT, int ldvt,
                                   f32* temp)
{
    f32 eps = slamch("E");
    int minmn = (m < n) ? m : n;
    int maxmn = (m > n) ? m : n;

    /* Compute ||A|| */
    f32 anrm = slange("F", m, n, A, m, NULL);
    if (anrm == 0.0f) anrm = 1.0f;

    /* Compute U*diag(S)*VT into temp */
    /* First: temp2 = diag(S)*VT (scale rows of VT by S) */
    f32* temp2 = malloc(minmn * n * sizeof(f32));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < minmn; i++) {
            temp2[i + j * minmn] = S[i] * VT[i + j * ldvt];
        }
    }

    /* Then: temp = U * temp2 */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, minmn, 1.0f, U, ldu, temp2, minmn, 0.0f, temp, m);

    free(temp2);

    /* Compute temp = A - temp */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            temp[i + j * m] -= A[i + j * m];
        }
    }

    /* Compute ||A - U*S*VT|| */
    f32 resid = slange("F", m, n, temp, m, NULL);

    return resid / (anrm * maxmn * eps);
}

/**
 * Helper: compute ||I - U'*U|| / (k*eps) where U is m x k
 */
static f32 compute_orthogonality_UtU(const f32* U, int m, int k, f32* temp)
{
    f32 eps = slamch("E");

    /* Compute U'*U into temp (k x k) */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                k, k, m, 1.0f, U, m, U, m, 0.0f, temp, k);

    /* Compute ||I - U'*U|| */
    for (int i = 0; i < k; i++) {
        temp[i + i * k] -= 1.0f;
    }

    f32 resid = slange("F", k, k, temp, k, NULL);
    return resid / (k * eps);
}

/**
 * Helper: compute ||I - VT*VT'|| / (k*eps) where VT is k x n with leading dim ldvt
 */
static f32 compute_orthogonality_VTVTt(const f32* VT, int k, int n, int ldvt, f32* temp)
{
    f32 eps = slamch("E");

    /* Compute VT*VT' into temp (k x k) */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                k, k, n, 1.0f, VT, ldvt, VT, ldvt, 0.0f, temp, k);

    /* Compute ||I - VT*VT'|| */
    for (int i = 0; i < k; i++) {
        temp[i + i * k] -= 1.0f;
    }

    f32 resid = slange("F", k, k, temp, k, NULL);
    return resid / (k * eps);
}

/* ============== Test Functions ============== */

/**
 * Test workspace query (lwork = -1)
 */
static void test_workspace_query(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    f32 work_query;
    int info;

    /* Query with jobu='A', jobvt='A' */
    sgesvd("A", "A", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0f);

    /* Query with jobu='S', jobvt='S' */
    sgesvd("S", "S", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0f);

    /* Query with jobu='N', jobvt='N' */
    sgesvd("N", "N", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0f);
}

/**
 * Test zero matrix: all singular values should be zero
 */
static void test_zero_matrix(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    /* Zero matrix */
    memset(fix->A, 0, m * n * sizeof(f32));
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    sgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* All singular values should be zero */
    for (int i = 0; i < minmn; i++) {
        assert_true((double)fabsf(fix->S[i]) < 1e-14);
    }
}

/**
 * Test identity-like matrix (m x n diagonal of 1s)
 */
static void test_identity_like(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    /* Identity-like: diag(1,1,...,1) */
    memset(fix->A, 0, m * n * sizeof(f32));
    for (int i = 0; i < minmn; i++) {
        fix->A[i + i * m] = 1.0f;
    }
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    sgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* All singular values should be 1 */
    for (int i = 0; i < minmn; i++) {
        assert_true(fabsf(fix->S[i] - 1.0f) < THRESH * slamch("E"));
    }

    /* Check SVD residual */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                        fix->VT, n, fix->temp);
    assert_residual_below(resid, THRESH);
}

/**
 * Test random well-conditioned matrix
 */
static void test_random_wellcond(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    /* Generate random matrix */
    generate_random_matrix(fix->A, m, n, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    sgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual: ||A - U*S*VT|| */
    f32 resid1 = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                         fix->VT, n, fix->temp);
    assert_residual_below(resid1, THRESH);

    /* Check U orthogonality: ||I - U'*U|| */
    f32 resid2 = compute_orthogonality_UtU(fix->U, m, minmn, fix->temp);
    assert_residual_below(resid2, THRESH);

    /* Check VT orthogonality: ||I - VT*VT'|| */
    f32 resid3 = compute_orthogonality_VTVTt(fix->VT, minmn, n, n, fix->temp);
    assert_residual_below(resid3, THRESH);
}

/**
 * Test with known singular values
 */
static void test_known_sv(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;
    f32 eps = slamch("E");

    /* Specify known singular values */
    f32* sv_known = malloc(minmn * sizeof(f32));
    assert_non_null(sv_known);
    for (int i = 0; i < minmn; i++) {
        sv_known[i] = (f32)(minmn - i);  /* minmn, minmn-1, ..., 1 */
    }

    /* Generate matrix with these singular values */
    generate_matrix_with_sv(fix->A, m, n, sv_known, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    sgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check computed singular values match expected */
    for (int i = 0; i < minmn; i++) {
        f32 relerr = fabsf(fix->S[i] - sv_known[i]) / sv_known[i];
        assert_true(relerr < THRESH * eps * minmn);
    }

    /* Check SVD residual */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                        fix->VT, n, fix->temp);
    assert_residual_below(resid, THRESH);

    free(sv_known);
}

/**
 * Test rank-deficient matrix
 */
static void test_rank_deficient(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int rank = minmn / 2;  /* half rank */
    if (rank < 1) rank = 1;
    int info;

    /* Specify singular values with zeros */
    f32* sv_known = malloc(minmn * sizeof(f32));
    for (int i = 0; i < rank; i++) {
        sv_known[i] = (f32)(rank - i);
    }
    for (int i = rank; i < minmn; i++) {
        sv_known[i] = 0.0f;
    }

    generate_matrix_with_sv(fix->A, m, n, sv_known, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    sgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values are sorted and bottom ones are ~zero */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check that small singular values are indeed small */
    f32 tol = THRESH * slamch("E") * fix->S[0] * minmn;
    for (int i = rank; i < minmn; i++) {
        assert_true(fix->S[i] < tol);
    }

    /* Check SVD residual */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                        fix->VT, n, fix->temp);
    assert_residual_below(resid, THRESH);

    free(sv_known);
}

/**
 * Test jobu='S', jobvt='S' (economy size)
 */
static void test_economy_svd(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    generate_random_matrix(fix->A, m, n, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* Economy SVD: U is m x minmn, VT is minmn x n */
    sgesvd("S", "S", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, minmn, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual (U is m x minmn, VT is minmn x n) */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                        fix->VT, minmn, fix->temp);
    assert_residual_below(resid, THRESH);
}

/**
 * Test jobu='N', jobvt='N' (singular values only)
 */
static void test_sv_only(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    generate_random_matrix(fix->A, m, n, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* Singular values only */
    sgesvd("N", "N", m, n, fix->Acopy, m, fix->S,
           NULL, 1, NULL, 1, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Now compute full SVD to verify values match */
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);
    f32* S2 = malloc(minmn * sizeof(f32));

    sgesvd("A", "A", m, n, fix->Acopy, m, S2,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Compare singular values */
    f32 eps = slamch("E");
    for (int i = 0; i < minmn; i++) {
        f32 diff = fabsf(fix->S[i] - S2[i]);
        f32 denom = (S2[i] > 0) ? S2[i] : 1.0f;
        assert_true(diff / denom < THRESH * eps);
    }

    free(S2);
}

/**
 * Test jobu='O' (overwrite A with U)
 */
static void test_overwrite_U(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    /* This test only makes sense when m >= n */
    if (m < n) {
        skip();
        return;
    }

    generate_random_matrix(fix->A, m, n, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* jobu='O': first minmn columns of U overwrite A */
    sgesvd("O", "A", m, n, fix->Acopy, m, fix->S,
           NULL, 1, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual using Acopy as U */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->Acopy, m, fix->S,
                                        fix->VT, n, fix->temp);
    assert_residual_below(resid, THRESH);
}

/**
 * Test jobvt='O' (overwrite A with VT)
 */
static void test_overwrite_VT(void** state)
{
    dgesvd_fixture_t* fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = (m < n) ? m : n;
    int info;

    /* This test only makes sense when m <= n */
    if (m > n) {
        skip();
        return;
    }

    generate_random_matrix(fix->A, m, n, fix->rng_state);
    slacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* jobvt='O': first minmn rows of VT overwrite A */
    sgesvd("A", "O", m, n, fix->Acopy, m, fix->S,
           fix->U, m, NULL, 1, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual using Acopy as VT (leading dimension is m = lda) */
    f32 resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                        fix->Acopy, m, fix->temp);
    assert_residual_below(resid, THRESH);
}

/* ============== Main ============== */

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Square matrices */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_zero_matrix, setup_5x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_identity_like, setup_5x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_5x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_known_sv, setup_10x10, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_rank_deficient, setup_10x10, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_economy_svd, setup_10x10, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_sv_only, setup_10x10, dgesvd_teardown),

        /* Tall matrices (m > n) */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5x3, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_zero_matrix, setup_5x3, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_identity_like, setup_5x3, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_20x10, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_known_sv, setup_20x10, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_economy_svd, setup_50x30, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_overwrite_U, setup_20x10, dgesvd_teardown),

        /* Wide matrices (m < n) */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_3x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_zero_matrix, setup_3x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_identity_like, setup_3x5, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_10x20, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_known_sv, setup_10x20, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_economy_svd, setup_30x50, dgesvd_teardown),
        cmocka_unit_test_setup_teardown(test_overwrite_VT, setup_10x20, dgesvd_teardown),
    };

    return cmocka_run_group_tests_name("dgesvd", tests, NULL, NULL);
}
