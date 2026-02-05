/**
 * @file test_dgesvd.c
 * @brief CMocka test suite for dgesvd (Singular Value Decomposition).
 *
 * Tests the SVD routine dgesvd using LAPACK's verification methodology
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
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold */
#define THRESH 30.0

/* Routine under test */
extern void dgesvd(const char* jobu, const char* jobvt,
                   const int m, const int n,
                   double* A, const int lda,
                   double* S,
                   double* U, const int ldu,
                   double* VT, const int ldvt,
                   double* work, const int lwork,
                   int* info);

/* Utilities */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);

/* ---- Simple xoshiro256+ RNG ---- */
static uint64_t rng_state[4];

static void rng_seed(uint64_t s)
{
    for (int i = 0; i < 4; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state[i] = s;
    }
}

static double rng_uniform(void)
{
    uint64_t s = rng_state[1] * 5;
    uint64_t r = ((s << 7) | (s >> 57)) * 9;
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 45) | (rng_state[3] >> 19);
    return (double)(r >> 11) * 0x1.0p-53;
}

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int m, n;
    double* A;          /* original matrix (preserved) */
    double* Acopy;      /* working copy (overwritten by dgesvd) */
    double* S;          /* singular values */
    double* U;          /* left singular vectors */
    double* VT;         /* right singular vectors (transposed) */
    double* work;       /* workspace */
    int lwork;          /* workspace size */
    double* temp;       /* temporary workspace for verification */
    uint64_t seed;
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
    fix->seed = g_seed++;

    int minmn = (m < n) ? m : n;
    int maxmn = (m > n) ? m : n;

    /* Query optimal workspace size */
    double work_query;
    int info;
    dgesvd("A", "A", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    fix->lwork = (int)work_query;
    if (fix->lwork < 1) fix->lwork = 5 * maxmn;

    /* Allocate arrays */
    fix->A = malloc(m * n * sizeof(double));
    fix->Acopy = malloc(m * n * sizeof(double));
    fix->S = malloc(minmn * sizeof(double));
    fix->U = malloc(m * m * sizeof(double));
    fix->VT = malloc(n * n * sizeof(double));
    fix->work = malloc(fix->lwork * sizeof(double));
    fix->temp = malloc(maxmn * maxmn * sizeof(double));

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
static void generate_random_matrix(double* A, int m, int n)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * m] = 2.0 * rng_uniform() - 1.0;
        }
    }
}

/**
 * Helper: generate random matrix with specified singular values
 */
static void generate_matrix_with_sv(double* A, int m, int n, const double* sv)
{
    int minmn = (m < n) ? m : n;

    /* Start with diagonal of singular values */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (i == j && i < minmn) {
                A[i + j * m] = sv[i];
            } else {
                A[i + j * m] = 0.0;
            }
        }
    }

    /* Apply random orthogonal transformations from left and right */
    /* For simplicity, just use Householder reflections */
    double* v = malloc(((m > n) ? m : n) * sizeof(double));

    /* Left transformations */
    for (int k = 0; k < minmn; k++) {
        /* Random Householder vector */
        double nrm = 0.0;
        for (int i = 0; i < m; i++) {
            v[i] = 2.0 * rng_uniform() - 1.0;
            nrm += v[i] * v[i];
        }
        nrm = sqrt(nrm);
        if (nrm > 0) {
            for (int i = 0; i < m; i++) v[i] /= nrm;
        }

        /* Apply H = I - 2*v*v' to A from left */
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++) {
                dot += v[i] * A[i + j * m];
            }
            for (int i = 0; i < m; i++) {
                A[i + j * m] -= 2.0 * v[i] * dot;
            }
        }
    }

    /* Right transformations */
    for (int k = 0; k < minmn; k++) {
        /* Random Householder vector */
        double nrm = 0.0;
        for (int j = 0; j < n; j++) {
            v[j] = 2.0 * rng_uniform() - 1.0;
            nrm += v[j] * v[j];
        }
        nrm = sqrt(nrm);
        if (nrm > 0) {
            for (int j = 0; j < n; j++) v[j] /= nrm;
        }

        /* Apply H = I - 2*v*v' to A from right */
        for (int i = 0; i < m; i++) {
            double dot = 0.0;
            for (int j = 0; j < n; j++) {
                dot += A[i + j * m] * v[j];
            }
            for (int j = 0; j < n; j++) {
                A[i + j * m] -= 2.0 * dot * v[j];
            }
        }
    }

    free(v);
}

/**
 * Helper: check singular values are non-negative and sorted descending
 */
static int check_sv_sorted(const double* S, int n)
{
    for (int i = 0; i < n; i++) {
        if (S[i] < 0.0) return 0;  /* negative singular value */
    }
    for (int i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1]) return 0;  /* not descending */
    }
    return 1;
}

/**
 * Helper: compute ||A - U*diag(S)*VT|| / (||A||*max(m,n)*eps)
 */
static double compute_svd_residual(const double* A, int m, int n,
                                   const double* U, int ldu,
                                   const double* S,
                                   const double* VT, int ldvt,
                                   double* temp)
{
    double eps = dlamch("E");
    int minmn = (m < n) ? m : n;
    int maxmn = (m > n) ? m : n;

    /* Compute ||A|| */
    double anrm = dlange("F", m, n, A, m, NULL);
    if (anrm == 0.0) anrm = 1.0;

    /* Compute U*diag(S)*VT into temp */
    /* First: temp2 = diag(S)*VT (scale rows of VT by S) */
    double* temp2 = malloc(minmn * n * sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < minmn; i++) {
            temp2[i + j * minmn] = S[i] * VT[i + j * ldvt];
        }
    }

    /* Then: temp = U * temp2 */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, minmn, 1.0, U, ldu, temp2, minmn, 0.0, temp, m);

    free(temp2);

    /* Compute temp = A - temp */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            temp[i + j * m] -= A[i + j * m];
        }
    }

    /* Compute ||A - U*S*VT|| */
    double resid = dlange("F", m, n, temp, m, NULL);

    return resid / (anrm * maxmn * eps);
}

/**
 * Helper: compute ||I - U'*U|| / (k*eps) where U is m x k
 */
static double compute_orthogonality_UtU(const double* U, int m, int k, double* temp)
{
    double eps = dlamch("E");

    /* Compute U'*U into temp (k x k) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                k, k, m, 1.0, U, m, U, m, 0.0, temp, k);

    /* Compute ||I - U'*U|| */
    for (int i = 0; i < k; i++) {
        temp[i + i * k] -= 1.0;
    }

    double resid = dlange("F", k, k, temp, k, NULL);
    return resid / (k * eps);
}

/**
 * Helper: compute ||I - VT*VT'|| / (k*eps) where VT is k x n with leading dim ldvt
 */
static double compute_orthogonality_VTVTt(const double* VT, int k, int n, int ldvt, double* temp)
{
    double eps = dlamch("E");

    /* Compute VT*VT' into temp (k x k) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                k, k, n, 1.0, VT, ldvt, VT, ldvt, 0.0, temp, k);

    /* Compute ||I - VT*VT'|| */
    for (int i = 0; i < k; i++) {
        temp[i + i * k] -= 1.0;
    }

    double resid = dlange("F", k, k, temp, k, NULL);
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
    double work_query;
    int info;

    /* Query with jobu='A', jobvt='A' */
    dgesvd("A", "A", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0);

    /* Query with jobu='S', jobvt='S' */
    dgesvd("S", "S", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0);

    /* Query with jobu='N', jobvt='N' */
    dgesvd("N", "N", m, n, NULL, m, NULL, NULL, m, NULL, n,
           &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0);
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

    rng_seed(fix->seed);

    /* Zero matrix */
    memset(fix->A, 0, m * n * sizeof(double));
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    dgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* All singular values should be zero */
    for (int i = 0; i < minmn; i++) {
        assert_true(fabs(fix->S[i]) < 1e-14);
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

    rng_seed(fix->seed);

    /* Identity-like: diag(1,1,...,1) */
    memset(fix->A, 0, m * n * sizeof(double));
    for (int i = 0; i < minmn; i++) {
        fix->A[i + i * m] = 1.0;
    }
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    dgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* All singular values should be 1 */
    for (int i = 0; i < minmn; i++) {
        assert_true(fabs(fix->S[i] - 1.0) < THRESH * dlamch("E"));
    }

    /* Check SVD residual */
    double resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
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

    rng_seed(fix->seed);

    /* Generate random matrix */
    generate_random_matrix(fix->A, m, n);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    dgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual: ||A - U*S*VT|| */
    double resid1 = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
                                         fix->VT, n, fix->temp);
    assert_residual_below(resid1, THRESH);

    /* Check U orthogonality: ||I - U'*U|| */
    double resid2 = compute_orthogonality_UtU(fix->U, m, minmn, fix->temp);
    assert_residual_below(resid2, THRESH);

    /* Check VT orthogonality: ||I - VT*VT'|| */
    double resid3 = compute_orthogonality_VTVTt(fix->VT, minmn, n, n, fix->temp);
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
    double eps = dlamch("E");

    rng_seed(fix->seed);

    /* Specify known singular values */
    double* sv_known = malloc(minmn * sizeof(double));
    assert_non_null(sv_known);
    for (int i = 0; i < minmn; i++) {
        sv_known[i] = (double)(minmn - i);  /* minmn, minmn-1, ..., 1 */
    }

    /* Generate matrix with these singular values */
    generate_matrix_with_sv(fix->A, m, n, sv_known);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    dgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check computed singular values match expected */
    for (int i = 0; i < minmn; i++) {
        double relerr = fabs(fix->S[i] - sv_known[i]) / sv_known[i];
        assert_true(relerr < THRESH * eps * minmn);
    }

    /* Check SVD residual */
    double resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
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

    rng_seed(fix->seed);

    /* Specify singular values with zeros */
    double* sv_known = malloc(minmn * sizeof(double));
    for (int i = 0; i < rank; i++) {
        sv_known[i] = (double)(rank - i);
    }
    for (int i = rank; i < minmn; i++) {
        sv_known[i] = 0.0;
    }

    generate_matrix_with_sv(fix->A, m, n, sv_known);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    dgesvd("A", "A", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values are sorted and bottom ones are ~zero */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check that small singular values are indeed small */
    double tol = THRESH * dlamch("E") * fix->S[0] * minmn;
    for (int i = rank; i < minmn; i++) {
        assert_true(fix->S[i] < tol);
    }

    /* Check SVD residual */
    double resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
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

    rng_seed(fix->seed);

    generate_random_matrix(fix->A, m, n);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* Economy SVD: U is m x minmn, VT is minmn x n */
    dgesvd("S", "S", m, n, fix->Acopy, m, fix->S,
           fix->U, m, fix->VT, minmn, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual (U is m x minmn, VT is minmn x n) */
    double resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
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

    rng_seed(fix->seed);

    generate_random_matrix(fix->A, m, n);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* Singular values only */
    dgesvd("N", "N", m, n, fix->Acopy, m, fix->S,
           NULL, 1, NULL, 1, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Now compute full SVD to verify values match */
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);
    double* S2 = malloc(minmn * sizeof(double));

    dgesvd("A", "A", m, n, fix->Acopy, m, S2,
           fix->U, m, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Compare singular values */
    double eps = dlamch("E");
    for (int i = 0; i < minmn; i++) {
        double diff = fabs(fix->S[i] - S2[i]);
        double denom = (S2[i] > 0) ? S2[i] : 1.0;
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

    rng_seed(fix->seed);

    generate_random_matrix(fix->A, m, n);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* jobu='O': first minmn columns of U overwrite A */
    dgesvd("O", "A", m, n, fix->Acopy, m, fix->S,
           NULL, 1, fix->VT, n, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual using Acopy as U */
    double resid = compute_svd_residual(fix->A, m, n, fix->Acopy, m, fix->S,
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

    rng_seed(fix->seed);

    generate_random_matrix(fix->A, m, n);
    dlacpy("A", m, n, fix->A, m, fix->Acopy, m);

    /* jobvt='O': first minmn rows of VT overwrite A */
    dgesvd("A", "O", m, n, fix->Acopy, m, fix->S,
           fix->U, m, NULL, 1, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Check singular values sorted */
    assert_true(check_sv_sorted(fix->S, minmn));

    /* Check SVD residual using Acopy as VT (leading dimension is m = lda) */
    double resid = compute_svd_residual(fix->A, m, n, fix->U, m, fix->S,
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
