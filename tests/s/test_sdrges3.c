/**
 * @file test_sdrges3.c
 * @brief Generalized nonsymmetric Schur form driver test - port of
 *        LAPACK TESTING/EIG/ddrges3.f
 *
 * Tests the nonsymmetric generalized eigenvalue (Schur form) problem
 * driver SGGES3.
 *
 * SGGES3 factors A and B as Q S Z' and Q T Z', where ' means
 * transpose, T is upper triangular, S is in generalized Schur form
 * (block upper triangular, with 1x1 and 2x2 blocks on the diagonal,
 * the 2x2 blocks corresponding to complex conjugate pairs of
 * generalized eigenvalues), and Q and Z are orthogonal. It also
 * computes the generalized eigenvalues (alpha(j),beta(j)), j=1,...,n.
 * Optionally it also reorders the eigenvalues so that a selected
 * cluster of eigenvalues appears in the leading diagonal block of the
 * Schur forms.
 *
 * Test ratios (13 total):
 * Without ordering:
 *   (1)  | A - Q S Z' | / ( |A| n ulp )
 *   (2)  | B - Q T Z' | / ( |B| n ulp )
 *   (3)  | I - QQ' | / ( n ulp )
 *   (4)  | I - ZZ' | / ( n ulp )
 *   (5)  A is in Schur form S
 *   (6)  difference between (alpha,beta) and diagonals of (S,T)
 * With ordering:
 *   (7)  | (A,B) - Q (S,T) Z' | / ( |(A,B)| n ulp )
 *   (8)  | I - QQ' | / ( n ulp )
 *   (9)  | I - ZZ' | / ( n ulp )
 *   (10) A is in Schur form S
 *   (11) difference between (alpha,beta) and diagonals of (S,T)
 *   (12) SDIM is the correct number of selected eigenvalues
 *
 * Matrix types: 26 types generated via SLATM4 with optional
 * random orthogonal transforms (SLARFG + SORM2R).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 20.0f
#define MAXTYP 26

/* Test dimensions from dgg.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 16};
#define NSIZES ((int)(sizeof(NVAL) / sizeof(NVAL[0])))
#define NMAX 16   /* max(NVAL) */

/* External declarations */
extern void sgges3(const char* jobvsl, const char* jobvsr, const char* sort,
                   int (*selctg)(const f32*, const f32*, const f32*),
                   const int n, f32* A, const int lda,
                   f32* B, const int ldb, int* sdim,
                   f32* alphar, f32* alphai, f32* beta,
                   f32* VSL, const int ldvsl, f32* VSR, const int ldvsr,
                   f32* work, const int lwork, int* bwork, int* info);
extern f32  slamch(const char* cmach);
extern f32  slange(const char* norm, const int m, const int n,
                   const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta_val, f32* A, const int lda);
extern void slarfg(const int n, f32* alpha, f32* x, const int incx, f32* tau);
extern void sorm2r(const char* side, const char* trans, const int m,
                   const int n, const int k, const f32* A, const int lda,
                   const f32* tau, f32* C, const int ldc, f32* work, int* info);

/**
 * DLCTES returns true (1) if the eigenvalue (ZR/D) + sqrt(-1)*(ZI/D)
 * is to be selected (specifically, if the real part of the eigenvalue
 * is negative), and otherwise returns false (0).
 *
 * Port of LAPACK TESTING/EIG/dlctes.f
 */
static int dlctes(const f32* zr, const f32* zi, const f32* d)
{
    (void)zi;
    if (*d == 0.0f) {
        return (*zr < 0.0f);
    } else {
        return (copysignf(1.0f, *zr) != copysignf(1.0f, *d));
    }
}

/* Parameters for each test case */
typedef struct {
    int jsize;   /* index into NVAL[] */
    int jtype;   /* matrix type 1..26 */
    char name[64];
} ddrges3_params_t;

/* Shared workspace */
typedef struct {
    f32* A;
    f32* B;
    f32* S;
    f32* T;
    f32* Q;
    f32* Z;
    f32* alphar;
    f32* alphai;
    f32* beta;
    f32* work;
    int* bwork;
    int lwork;
} ddrges3_workspace_t;

static ddrges3_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_ws = calloc(1, sizeof(ddrges3_workspace_t));
    if (!g_ws) return -1;

    const int lda = NMAX;
    const int n2 = lda * NMAX;

    g_ws->A      = malloc(n2 * sizeof(f32));
    g_ws->B      = malloc(n2 * sizeof(f32));
    g_ws->S      = malloc(n2 * sizeof(f32));
    g_ws->T      = malloc(n2 * sizeof(f32));
    g_ws->Q      = malloc(n2 * sizeof(f32));
    g_ws->Z      = malloc(n2 * sizeof(f32));
    g_ws->alphar = malloc(NMAX * sizeof(f32));
    g_ws->alphai = malloc(NMAX * sizeof(f32));
    g_ws->beta   = malloc(NMAX * sizeof(f32));
    g_ws->bwork  = malloc(NMAX * sizeof(int));

    int minwrk = 10 * (NMAX + 1);
    int tmp = 3 * NMAX * NMAX;
    if (tmp > minwrk) minwrk = tmp;
    g_ws->lwork = minwrk + NMAX * NMAX;
    g_ws->work = malloc(g_ws->lwork * sizeof(f32));

    if (!g_ws->A || !g_ws->B || !g_ws->S || !g_ws->T ||
        !g_ws->Q || !g_ws->Z || !g_ws->alphar || !g_ws->alphai ||
        !g_ws->beta || !g_ws->bwork || !g_ws->work) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->S);
        free(g_ws->T);
        free(g_ws->Q);
        free(g_ws->Z);
        free(g_ws->alphar);
        free(g_ws->alphai);
        free(g_ws->beta);
        free(g_ws->bwork);
        free(g_ws->work);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* DATA arrays from ddrges3.f (converted to 0-based) */
static const int kclass[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3
};
static const int kz1[6]  = {0, 1, 2, 1, 3, 3};
static const int kz2[6]  = {0, 0, 1, 2, 1, 1};
static const int kadd[6] = {0, 0, 0, 0, 3, 2};
static const int katype[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4,
    4, 4, 2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
};
static const int kbtype[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4,
    1, 1, -4, 2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
};
static const int kazero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 1, 3,
    5, 5, 5, 5, 3, 3, 3, 3, 1
};
static const int kbzero[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4, 1, 4,
    6, 6, 6, 6, 4, 4, 4, 4, 1
};
static const int kamagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    2, 3, 3, 2, 1
};
static const int kbmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1,
    3, 2, 3, 2, 1
};
static const int ktrian[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static const int iasign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0,
    2, 2, 2, 2, 2, 0
};
static const int ibsign[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0
};

static void test_ddrges3(void** state)
{
    ddrges3_params_t* params = (ddrges3_params_t*)(*state);
    const int n = NVAL[params->jsize];
    const int jtype = params->jtype;  /* 1-based */
    const int lda = NMAX;
    const int ldq = NMAX;

    /* Quick return for n=0 */
    if (n == 0) return;

    const f32 safmin = slamch("S");
    const f32 ulp = slamch("P");
    const f32 safmax = 1.0f / safmin;
    const f32 ulpinv = 1.0f / ulp;

    /* RMAGN(0:3) */
    const int n1 = (n > 1) ? n : 1;
    f32 rmagn[4];
    rmagn[0] = 0.0f;
    rmagn[1] = 1.0f;
    rmagn[2] = safmax * ulp / (f32)n1;
    rmagn[3] = safmin * ulpinv * (f32)n1;

    /* RNG state (seeded from size and type for reproducibility) */
    uint64_t rng_state[4];
    rng_seed(rng_state, (uint64_t)(params->jsize * 1000 + jtype));

    f32 result[13];
    for (int i = 0; i < 13; i++) result[i] = 0.0f;

    int iinfo = 0;
    int jt = jtype - 1;  /* 0-based index into DATA arrays */

    /* Generate test matrices A and B */
    if (kclass[jt] < 3) {

        /* Generate A (w/o rotation) */
        int in = n;
        if (abs(katype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, g_ws->A, lda);
        }
        slatm4(katype[jt], in, kz1[kazero[jt] - 1],
                kz2[kazero[jt] - 1], iasign[jt],
                rmagn[kamagn[jt]], ulp,
                rmagn[ktrian[jt] * kamagn[jt]], 2,
                g_ws->A, lda, rng_state);
        int iadd = kadd[kazero[jt] - 1];
        if (iadd > 0 && iadd <= n)
            g_ws->A[(iadd - 1) + (iadd - 1) * lda] = 1.0f;

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(kbtype[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                slaset("Full", n, n, 0.0f, 0.0f, g_ws->B, lda);
        }
        slatm4(kbtype[jt], in, kz1[kbzero[jt] - 1],
                kz2[kbzero[jt] - 1], ibsign[jt],
                rmagn[kbmagn[jt]], 1.0f,
                rmagn[ktrian[jt] * kbmagn[jt]], 2,
                g_ws->B, lda, rng_state);
        iadd = kadd[kbzero[jt] - 1];
        if (iadd != 0 && iadd <= n)
            g_ws->B[(iadd - 1) + (iadd - 1) * lda] = 1.0f;

        if (kclass[jt] == 2 && n > 0) {
            /*
             * Include rotations
             *
             * Generate Q, Z as Householder transformations times
             * a diagonal matrix.
             */
            for (int jc = 0; jc < n - 1; jc++) {
                for (int jr = jc; jr < n; jr++) {
                    g_ws->Q[jr + jc * ldq] = rng_dist_f32(rng_state, 3);
                    g_ws->Z[jr + jc * ldq] = rng_dist_f32(rng_state, 3);
                }
                slarfg(n - jc, &g_ws->Q[jc + jc * ldq],
                       &g_ws->Q[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[jc]);
                g_ws->work[2 * n + jc] = copysignf(1.0f, g_ws->Q[jc + jc * ldq]);
                g_ws->Q[jc + jc * ldq] = 1.0f;
                slarfg(n - jc, &g_ws->Z[jc + jc * ldq],
                       &g_ws->Z[(jc + 1) + jc * ldq], 1,
                       &g_ws->work[n + jc]);
                g_ws->work[3 * n + jc] = copysignf(1.0f, g_ws->Z[jc + jc * ldq]);
                g_ws->Z[jc + jc * ldq] = 1.0f;
            }
            g_ws->Q[(n - 1) + (n - 1) * ldq] = 1.0f;
            g_ws->work[n - 1] = 0.0f;
            g_ws->work[3 * n - 1] = copysignf(1.0f, rng_dist_f32(rng_state, 2));
            g_ws->Z[(n - 1) + (n - 1) * ldq] = 1.0f;
            g_ws->work[2 * n - 1] = 0.0f;
            g_ws->work[4 * n - 1] = copysignf(1.0f, rng_dist_f32(rng_state, 2));

            /* Apply the diagonal matrices */
            for (int jc = 0; jc < n; jc++) {
                for (int jr = 0; jr < n; jr++) {
                    g_ws->A[jr + jc * lda] = g_ws->work[2 * n + jr] *
                                              g_ws->work[3 * n + jc] *
                                              g_ws->A[jr + jc * lda];
                    g_ws->B[jr + jc * lda] = g_ws->work[2 * n + jr] *
                                              g_ws->work[3 * n + jc] *
                                              g_ws->B[jr + jc * lda];
                }
            }
            sorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->A, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("L", "N", n, n, n - 1, g_ws->Q, ldq, g_ws->work,
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
            sorm2r("R", "T", n, n, n - 1, g_ws->Z, ldq, &g_ws->work[n],
                   g_ws->B, lda, &g_ws->work[2 * n], &iinfo);
            if (iinfo != 0) goto gen_error;
        }
    } else {
        /* Random matrices (class 3, type 26) */
        for (int jc = 0; jc < n; jc++) {
            for (int jr = 0; jr < n; jr++) {
                g_ws->A[jr + jc * lda] = rmagn[kamagn[jt]] *
                                          rng_dist_f32(rng_state, 2);
                g_ws->B[jr + jc * lda] = rmagn[kbmagn[jt]] *
                                          rng_dist_f32(rng_state, 2);
            }
        }
    }

    goto gen_ok;
gen_error:
    print_message("Generator returned INFO=%d for N=%d JTYPE=%d\n",
                  iinfo, n, jtype);
    assert_int_equal(iinfo, 0);
    return;
gen_ok:

    for (int i = 0; i < 13; i++) result[i] = -1.0f;

    /* Test with and without sorting of eigenvalues */
    for (int isort = 0; isort <= 1; isort++) {
        const char* sort;
        int rsub;
        if (isort == 0) {
            sort = "N";
            rsub = 0;
        } else {
            sort = "S";
            rsub = 5;
        }

        /* Copy A and B into S and T */
        slacpy("Full", n, n, g_ws->A, lda, g_ws->S, lda);
        slacpy("Full", n, n, g_ws->B, lda, g_ws->T, lda);
        result[rsub + isort] = ulpinv;
        int sdim;
        sgges3("V", "V", sort, dlctes, n, g_ws->S, lda, g_ws->T, lda,
               &sdim, g_ws->alphar, g_ws->alphai, g_ws->beta,
               g_ws->Q, ldq, g_ws->Z, ldq,
               g_ws->work, g_ws->lwork, g_ws->bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[rsub + isort] = ulpinv;
            print_message("SGGES3 returned INFO=%d for N=%d JTYPE=%d SORT=%s\n",
                          iinfo, n, jtype, sort);
            goto check_results;
        }

        /* Do tests 1--4 (or tests 7--9 when reordering) */
        if (isort == 0) {
            sget51(1, n, g_ws->A, lda, g_ws->S, lda, g_ws->Q, ldq,
                   g_ws->Z, ldq, g_ws->work, &result[0]);
            sget51(1, n, g_ws->B, lda, g_ws->T, lda, g_ws->Q, ldq,
                   g_ws->Z, ldq, g_ws->work, &result[1]);
        } else {
            sget54(n, g_ws->A, lda, g_ws->B, lda, g_ws->S, lda,
                   g_ws->T, lda, g_ws->Q, ldq, g_ws->Z, ldq,
                   g_ws->work, &result[6]);
        }
        sget51(3, n, g_ws->A, lda, g_ws->T, lda, g_ws->Q, ldq,
               g_ws->Q, ldq, g_ws->work, &result[2 + rsub]);
        sget51(3, n, g_ws->B, lda, g_ws->T, lda, g_ws->Z, ldq,
               g_ws->Z, ldq, g_ws->work, &result[3 + rsub]);

        /*
         * Do test 5 and 6 (or Tests 10 and 11 when reordering):
         * check Schur form of A and compare eigenvalues with
         * diagonals.
         */
        f32 temp1 = 0.0f;

        for (int j = 0; j < n; j++) {
            int ilabad = 0;
            f32 temp2;
            if (g_ws->alphai[j] == 0.0f) {
                temp2 = (fabsf(g_ws->alphar[j] - g_ws->S[j + j * lda]) /
                         fmaxf(safmin, fmaxf(fabsf(g_ws->alphar[j]),
                              fabsf(g_ws->S[j + j * lda]))) +
                         fabsf(g_ws->beta[j] - g_ws->T[j + j * lda]) /
                         fmaxf(safmin, fmaxf(fabsf(g_ws->beta[j]),
                              fabsf(g_ws->T[j + j * lda])))) / ulp;

                if (j < n - 1) {
                    if (g_ws->S[(j + 1) + j * lda] != 0.0f) {
                        ilabad = 1;
                        result[4 + rsub] = ulpinv;
                    }
                }
                if (j > 0) {
                    if (g_ws->S[j + (j - 1) * lda] != 0.0f) {
                        ilabad = 1;
                        result[4 + rsub] = ulpinv;
                    }
                }

            } else {
                int i1;
                if (g_ws->alphai[j] > 0.0f) {
                    i1 = j;
                } else {
                    i1 = j - 1;
                }
                if (i1 < 0 || i1 >= n) {
                    ilabad = 1;
                } else if (i1 < n - 2) {
                    if (g_ws->S[(i1 + 2) + (i1 + 1) * lda] != 0.0f) {
                        ilabad = 1;
                        result[4 + rsub] = ulpinv;
                    }
                } else if (i1 > 0) {
                    if (g_ws->S[i1 + (i1 - 1) * lda] != 0.0f) {
                        ilabad = 1;
                        result[4 + rsub] = ulpinv;
                    }
                }
                if (!ilabad) {
                    int ierr;
                    sget53(&g_ws->S[i1 + i1 * lda], lda,
                           &g_ws->T[i1 + i1 * lda], lda,
                           g_ws->beta[j], g_ws->alphar[j],
                           g_ws->alphai[j], &temp2, &ierr);
                    if (ierr >= 3) {
                        print_message("SGET53 returned INFO=%d for eigenvalue "
                                      "%d, N=%d JTYPE=%d\n",
                                      ierr, j + 1, n, jtype);
                    }
                } else {
                    temp2 = ulpinv;
                }
            }
            temp1 = fmaxf(temp1, temp2);
            if (ilabad) {
                print_message("S not in Schur form at eigenvalue %d, "
                              "N=%d JTYPE=%d\n", j + 1, n, jtype);
            }
        }
        result[5 + rsub] = temp1;

        if (isort >= 1) {
            /* Do test 12 */
            result[11] = 0.0f;
            int knteig = 0;
            for (int i = 0; i < n; i++) {
                if (dlctes(&g_ws->alphar[i], &g_ws->alphai[i],
                           &g_ws->beta[i]) ||
                    dlctes(&g_ws->alphar[i],
                           &(f32){-g_ws->alphai[i]},
                           &g_ws->beta[i])) {
                    knteig++;
                }
                if (i < n - 1) {
                    f32 neg_alphai_next = -g_ws->alphai[i + 1];
                    if ((dlctes(&g_ws->alphar[i + 1], &g_ws->alphai[i + 1],
                                &g_ws->beta[i + 1]) ||
                         dlctes(&g_ws->alphar[i + 1], &neg_alphai_next,
                                &g_ws->beta[i + 1])) &&
                        (!(dlctes(&g_ws->alphar[i], &g_ws->alphai[i],
                                  &g_ws->beta[i]) ||
                           dlctes(&g_ws->alphar[i],
                                  &(f32){-g_ws->alphai[i]},
                                  &g_ws->beta[i]))) &&
                        iinfo != n + 2) {
                        result[11] = ulpinv;
                    }
                }
            }
            if (sdim != knteig) {
                result[11] = ulpinv;
            }
        }
    }

check_results:
    ;
    /* Check results against threshold */
    int any_fail = 0;
    for (int jr = 0; jr < 13; jr++) {
        if (result[jr] >= THRESH) {
            print_message("N=%d JTYPE=%d test(%d)=%g\n",
                          n, jtype, jr + 1, result[jr]);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

int main(void)
{
    /* Total: NSIZES * MAXTYP test cases (7 * 26 = 182) */
    static ddrges3_params_t all_params[NSIZES * MAXTYP];
    static struct CMUnitTest all_tests[NSIZES * MAXTYP];
    int idx = 0;

    for (int js = 0; js < NSIZES; js++) {
        for (int jt = 1; jt <= MAXTYP; jt++) {
            ddrges3_params_t* p = &all_params[idx];
            p->jsize = js;
            p->jtype = jt;
            snprintf(p->name, sizeof(p->name),
                     "N=%d_type=%d", NVAL[js], jt);

            all_tests[idx].name = p->name;
            all_tests[idx].test_func = test_ddrges3;
            all_tests[idx].setup_func = NULL;
            all_tests[idx].teardown_func = NULL;
            all_tests[idx].initial_state = p;
            idx++;
        }
    }

    return cmocka_run_group_tests_name("ddrges3", all_tests, group_setup,
                                       group_teardown);
}
