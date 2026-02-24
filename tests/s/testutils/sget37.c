/**
 * @file sget37.c
 * @brief SGET37 tests STRSNA, a routine for estimating condition numbers of
 *        eigenvalues and/or right eigenvectors of a matrix.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define LDT   20
#define LWORK (2 * LDT * (10 + LDT))

#define NCASES37 39

/*
 * Embedded test data from LAPACK dec.in lines 121-636.
 * 39 test cases. Format per case:
 *   n*n matrix values (row-major), then n*(wr,wi,s,sep) quadruplets.
 */
static const f32 dget37_data[] = {
    /* Case 0: N=1 */
    0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 1: N=1 */
    1.0f,
    1.0f, 0.0f, 1.0f, 1.0f,

    /* Case 2: N=2 */
    0.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,

    /* Case 3: N=2 */
    3.0f, 2.0f,
    2.0f, 3.0f,
    1.0f, 0.0f, 1.0f, 4.0f,
    5.0f, 0.0f, 1.0f, 4.0f,

    /* Case 4: N=2 */
    3.0f, -2.0f,
    2.0f, 3.0f,
    3.0f, 2.0f, 1.0f, 4.0f,
    3.0f, -2.0f, 1.0f, 4.0f,

    /* Case 5: N=6 */
    1.0000e-07f, -1.0000e-07f, 1.0f, 1.1000f, 2.3000f, 3.7000f,
    3.0000e-07f, 1.0000e-07f, 1.0f, 1.0f, -1.3000f, -7.7000f,
    0.0f, 0.0f, 3.0000e-07f, 1.0000e-07f, 2.2000f, 3.3000f,
    0.0f, 0.0f, -1.0000e-07f, 3.0000e-07f, 1.8000f, 1.6000f,
    0.0f, 0.0f, 0.0f, 0.0f, 4.0000e-06f, 5.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0000e-06f,
    -3.8730f, 0.0f, 0.6986f, 2.2823f,
    1.0000e-07f, 1.7321e-07f, 9.7611e-08f, 5.0060e-14f,
    1.0000e-07f, -1.7321e-07f, 9.7611e-08f, 5.0060e-14f,
    3.0000e-07f, 1.0000e-07f, 1.0000e-07f, 9.4094e-14f,
    3.0000e-07f, -1.0000e-07f, 1.0000e-07f, 9.4094e-14f,
    3.8730f, 0.0f, 0.4066f, 1.5283f,

    /* Case 6: N=4 */
    7.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 5.0f, -3.0f,
    1.0f, -1.0f, 3.0f, 3.0f,
    3.9603f, 0.0404f, 1.1244e-05f, 3.1179e-05f,
    3.9603f, -0.0404f, 1.1244e-05f, 3.1179e-05f,
    4.0397f, 0.0389f, 1.0807e-05f, 2.9981e-05f,
    4.0397f, -0.0389f, 1.0807e-05f, 2.9981e-05f,

    /* Case 7: N=5 */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.9722e-31f,
    0.0f, 0.0f, 1.0f, 1.9722e-31f,
    0.0f, 0.0f, 1.0f, 1.9722e-31f,
    0.0f, 0.0f, 1.0f, 1.9722e-31f,
    0.0f, 0.0f, 1.0f, 1.9722e-31f,

    /* Case 8: N=5 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f, 1.9722e-31f,
    1.0f, 0.0f, 1.0f, 1.9722e-31f,
    1.0f, 0.0f, 1.0f, 1.9722e-31f,
    1.0f, 0.0f, 1.0f, 1.9722e-31f,
    1.0f, 0.0f, 1.0f, 1.9722e-31f,

    /* Case 9: N=6 */
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,

    /* Case 10: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,
    1.0f, 0.0f, 2.4074e-35f, 2.4074e-35f,

    /* Case 11: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f,
    1.0f, 0.0f, 1.0f, 1.0f,
    2.0f, 0.0f, 1.0f, 1.0f,
    3.0f, 0.0f, 1.0f, 1.0f,
    4.0f, 0.0f, 1.0f, 1.0f,
    5.0f, 0.0f, 1.0f, 1.0f,
    6.0f, 0.0f, 1.0f, 1.0f,

    /* Case 12: N=4 */
    0.9448f, 0.6767f, 0.6908f, 0.5965f,
    0.5876f, 0.8642f, 0.6769f, 0.0726f,
    0.7256f, 0.1943f, 0.9687f, 0.2831f,
    0.2849f, 0.0580f, 0.4845f, 0.7361f,
    0.2433f, 0.2141f, 0.8710f, 0.3507f,
    0.2433f, -0.2141f, 0.8710f, 0.3507f,
    0.7409f, 0.0f, 0.9819f, 0.4699f,
    2.2864f, 0.0f, 0.9772f, 1.5455f,

    /* Case 13: N=6 */
    0.5041f, 0.6652f, 0.7719f, 0.6387f, 0.5955f, 0.6131f,
    0.1574f, 0.3734f, 0.5984f, 0.1547f, 0.9427f, 0.0659f,
    0.4417f, 0.0723f, 0.1544f, 0.5492f, 0.008700f, 0.3004f,
    0.2008f, 0.6080f, 0.3034f, 0.8439f, 0.2390f, 0.5768f,
    0.9361f, 0.7413f, 0.1444f, 0.1786f, 0.1428f, 0.7263f,
    0.5599f, 0.9336f, 0.0780f, 0.4093f, 0.6714f, 0.5617f,
    -0.5228f, 0.0f, 0.2789f, 0.1179f,
    -0.3538f, 0.0f, 0.3543f, 0.0689f,
    -0.008088f, 0.0f, 0.3456f, 0.1349f,
    0.3476f, 0.3053f, 0.5466f, 0.1773f,
    0.3476f, -0.3053f, 0.5466f, 0.1773f,
    2.7698f, 0.0f, 0.9664f, 1.8270f,

    /* Case 14: N=5 */
    0.002000f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.001000f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.001000f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, -0.002000f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    -0.002000f, 0.0f, 2.4000e-11f, 2.3952e-11f,
    -0.001000f, 0.0f, 6.0000e-12f, 5.9940e-12f,
    0.0f, 0.0f, 4.0000e-12f, 3.9920e-12f,
    0.001000f, 0.0f, 6.0000e-12f, 5.9940e-12f,
    0.002000f, 0.0f, 2.4000e-11f, 2.3952e-11f,

    /* Case 15: N=10 */
    0.4863f, 0.9126f, 0.0219f, 0.6011f, 0.1405f, 0.2084f, 0.8264f, 0.8441f, 0.3142f, 0.8675f,
    0.7150f, 0.2648f, 0.8851f, 0.2615f, 0.5952f, 0.4780f, 0.7673f, 0.4611f, 0.5732f, 0.007700f,
    0.2121f, 0.5508f, 0.5235f, 0.3081f, 0.6602f, 0.2890f, 0.2314f, 0.2279f, 0.0966f, 0.1091f,
    0.7151f, 0.8579f, 0.5771f, 0.5114f, 0.1901f, 0.9081f, 0.6009f, 0.7198f, 0.1064f, 0.8684f,
    0.5680f, 0.0281f, 0.4014f, 0.6315f, 0.1148f, 0.0758f, 0.9423f, 0.7203f, 0.3685f, 0.1743f,
    0.7721f, 0.3028f, 0.5564f, 0.9998f, 0.3652f, 0.5258f, 0.3703f, 0.6779f, 0.9935f, 0.5027f,
    0.7396f, 0.0456f, 0.7474f, 0.9288f, 0.002200f, 0.0826f, 0.3634f, 0.4912f, 0.9405f, 0.3891f,
    0.5637f, 0.8554f, 0.0321f, 0.2638f, 0.3609f, 0.6497f, 0.8469f, 0.9350f, 0.0370f, 0.2917f,
    0.8656f, 0.6327f, 0.3562f, 0.6356f, 0.2736f, 0.6512f, 0.1022f, 0.2888f, 0.5762f, 0.4079f,
    0.5332f, 0.4121f, 0.7287f, 0.2311f, 0.6830f, 0.7386f, 0.8180f, 0.9815f, 0.8055f, 0.2566f,
    -0.4612f, 0.7266f, 0.4778f, 0.1584f,
    -0.4612f, -0.7266f, 0.4778f, 0.1584f,
    -0.4516f, 0.0f, 0.4603f, 0.1993f,
    -0.1492f, 0.4825f, 0.4750f, 0.0917f,
    -0.1492f, -0.4825f, 0.4750f, 0.0917f,
    0.0331f, 0.0f, 0.2973f, 0.0825f,
    0.3085f, 0.1195f, 0.4295f, 0.0397f,
    0.3085f, -0.1195f, 0.4295f, 0.0397f,
    0.5451f, 0.0f, 0.7078f, 0.1503f,
    5.0352f, 0.0f, 0.9726f, 3.5548f,

    /* Case 16: N=4 */
    -0.3873f, 0.3656f, 0.0312f, -0.5834f,
    0.5523f, -1.1854f, 0.9833f, 0.7667f,
    1.6746f, -0.0199f, -1.8293f, 0.5718f,
    -0.5250f, 0.3534f, -0.2721f, -0.0883f,
    -1.8952f, 0.7506f, 0.8191f, 0.7709f,
    -1.8952f, -0.7506f, 0.8191f, 0.7709f,
    -0.0952f, 0.0f, 0.8050f, 0.4904f,
    0.3952f, 0.0f, 0.9822f, 0.4904f,

    /* Case 17: N=6 */
    -1.0777f, 1.7027f, 0.2651f, 0.8516f, 1.0121f, 0.2571f,
    -0.0134f, 0.3903f, -1.2680f, 0.2753f, -0.3235f, -1.3844f,
    0.1523f, 0.3068f, 0.8733f, -0.3341f, -0.4831f, -1.5416f,
    0.1447f, -0.6057f, 0.0319f, -1.0905f, -0.0837f, 0.6241f,
    -0.7651f, -1.7889f, -1.5069f, -0.6021f, 0.5217f, 0.6470f,
    0.8194f, 0.2110f, 0.5432f, 0.7561f, 0.1713f, 0.5540f,
    -1.7029f, 0.0f, 0.6791f, 0.6722f,
    -1.0307f, 0.0f, 0.7267f, 0.2044f,
    0.2849f, 1.2101f, 0.3976f, 0.4980f,
    0.2849f, -1.2101f, 0.3976f, 0.4980f,
    1.1675f, 0.4663f, 0.4233f, 0.1905f,
    1.1675f, -0.4663f, 0.4233f, 0.1905f,

    /* Case 18: N=10 */
    -1.0639f, 0.1612f, 0.1562f, 0.3436f, -0.6748f, 1.6598f, 0.6465f, -0.7863f, -0.2610f, 0.7019f,
    -0.8440f, -2.2439f, 1.8800f, -1.0005f, 0.0745f, -1.6156f, 0.2822f, 0.8560f, 1.3497f, -1.5883f,
    1.5988f, 1.1758f, 1.2398f, 1.1173f, 0.2150f, 0.4314f, 0.1850f, 0.7947f, 0.6626f, 0.8646f,
    -0.2296f, 1.2442f, 2.3242f, -0.5069f, -0.7516f, -0.5437f, -0.2599f, 1.2830f, -1.1067f, -0.1115f,
    -0.3604f, 0.4042f, 0.6124f, -1.2164f, -0.9465f, -0.3146f, 0.1831f, 0.7371f, 1.4278f, 0.2922f,
    0.4615f, 0.3874f, -0.0429f, -0.9360f, 0.7116f, -0.8259f, -1.7640f, -0.9466f, 1.8202f, -0.2548f,
    1.2934f, -0.9755f, 0.6748f, -1.0481f, -1.8442f, -0.0546f, 0.7405f, 0.006100f, 1.2430f, -0.1849f,
    -0.3471f, -0.9580f, 0.1653f, 0.0913f, -0.5201f, -1.1832f, 0.8541f, -0.2320f, -1.6155f, 0.5518f,
    1.0190f, -0.6824f, 0.8085f, 0.2595f, -0.3758f, -1.8825f, 1.6473f, -0.6592f, 0.8025f, -0.004900f,
    1.2670f, -0.0424f, 0.8957f, -0.1677f, 0.1462f, 0.9880f, -0.2317f, -1.4483f, -0.0582f, 0.0197f,
    -2.6992f, 0.9039f, 0.6401f, 0.4162f,
    -2.6992f, -0.9039f, 0.6401f, 0.4162f,
    -2.4366f, 0.0f, 0.6908f, 0.2548f,
    -1.2882f, 0.8893f, 0.5343f, 0.6088f,
    -1.2882f, -0.8893f, 0.5343f, 0.6088f,
    0.9028f, 0.0f, 0.2980f, 0.4753f,
    0.9044f, 2.5661f, 0.7319f, 0.6202f,
    0.9044f, -2.5661f, 0.7319f, 0.6202f,
    1.6774f, 0.0f, 0.3074f, 0.4173f,
    3.0060f, 0.0f, 0.8562f, 0.4318f,

    /* Case 19: N=4 */
    -1.2298f, -2.3142f, -0.0698f, 1.0523f,
    0.2039f, -1.2298f, 0.0805f, 0.9786f,
    0.0f, 0.0f, 0.2560f, -0.8910f,
    0.0f, 0.0f, 0.2748f, 0.2560f,
    -1.2298f, 0.6869f, 0.4714f, 0.7177f,
    -1.2298f, -0.6869f, 0.4714f, 0.7177f,
    0.2560f, 0.4948f, 0.8096f, 0.5141f,
    0.2560f, -0.4948f, 0.8096f, 0.5141f,

    /* Case 20: N=6 */
    0.5993f, 1.9372f, -0.1616f, -1.4602f, 0.6018f, 2.7120f,
    -2.2049f, 0.5993f, -1.0679f, 1.9405f, -1.4400f, -0.2211f,
    0.0f, 0.0f, -2.4567f, -0.6865f, -1.9101f, 0.6496f,
    0.0f, 0.0f, 0.0f, 0.7362f, 0.3970f, -0.1519f,
    0.0f, 0.0f, 0.0f, 0.0f, -1.0034f, 1.1954f,
    0.0f, 0.0f, 0.0f, 0.0f, -0.1340f, -1.0034f,
    -2.4567f, 0.0f, 0.4709f, 0.8579f,
    -1.0034f, 0.4002f, 0.3689f, 0.1891f,
    -1.0034f, -0.4002f, 0.3689f, 0.1891f,
    0.5993f, 2.0667f, 0.5885f, 1.3299f,
    0.5993f, -2.0667f, 0.5885f, 1.3299f,
    0.7362f, 0.0f, 0.6085f, 0.9673f,

    /* Case 21: N=4 */
    1.0000e-04f, 1.0f, 0.0f, 0.0f,
    0.0f, -1.0000e-04f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0100f, 1.0f,
    0.0f, 0.0f, 0.0f, -0.005000f,
    -0.005000f, 0.0f, 3.7485e-07f, 3.6932e-07f,
    -1.0000e-04f, 0.0f, 9.8979e-09f, 9.8493e-09f,
    1.0000e-04f, 0.0f, 1.0098e-08f, 1.0046e-08f,
    0.0100f, 0.0f, 1.4996e-06f, 1.4773e-06f,

    /* Case 22: N=3 */
    2.0000e-06f, 1.0f, -2.0f,
    1.0000e-06f, -2.0f, 4.0f,
    0.0f, 1.0f, -2.0f,
    -4.0f, 0.0f, 0.7303f, 4.0f,
    0.0f, 0.0f, 0.7280f, 1.3726e-06f,
    2.2096e-06f, 0.0f, 0.8276f, 2.2096e-06f,

    /* Case 23: N=6 */
    0.2408f, 0.6553f, 0.9166f, 0.0503f, 0.2849f, 0.2408f,
    0.6907f, 0.9700f, 0.1402f, 0.5782f, 0.6767f, 0.6907f,
    0.1062f, 0.0380f, 0.7054f, 0.2432f, 0.8642f, 0.1062f,
    0.2640f, 0.0988f, 0.0178f, 0.9448f, 0.1943f, 0.2640f,
    0.7034f, 0.2560f, 0.2611f, 0.5876f, 0.0580f, 0.7034f,
    0.4021f, 0.5598f, 0.1358f, 0.7256f, 0.6908f, 0.4021f,
    -0.3401f, 0.3213f, 0.5784f, 0.2031f,
    -0.3401f, -0.3213f, 0.5784f, 0.2031f,
    -1.6998e-07f, 0.0f, 0.4964f, 0.2157f,
    0.7231f, 0.0594f, 0.7004f, 0.0419f,
    0.7231f, -0.0594f, 0.7004f, 0.0419f,
    2.5551f, 0.0f, 0.9252f, 1.7390f,

    /* Case 24: N=6 */
    3.4800f, -2.9900f, 0.0f, 0.0f, 0.0f, 0.0f,
    -0.4900f, 2.4800f, -1.9900f, 0.0f, 0.0f, 0.0f,
    0.0f, -0.4900f, 1.4800f, -0.9900f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.9900f, 1.4800f, -0.4900f, 0.0f,
    0.0f, 0.0f, 0.0f, -1.9900f, 2.4800f, -0.4900f,
    0.0f, 0.0f, 0.0f, 0.0f, -2.9900f, 3.4800f,
    0.0130f, 0.0f, 0.7530f, 0.6053f,
    1.1294f, 0.0f, 0.6048f, 0.2861f,
    2.0644f, 0.0f, 0.5466f, 0.1738f,
    2.8388f, 0.0f, 0.4277f, 0.3091f,
    4.3726f, 0.0f, 0.6637f, 0.0764f,
    4.4618f, 0.0f, 0.5739f, 0.0892f,

    /* Case 25: N=6 */
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    -1.7321f, 0.0f, 0.8660f, 0.7260f,
    -1.0f, 0.0f, 0.5000f, 0.2642f,
    0.0f, 0.0f, 2.9582e-31f, 1.4600e-07f,
    0.0f, 0.0f, 2.9582e-31f, 6.2446e-08f,
    1.0f, 0.0f, 0.5000f, 0.2642f,
    1.7321f, 0.0f, 0.8660f, 0.3790f,

    /* Case 26: N=6 */
    0.3534f, 0.9302f, 0.0747f, -0.0101f, 0.0467f, -0.0435f,
    0.9355f, -0.3515f, -0.0282f, 0.003801f, -0.0176f, 0.0164f,
    0.0f, -0.1056f, 0.7521f, -0.1013f, 0.4703f, -0.4379f,
    0.0f, 0.0f, 0.6542f, 0.1178f, -0.5468f, 0.5091f,
    0.0f, 0.0f, 0.0f, -0.9878f, -0.1140f, 0.1061f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.6814f, 0.7319f,
    -0.9998f, 0.0196f, 1.0f, 0.0393f,
    -0.9998f, -0.0196f, 1.0f, 0.0393f,
    0.7454f, 0.6666f, 1.0f, 0.5212f,
    0.7454f, -0.6666f, 1.0f, 0.5212f,
    0.9993f, 0.0375f, 1.0f, 0.0751f,
    0.9993f, -0.0375f, 1.0f, 0.0751f,

    /* Case 27: N=6 */
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5000f, 0.3333f, 0.2500f, 0.2000f, 0.1667f, 0.1429f,
    0.3333f, 0.2500f, 0.2000f, 0.1667f, 0.1429f, 0.1250f,
    0.2500f, 0.2000f, 0.1667f, 0.1429f, 0.1250f, 0.1111f,
    0.2000f, 0.1667f, 0.1429f, 0.1250f, 0.1111f, 0.1000f,
    0.1667f, 0.1429f, 0.1250f, 0.1111f, 0.1000f, 0.0909f,
    -0.2213f, 0.0f, 0.4084f, 0.1661f,
    -0.0320f, 0.0f, 0.3793f, 0.0305f,
    -8.5031e-04f, 0.0f, 0.6279f, 7.8195e-04f,
    -5.8584e-05f, 0.0f, 0.8116f, 7.2478e-05f,
    1.3895e-05f, 0.0f, 0.9709f, 7.2478e-05f,
    2.1324f, 0.0f, 0.8433f, 1.8048f,

    /* Case 28: N=12 */
    12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    11.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 10.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 9.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    -0.0282f, 0.0f, 2.8690e-06f, 3.2094e-06f,
    0.0726f, 0.0907f, 1.5885e-06f, 9.9934e-07f,
    0.0726f, -0.0907f, 1.5885e-06f, 9.9934e-07f,
    0.1853f, 0.0f, 6.5757e-07f, 7.8673e-07f,
    0.2883f, 0.0f, 1.8324e-06f, 2.0796e-06f,
    0.6431f, 0.0f, 6.8640e-05f, 6.1058e-05f,
    1.5539f, 0.0f, 0.004626f, 0.006403f,
    3.5119f, 0.0f, 0.1445f, 0.1947f,
    6.9615f, 0.0f, 0.5845f, 1.2016f,
    12.3110f, 0.0f, 0.3182f, 1.4273f,
    20.1990f, 0.0f, 0.2008f, 2.4358f,
    32.2290f, 0.0f, 0.3042f, 5.6865f,

    /* Case 29: N=6 */
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    5.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 4.0f, 0.0f, 3.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 5.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    -5.0f, 0.0f, 0.8229f, 1.2318f,
    -3.0f, 0.0f, 0.7228f, 0.7597f,
    -1.0f, 0.0f, 0.6285f, 0.6967f,
    1.0f, 0.0f, 0.6285f, 0.6967f,
    3.0f, 0.0f, 0.7228f, 0.7597f,
    5.0f, 0.0f, 0.8229f, 1.2318f,

    /* Case 30: N=6 */
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
    0.0803f, 2.4187f, 0.8997f, 1.5236f,
    0.0803f, -2.4187f, 0.8997f, 1.5236f,
    1.4415f, 0.6285f, 0.9673f, 0.4279f,
    1.4415f, -0.6285f, 0.9673f, 0.4279f,
    1.4782f, 0.1564f, 0.9760f, 0.2200f,
    1.4782f, -0.1564f, 0.9760f, 0.2200f,

    /* Case 31: N=6 */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
    -0.0353f, 0.7481f, 0.3935f, 0.1842f,
    -0.0353f, -0.7481f, 0.3935f, 0.1842f,
    5.8440e-07f, 0.0f, 0.2887f, 0.1700f,
    0.6409f, 0.7282f, 0.4501f, 0.2943f,
    0.6409f, -0.7282f, 0.4501f, 0.2943f,
    3.7889f, 0.0f, 0.9630f, 2.2469f,

    /* Case 32: N=6 */
    1.0f, 4.0112f, 12.7500f, 40.2130f, 126.5600f, 397.8800f,
    1.0f, 3.2616f, 10.6290f, 33.3420f, 104.7900f, 329.3600f,
    1.0f, 3.1500f, 9.8006f, 30.6300f, 96.1640f, 302.1500f,
    1.0f, 3.2755f, 10.4200f, 32.9570f, 103.7400f, 326.1600f,
    1.0f, 2.8214f, 8.4558f, 26.2960f, 82.4430f, 258.9300f,
    1.0f, 2.6406f, 8.3565f, 26.5580f, 83.5580f, 262.6800f,
    -0.5322f, 0.0f, 0.5329f, 0.3856f,
    -0.1012f, 0.0f, 0.7234f, 0.0913f,
    -0.009875f, 0.0f, 0.7371f, 0.0110f,
    0.002986f, 0.0f, 0.4461f, 0.0129f,
    0.1807f, 0.0f, 0.4288f, 0.1738f,
    392.6000f, 0.0f, 0.4806f, 392.0100f,

    /* Case 33: N=8 */
    0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 4.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    -3.7588f, 0.0f, 0.1225f, 0.1298f,
    -3.0642f, 0.0f, 0.0498f, 0.0802f,
    -2.0f, 0.0f, 0.0369f, 0.0829f,
    -0.6946f, 0.0f, 0.0333f, 0.1374f,
    0.6946f, 0.0f, 0.0333f, 0.1117f,
    2.0f, 0.0f, 0.0369f, 0.0722f,
    3.0642f, 0.0f, 0.0498f, 0.0684f,
    3.7588f, 0.0f, 0.1225f, 0.1153f,

    /* Case 34: N=6 */
    8.5000f, -10.4720f, 2.8944f, -1.5279f, 1.1056f, -0.5000f,
    2.6180f, -1.1708f, -2.0f, 0.8944f, -0.6180f, 0.2764f,
    -0.7236f, 2.0f, -0.1708f, -1.6180f, 0.8944f, -0.3820f,
    0.3820f, -0.8944f, 1.6180f, 0.1708f, -2.0f, 0.7236f,
    -0.2764f, 0.6180f, -0.8944f, 2.0f, 1.1708f, -2.6180f,
    0.5000f, -1.1056f, 1.5279f, -2.8944f, 10.4720f, -8.5000f,
    -0.5893f, 0.0f, 1.7357e-04f, 2.8157e-04f,
    -0.2763f, 0.4985f, 1.7486e-04f, 1.6704e-04f,
    -0.2763f, -0.4985f, 1.7486e-04f, 1.6704e-04f,
    0.2751f, 0.5006f, 1.7635e-04f, 1.6828e-04f,
    0.2751f, -0.5006f, 1.7635e-04f, 1.6828e-04f,
    0.5917f, 0.0f, 1.7623e-04f, 3.0778e-04f,

    /* Case 35: N=4 */
    4.0f, -5.0f, 0.0f, 3.0f,
    0.0f, 4.0f, -3.0f, -5.0f,
    5.0f, -3.0f, 4.0f, 0.0f,
    3.0f, 0.0f, 5.0f, 4.0f,
    1.0f, 5.0f, 1.0f, 4.3333f,
    1.0f, -5.0f, 1.0f, 4.3333f,
    2.0f, 0.0f, 1.0f, 4.3333f,
    12.0f, 0.0f, 1.0f, 9.1250f,

    /* Case 36: N=5 */
    15.0f, 11.0f, 6.0f, -9.0f, -15.0f,
    1.0f, 3.0f, 9.0f, -3.0f, -8.0f,
    7.0f, 6.0f, 6.0f, -3.0f, -11.0f,
    7.0f, 7.0f, 5.0f, -3.0f, -11.0f,
    17.0f, 12.0f, 5.0f, -10.0f, -16.0f,
    -1.0000f, 0.0f, 0.2177f, 0.5226f,
    1.4980f, 3.5752f, 3.9966e-04f, 0.006095f,
    1.4980f, -3.5752f, 3.9966e-04f, 0.006095f,
    1.5020f, 3.5662f, 3.9976e-04f, 0.006096f,
    1.5020f, -3.5662f, 3.9976e-04f, 0.006096f,

    /* Case 37: N=6 */
    -9.0f, 21.0f, -15.0f, 4.0f, 2.0f, 0.0f,
    -10.0f, 21.0f, -14.0f, 4.0f, 2.0f, 0.0f,
    -8.0f, 16.0f, -11.0f, 4.0f, 2.0f, 0.0f,
    -6.0f, 12.0f, -9.0f, 3.0f, 3.0f, 0.0f,
    -4.0f, 8.0f, -6.0f, 0.0f, 5.0f, 0.0f,
    -2.0f, 4.0f, -3.0f, 0.0f, 1.0f, 3.0f,
    1.0f, 6.2559e-04f, 6.4875e-05f, 5.0367e-04f,
    1.0f, -6.2559e-04f, 6.4875e-05f, 5.0367e-04f,
    2.0f, 1.0001f, 0.0541f, 0.2351f,
    2.0f, -1.0001f, 0.0541f, 0.2351f,
    3.0f, 0.0f, 0.8615f, 5.4838e-07f,
    3.0f, 0.0f, 0.1242f, 1.2770e-06f,

    /* Case 38: N=10 */
    1.0f, 1.0f, 1.0f, -2.0f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -3.0f,
    -1.0f, 2.0f, 3.0f, -4.0f, 2.0f, -2.0f, 4.0f, -4.0f, 8.0f, -6.0f,
    -1.0f, 0.0f, 5.0f, -5.0f, 3.0f, -3.0f, 6.0f, -6.0f, 12.0f, -9.0f,
    -1.0f, 0.0f, 3.0f, -4.0f, 4.0f, -4.0f, 8.0f, -8.0f, 16.0f, -12.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 5.0f, -4.0f, 10.0f, -10.0f, 20.0f, -15.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -2.0f, 12.0f, -12.0f, 24.0f, -18.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 15.0f, -13.0f, 28.0f, -21.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -11.0f, 32.0f, -24.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -14.0f, 37.0f, -26.0f,
    -1.0f, 0.0f, 3.0f, -6.0f, 2.0f, -5.0f, 12.0f, -14.0f, 36.0f, -25.0f,
    1.0f, 0.0f, 0.0360f, 0.0796f,
    1.9867f, 0.0f, 7.4283e-05f, 7.4025e-06f,
    2.0f, 0.002505f, 1.4346e-04f, 6.7839e-07f,
    2.0f, -0.002505f, 1.4346e-04f, 6.7839e-07f,
    2.0067f, 0.0118f, 6.7873e-05f, 5.7496e-06f,
    2.0067f, -0.0118f, 6.7873e-05f, 5.7496e-06f,
    2.9970f, 0.0f, 9.2779e-05f, 2.6519e-06f,
    3.0f, 8.7028e-04f, 2.7358e-04f, 1.9407e-07f,
    3.0f, -8.7028e-04f, 2.7358e-04f, 1.9407e-07f,
    3.0030f, 0.0f, 9.2696e-05f, 2.6477e-06f,
};

static const INT dget37_n[] = {
    1, 1, 2, 2, 2, 6, 4, 5, 5, 6,
    6, 6, 4, 6, 5, 10, 4, 6, 10, 4,
    6, 4, 3, 6, 6, 6, 6, 6, 12, 6,
    6, 6, 6, 8, 6, 4, 5, 6, 10,
};

static void rowmajor_to_colmajor(const f32* rows, f32* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(f32));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

void sget37(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;
    const f32 TWO  = 2.0f;
    const f32 EPSIN = 5.9605e-8f;

    INT    i, icmp, ifnd, info, iscl, j, kmin, m, n;
    f32    bignum, eps, smlnum, tnrm, tol, tolin, v,
           vimin, vmax, vmul, vrmin;

    INT    select[LDT];
    INT    iwork[2 * LDT], lcmp[3];
    f32    dum[1], le[LDT * LDT], re[LDT * LDT],
           s[LDT], sep[LDT], sepin[LDT],
           septmp[LDT], sin_vals[LDT], stmp[LDT],
           t[LDT * LDT], tmp[LDT * LDT], val[3],
           wi[LDT], witmp[LDT],
           work[LWORK], wr[LDT],
           wrtmp[LDT];

    eps = slamch("P");
    smlnum = slamch("S") / eps;
    bignum = ONE / smlnum;

    eps = fmaxf(eps, EPSIN);
    rmax[0] = ZERO;
    rmax[1] = ZERO;
    rmax[2] = ZERO;
    lmax[0] = 0;
    lmax[1] = 0;
    lmax[2] = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    ninfo[2] = 0;

    val[0] = sqrtf(smlnum);
    val[1] = ONE;
    val[2] = sqrtf(bignum);

    INT data_offset = 0;

    for (INT icase = 0; icase < NCASES37; icase++) {
        n = dget37_n[icase];

        rowmajor_to_colmajor(&dget37_data[data_offset], tmp, n, LDT);
        data_offset += n * n;

        for (i = 0; i < n; i++) {
            sin_vals[i] = dget37_data[data_offset + i * 4 + 2];
            sepin[i]   = dget37_data[data_offset + i * 4 + 3];
        }
        data_offset += n * 4;

        tnrm = slange("M", n, n, tmp, LDT, work);

        for (iscl = 0; iscl < 3; iscl++) {

            *knt = *knt + 1;
            slacpy("F", n, n, tmp, LDT, t, LDT);
            vmul = val[iscl];
            for (i = 0; i < n; i++)
                cblas_sscal(n, vmul, &t[i * LDT], 1);
            if (tnrm == ZERO)
                vmul = ONE;

            sgehrd(n, 0, n - 1, t, LDT, &work[0], &work[n], LWORK - n,
                   &info);
            if (info != 0) {
                lmax[0] = *knt;
                ninfo[0] = ninfo[0] + 1;
                continue;
            }
            for (j = 0; j < n - 2; j++)
                for (i = j + 2; i < n; i++)
                    t[i + j * LDT] = ZERO;

            shseqr("S", "N", n, 0, n - 1, t, LDT, wr, wi, dum, 1, work,
                   LWORK, &info);
            if (info != 0) {
                lmax[1] = *knt;
                ninfo[1] = ninfo[1] + 1;
                continue;
            }

            strevc("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, n, &m, work, &info);

            strsna("Both", "All", select, n, t, LDT, le, LDT, re,
                   LDT, s, sep, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }

            cblas_scopy(n, wr, 1, wrtmp, 1);
            cblas_scopy(n, wi, 1, witmp, 1);
            cblas_scopy(n, s, 1, stmp, 1);
            cblas_scopy(n, sep, 1, septmp, 1);
            cblas_sscal(n, ONE / vmul, septmp, 1);
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
                vrmin = stmp[kmin];
                stmp[kmin] = stmp[i];
                stmp[i] = vrmin;
                vrmin = septmp[kmin];
                septmp[kmin] = septmp[i];
                septmp[i] = vrmin;
            }

            v = fmaxf(TWO * (f32)n * eps * tnrm, smlnum);
            if (tnrm == ZERO)
                v = ONE;
            for (i = 0; i < n; i++) {
                if (v > septmp[i])
                    tol = ONE;
                else
                    tol = v / septmp[i];
                if (v > sepin[i])
                    tolin = ONE;
                else
                    tolin = v / sepin[i];
                tol = fmaxf(tol, smlnum / eps);
                tolin = fmaxf(tolin, smlnum / eps);
                if (eps * (sin_vals[i] - tolin) > stmp[i] + tol) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] - tolin > stmp[i] + tol) {
                    vmax = (sin_vals[i] - tolin) / (stmp[i] + tol);
                } else if (sin_vals[i] + tolin < eps * (stmp[i] - tol)) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] + tolin < stmp[i] - tol) {
                    vmax = (stmp[i] - tol) / (sin_vals[i] + tolin);
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[1]) {
                    rmax[1] = vmax;
                    if (ninfo[1] == 0)
                        lmax[1] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (v > septmp[i] * stmp[i])
                    tol = septmp[i];
                else
                    tol = v / stmp[i];
                if (v > sepin[i] * sin_vals[i])
                    tolin = sepin[i];
                else
                    tolin = v / sin_vals[i];
                tol = fmaxf(tol, smlnum / eps);
                tolin = fmaxf(tolin, smlnum / eps);
                if (eps * (sepin[i] - tolin) > septmp[i] + tol) {
                    vmax = ONE / eps;
                } else if (sepin[i] - tolin > septmp[i] + tol) {
                    vmax = (sepin[i] - tolin) / (septmp[i] + tol);
                } else if (sepin[i] + tolin < eps * (septmp[i] - tol)) {
                    vmax = ONE / eps;
                } else if (sepin[i] + tolin < septmp[i] - tol) {
                    vmax = (septmp[i] - tol) / (sepin[i] + tolin);
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[1]) {
                    rmax[1] = vmax;
                    if (ninfo[1] == 0)
                        lmax[1] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (sin_vals[i] <= (f32)(2 * n) * eps && stmp[i] <=
                    (f32)(2 * n) * eps) {
                    vmax = ONE;
                } else if (eps * sin_vals[i] > stmp[i]) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] > stmp[i]) {
                    vmax = sin_vals[i] / stmp[i];
                } else if (sin_vals[i] < eps * stmp[i]) {
                    vmax = ONE / eps;
                } else if (sin_vals[i] < stmp[i]) {
                    vmax = stmp[i] / sin_vals[i];
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[2]) {
                    rmax[2] = vmax;
                    if (ninfo[2] == 0)
                        lmax[2] = *knt;
                }
            }

            for (i = 0; i < n; i++) {
                if (sepin[i] <= v && septmp[i] <= v) {
                    vmax = ONE;
                } else if (eps * sepin[i] > septmp[i]) {
                    vmax = ONE / eps;
                } else if (sepin[i] > septmp[i]) {
                    vmax = sepin[i] / septmp[i];
                } else if (sepin[i] < eps * septmp[i]) {
                    vmax = ONE / eps;
                } else if (sepin[i] < septmp[i]) {
                    vmax = septmp[i] / sepin[i];
                } else {
                    vmax = ONE;
                }
                if (vmax > rmax[2]) {
                    rmax[2] = vmax;
                    if (ninfo[2] == 0)
                        lmax[2] = *knt;
                }
            }

            vmax = ZERO;
            dum[0] = -ONE;
            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            strsna("Eigcond", "All", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            strsna("Veccond", "All", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
            }

            for (i = 0; i < n; i++)
                select[i] = 1;
            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            strsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
                   re, LDT, stmp, septmp, n, &m, work, n, iwork,
                   &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
            }

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            strsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != s[i])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_scopy(n, dum, 0, stmp, 1);
            cblas_scopy(n, dum, 0, septmp, 1);
            strsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < n; i++) {
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[i])
                    vmax = ONE / eps;
            }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

            if (wi[0] == ZERO) {
                lcmp[0] = 0;
                ifnd = 0;
                for (i = 1; i < n; i++) {
                    if (ifnd == 1 || wi[i] == ZERO) {
                        select[i] = 0;
                    } else {
                        ifnd = 1;
                        lcmp[1] = i;
                        lcmp[2] = i + 1;
                        cblas_scopy(n, &re[i * LDT], 1, &re[1 * LDT], 1);
                        cblas_scopy(n, &re[(i + 1) * LDT], 1, &re[2 * LDT], 1);
                        cblas_scopy(n, &le[i * LDT], 1, &le[1 * LDT], 1);
                        cblas_scopy(n, &le[(i + 1) * LDT], 1, &le[2 * LDT], 1);
                    }
                }
                if (ifnd == 0) {
                    icmp = 1;
                } else {
                    icmp = 3;
                }
            } else {
                lcmp[0] = 0;
                lcmp[1] = 1;
                ifnd = 0;
                for (i = 2; i < n; i++) {
                    if (ifnd == 1 || wi[i] != ZERO) {
                        select[i] = 0;
                    } else {
                        lcmp[2] = i;
                        ifnd = 1;
                        cblas_scopy(n, &re[i * LDT], 1, &re[2 * LDT], 1);
                        cblas_scopy(n, &le[i * LDT], 1, &le[2 * LDT], 1);
                    }
                }
                if (ifnd == 0) {
                    icmp = 2;
                } else {
                    icmp = 3;
                }
            }

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            strsna("Bothcond", "Some", select, n, t, LDT, le, LDT,
                   re, LDT, stmp, septmp, n, &m, work, n, iwork,
                   &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (septmp[i] != sep[j])
                    vmax = ONE / eps;
                if (stmp[i] != s[j])
                    vmax = ONE / eps;
            }

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            strsna("Eigcond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (stmp[i] != s[j])
                    vmax = ONE / eps;
                if (septmp[i] != dum[0])
                    vmax = ONE / eps;
            }

            cblas_scopy(icmp, dum, 0, stmp, 1);
            cblas_scopy(icmp, dum, 0, septmp, 1);
            strsna("Veccond", "Some", select, n, t, LDT, le, LDT, re,
                   LDT, stmp, septmp, n, &m, work, n, iwork, &info);
            if (info != 0) {
                lmax[2] = *knt;
                ninfo[2] = ninfo[2] + 1;
                continue;
            }
            for (i = 0; i < icmp; i++) {
                j = lcmp[i];
                if (stmp[i] != dum[0])
                    vmax = ONE / eps;
                if (septmp[i] != sep[j])
                    vmax = ONE / eps;
            }
            if (vmax > rmax[0]) {
                rmax[0] = vmax;
                if (ninfo[0] == 0)
                    lmax[0] = *knt;
            }

        } /* end iscl loop */
    } /* end icase loop */
}
