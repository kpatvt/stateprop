"""Starter UNIFAC subgroup database (original VLE-UNIFAC, Hansen 1991).

This is a small starter set covering the most common groups for
hydrocarbon, alcohol, water, ester, amine, ketone, and aromatic
chemistry. For comprehensive process-simulation coverage users
should extend with the full ~90 main-group / ~150 subgroup database
from Hansen et al. (1991) and subsequent revisions.

Data format:

    SUBGROUPS[name] = (subgroup_id, main_group_id, R, Q)

R is the Bondi-style reduced van der Waals volume; Q is the reduced
surface area (per Bondi 1968 normalization).

    A[main_i][main_j] = a_mn  (units: K)

Asymmetric: a_mn != a_nm in general.
Diagonal: a_mm = 0.

If a main-group pair is not in A, calling code should raise
KeyError -- this prevents silent zero defaults that would yield
wrong gamma values without warning.

Reference: Hansen, H.K., Rasmussen, P., Fredenslund, A., Schiller,
M., Gmehling, J., "Vapor-Liquid Equilibria by UNIFAC Group
Contribution. Revision and Extension. 5", Ind. Eng. Chem. Res. 30,
2352 (1991). Selected entries verified against published tables.
"""

# (name) -> (subgroup_id, main_group_id, R, Q)
SUBGROUPS = {
    # Main group 1: CH2 (paraffinic / aliphatic chain)
    'CH3':       (1, 1, 0.9011, 0.848),
    'CH2':       (2, 1, 0.6744, 0.540),
    'CH':        (3, 1, 0.4469, 0.228),
    'C':         (4, 1, 0.2195, 0.000),

    # Main group 2: C=C
    'CH2=CH':    (5, 2, 1.3454, 1.176),
    'CH=CH':     (6, 2, 1.1167, 0.867),
    'CH2=C':     (7, 2, 1.1173, 0.988),
    'CH=C':      (8, 2, 0.8886, 0.676),

    # Main group 3: ACH (aromatic CH)
    'ACH':       (9,  3, 0.5313, 0.400),
    'AC':        (10, 3, 0.3652, 0.120),

    # Main group 4: ACCH2 (toluene-type)
    'ACCH3':     (11, 4, 1.2663, 0.968),
    'ACCH2':     (12, 4, 1.0396, 0.660),
    'ACCH':      (13, 4, 0.8121, 0.348),

    # Main group 5: OH (alcohols)
    'OH':        (14, 5, 1.0000, 1.200),

    # Main group 6: CH3OH (methanol special)
    'CH3OH':     (15, 6, 1.4311, 1.432),

    # Main group 7: H2O (water)
    'H2O':       (16, 7, 0.9200, 1.400),

    # Main group 9: CH2CO (ketones)
    'CH3CO':     (18, 9, 1.6724, 1.488),
    'CH2CO':     (19, 9, 1.4457, 1.180),

    # Main group 11: CCOO (esters)
    'CH3COO':    (21, 11, 1.9031, 1.728),
    'CH2COO':    (22, 11, 1.6764, 1.420),

    # Main group 13: CH2O (ethers)
    'CH3O':      (25, 13, 1.1450, 1.088),
    'CH2O':      (26, 13, 0.9183, 0.780),
    'CHO':       (27, 13, 0.6908, 0.468),

    # Main group 19: CCN (nitriles)
    'CH3CN':     (41, 19, 1.8701, 1.724),
    'CH2CN':     (42, 19, 1.6434, 1.416),

    # Main group 20: COOH
    'COOH':      (43, 20, 1.3013, 1.224),
    'HCOOH':     (44, 20, 1.5280, 1.532),
}


# Main-group interaction parameters a_mn [K] (Hansen 1991 Table 5).
# A[m][n] = a_mn. Asymmetric. Missing entries raise KeyError.
# Note: a_ii = 0 by convention.
A_MAIN = {
    # CH2 (1)
    1: {
        1: 0.0,    2: 86.02,  3: 61.13,  4: 76.50,  5: 986.5,  6: 697.2,
        7: 1318.0, 9: 476.4,  11: 232.1, 13: 251.5, 19: 597.0, 20: 663.5,
    },
    # C=C (2)
    2: {
        1: -35.36, 2: 0.0,    3: 38.81,  4: 74.15,  5: 524.1,  6: 787.6,
        7: 270.6,  9: 524.5,  11: 71.23, 13: 214.5, 19: 405.9, 20: 730.4,
    },
    # ACH (3)
    3: {
        1: -11.12, 2: 3.446,  3: 0.0,    4: 167.0,  5: 636.1,  6: 637.35,
        7: 903.8,  9: 25.77,  11: 5.994, 13: 32.14, 19: 212.5, 20: 537.4,
    },
    # ACCH2 (4)
    4: {
        1: -69.70, 2: -113.6, 3: -146.8, 4: 0.0,    5: 803.2,  6: 603.25,
        7: 5695.0, 9: -52.10, 11: 5688.0,13: 213.1, 19: 6096.0,20: 872.3,
    },
    # OH (5)
    5: {
        1: 156.4,  2: 457.0,  3: 89.6,   4: 25.82,  5: 0.0,    6: -137.1,
        7: 353.5,  9: 84.0,   11: 101.1, 13: 28.06, 19: 6.712, 20: 199.0,
    },
    # CH3OH (6)
    6: {
        1: 16.51,  2: -12.52, 3: -50.0,  4: -44.50, 5: 249.1,  6: 0.0,
        7: -181.0, 9: 23.39,  11: -10.72,13: -128.6,19: 36.23, 20: -202.0,
    },
    # H2O (7)
    7: {
        1: 300.0,  2: 496.1,  3: 362.3,  4: 377.6,  5: -229.1, 6: 289.6,
        7: 0.0,    9: -195.4, 11: 72.87, 13: 540.5, 19: 112.6, 20: -14.09,
    },
    # CH2CO (9)
    9: {
        1: 26.76,  2: 42.92,  3: 140.1,  4: 365.8,  5: 164.5,  6: 108.7,
        7: 472.5,  9: 0.0,    11: -213.7,13: 5.202, 19: 481.7, 20: 669.4,
    },
    # CCOO (11)
    11: {
        1: 114.8,  2: 132.1,  3: 85.84,  4: -170.0, 5: 245.4,  6: 249.6,
        7: 200.8,  9: 372.2,  11: 0.0,   13: -235.7,19: -213.7,20: 660.2,
    },
    # CH2O (13)
    13: {
        1: 83.36,  2: 26.51,  3: 52.13,  4: 65.69,  5: 237.7,  6: 339.7,
        7: -314.7, 9: 52.38,  11: 461.3, 13: 0.0,   19: -18.51,20: 664.6,
    },
    # CCN (19)
    19: {
        1: 24.82,  2: -40.62, 3: -22.97, 4: -138.4, 5: 185.4,  6: 157.8,
        7: 242.8,  9: -287.5, 11: 152.4, 13: 254.8, 19: 0.0,   20: 565.9,
    },
    # COOH (20)
    20: {
        1: 315.3,  2: 1264.0, 3: 62.32,  4: 89.86,  5: -151.0, 6: 339.8,
        7: -66.17, 9: -297.8, 11: -337.0,13: -338.5,19: -155.6,20: 0.0,
    },
}
