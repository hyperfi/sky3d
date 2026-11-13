# Lead-208 on a 40³ grid to 3000 fm/c

Both builds resumed the same padded-grid checkpoint at 4 fm/c and evolved 14,980 TDHF steps at 0.2 fm/c on a 40³ grid with 1 fm spacing. The GPU used eight host threads and the optimized CPU build used 24 OpenMP threads. The runs were sequential.

| Build | Wall time (s) |
| --- | ---: |
| RTX A2000 | 26615.12 |
| Xeon w7-3455 | 31214.86 |

The GPU throughput was **1.173×** the CPU throughput.

CPU–GPU numerical comparison **passed**. The endpoint wavefunction relative L2 error was 1.058e-12. The endpoint conservation checks **failed**. The per-backend values and observable checks are in [summary.json](summary.json).

The [response plot](response/pb208-isq20-gamma1.png) overlays the isoscalar quadrupole spectra and plots the GPU–CPU difference. Both transforms use the common CPU samples before the restart and exponential damping with a 1 MeV Lorentzian full width at half maximum. The plotted interval is 0–3000 fm/c. The first sample sets the baseline. The archive includes the [PDF](response/pb208-isq20-gamma1.pdf) and [spectrum CSV](response/pb208-isq20-gamma1.csv).
