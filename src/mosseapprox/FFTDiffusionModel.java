package mosseapprox;

import java.util.function.DoubleUnaryOperator;

/**
 * Java port of fft_diffusion.py's second-order (Strang-split, Duhamel-source)
 * FFT diffusion solver -- an alternative to SubstitutionModel.solveFMeanNVarN
 * for the special case of plain Brownian motion with CONSTANT drift phi0 and
 * variance rate sigma0^2 (not depending on r or tau). Reuses the closed-form
 * birth-death pieces (cFunc, c2Func) and the Result/interp/normalFT helpers
 * from SubstitutionModel, which must be on the classpath alongside this file.
 *
 * Under plain BM, a step of length dtau has an EXACT transition kernel: a
 * Gaussian with mean phi0*dtau and variance sigma0^2*dtau. Instead of the
 * implicit Crank-Nicolson / tridiagonal solve used for the general-coefficient
 * case, each diffusion sub-step is done by convolving the F/M1/M2 curves
 * directly with the exact discretized Gaussian kernel via FFT.
 *
 * IMPORTANT SIGN NOTE: F(r,tau) is a backward-Kolmogorov / Feynman-Kac
 * expectation, U(r,tau) = E[g(r(T)) | r(T-tau) = r], with r(u) evolving
 * FORWARD via dr = phi0 du + sigma0 dW. Writing the propagation step as a
 * standard convolution h(r) = integral g(r') K(r - r') dr' requires K to be
 * the density of the NEGATIVE of the forward increment, i.e.
 * K = N(x; -phi0*dtau, sigma0^2*dtau) -- NOT N(x; +phi0*dtau, ...). Using the
 * wrong (+) sign silently propagates every curve in the opposite direction; a
 * single-point sanity check at the terminal condition's own center can miss
 * this by symmetry coincidence, so always check the curve away from its peak
 * too. (This mirrors a real bug caught and fixed in the Python version.)
 *
 * Scheme per step (Strang splitting + second-order Duhamel source):
 *   1. half-reaction (exact, Simpson-integrated exp(integral c dtau))
 *   2. exact diffusion via FFT convolution with the Gaussian kernel
 *   3. half-reaction
 *   4. source term (for M1, M2) added via the second-order
 *      exponential/Lawson-trapezoidal Duhamel formula:
 *        M(tau_n) = Phi(tau_o,tau_n)[M(tau_o)]
 *                   + (dtau/2) * ( Phi(tau_o,tau_n)[S(tau_o)] + S(tau_n) )
 *
 * PERFORMANCE NOTE (real-input FFT + kernel caching): the Gaussian kernel is
 * fixed for an entire solve (it only depends on dr, phi0*dtau, sigma0^2*dtau,
 * none of which change across the time loop), but `propagate` is called
 * roughly 2*Nt times per curve (F, M1, M2) -- so a naive implementation that
 * re-transforms the kernel on every call redoes the same forward FFT
 * thousands of times for nothing. This version transforms the kernel ONCE
 * per solve (`makeKernelSpectrum`) and reuses that spectrum for every step.
 * It also uses a real-input-optimized FFT (`rfft`/`irfft`, built on the
 * complex radix-2 `fft` below via the standard "N/2-point complex FFT"
 * packing trick) instead of zero-padding the imaginary part and running a
 * full complex transform -- since F, M1, M2 and the kernel are all
 * real-valued, this roughly halves the arithmetic of both the forward and
 * inverse transforms, mirroring what numpy.fft.rfft/irfft do in the Python
 * version (which this Java port otherwise reimplements from scratch, since
 * no FFT library is available in this environment).
 */
public class FFTDiffusionModel {

    // ---------------------------------------------------------------
    // Minimal iterative radix-2 Cooley-Tukey FFT (in place, complex).
    // Used only as the building block for the real-input rfft/irfft below
    // -- no external library is available in this environment.
    // ---------------------------------------------------------------
    private static void fft(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = (inverse ? 2.0 : -2.0) * Math.PI / len;
            double wr = Math.cos(ang), wi = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double curWr = 1.0, curWi = 0.0;
                int half = len / 2;
                for (int k = 0; k < half; k++) {
                    double ur = re[i + k], ui = im[i + k];
                    double xr = re[i + k + half], xi = im[i + k + half];
                    double vr = xr * curWr - xi * curWi;
                    double vi = xr * curWi + xi * curWr;
                    re[i + k] = ur + vr;         im[i + k] = ui + vi;
                    re[i + k + half] = ur - vr;   im[i + k + half] = ui - vi;
                    double nextWr = curWr * wr - curWi * wi;
                    double nextWi = curWr * wi + curWi * wr;
                    curWr = nextWr; curWi = nextWi;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
        }
    }

    private static int nextPow2(int x) {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    // ---------------------------------------------------------------
    // Real-input FFT (rfft) / real-output inverse FFT (irfft), via the
    // standard "pack pairs of real samples into one N/2-point complex FFT"
    // trick. For even real length n, returns/consumes the non-redundant
    // half-spectrum X[0..n/2] (n/2+1 complex values); the rest is implied
    // by conjugate symmetry X[n-k] = conj(X[k]). This is what lets us do
    // half the arithmetic of a full complex FFT on real-valued data,
    // matching numpy.fft.rfft/irfft's role in the Python version.
    // ---------------------------------------------------------------
    private static double[][] rfft(double[] x) {
        int n = x.length;
        int m = n / 2;
        double[] zre = new double[m], zim = new double[m];
        for (int k = 0; k < m; k++) { zre[k] = x[2 * k]; zim[k] = x[2 * k + 1]; }
        fft(zre, zim, false);

        double[] Xre = new double[m + 1], Xim = new double[m + 1];
        for (int k = 0; k <= m; k++) {
            int kk = k % m, j = (m - k) % m;
            double zr = zre[kk], zi = zim[kk];
            double zjr = zre[j], zji = zim[j];
            double Xer = 0.5 * (zr + zjr), Xei = 0.5 * (zi - zji);
            double dr = zr - zjr, di = zi + zji;
            double Xor = 0.5 * di, Xoi = -0.5 * dr;
            double ang = -2.0 * Math.PI * k / n;
            double twr = Math.cos(ang), twi = Math.sin(ang);
            Xre[k] = Xer + (twr * Xor - twi * Xoi);
            Xim[k] = Xei + (twr * Xoi + twi * Xor);
        }
        return new double[][]{Xre, Xim};
    }

    private static double[] irfft(double[] Xre, double[] Xim, int n) {
        int m = n / 2;
        double[] Zre = new double[m], Zim = new double[m];
        for (int k = 0; k < m; k++) {
            double Xer, Xei, Xor, Xoi;
            if (k == 0) {
                double Xkr = Xre[0], Xki = Xim[0];
                double XMr = Xre[m], XMi = Xim[m];
                Xer = 0.5 * (Xkr + XMr); Xei = 0.5 * (Xki + XMi);
                Xor = 0.5 * (Xkr - XMr); Xoi = 0.5 * (Xki - XMi);
            } else {
                double Xkr = Xre[k], Xki = Xim[k];
                double cjr = Xre[m - k], cji = -Xim[m - k];
                Xer = 0.5 * (Xkr + cjr); Xei = 0.5 * (Xki + cji);
                double dr = Xkr - cjr, di = Xki - cji;
                double ang = -2.0 * Math.PI * k / n;
                double twr = Math.cos(ang), twi = Math.sin(ang);
                Xor = 0.5 * (dr * twr + di * twi);
                Xoi = 0.5 * (di * twr - dr * twi);
            }
            Zre[k] = Xer - Xoi;
            Zim[k] = Xei + Xor;
        }
        fft(Zre, Zim, true);
        double[] x = new double[n];
        for (int k = 0; k < m; k++) { x[2 * k] = Zre[k]; x[2 * k + 1] = Zim[k]; }
        return x;
    }

    /** The kernel's rfft spectrum, computed once per solve and reused across all steps. */
    private static class KernelSpectrum {
        final double[] re, im;
        final int fsize, kernelLen;
        KernelSpectrum(double[] re, double[] im, int fsize, int kernelLen) {
            this.re = re; this.im = im; this.fsize = fsize; this.kernelLen = kernelLen;
        }
    }

    private static KernelSpectrum makeKernelSpectrum(double[] kernel, int signalLen) {
        int size = signalLen + kernel.length - 1;
        int fsize = nextPow2(size);
        double[] kpad = new double[fsize];
        System.arraycopy(kernel, 0, kpad, 0, kernel.length);
        double[][] K = rfft(kpad);
        return new KernelSpectrum(K[0], K[1], fsize, kernel.length);
    }

    /**
     * Linear convolution of F with the kernel behind `ks`, via FFT, cropped
     * to len(F) and aligned so the kernel's center index lines up with each
     * F index -- matches numpy's fft_convolve_same in fft_diffusion.py, but
     * using the precomputed kernel spectrum and real-input FFTs.
     */
    private static double[] fftConvolveSame(double[] F, KernelSpectrum ks) {
        int n = F.length;
        double[] Fpad = new double[ks.fsize];
        System.arraycopy(F, 0, Fpad, 0, n);
        double[][] Fs = rfft(Fpad);
        double[] Fre = Fs[0], Fim = Fs[1];

        int M = ks.fsize / 2;
        double[] Pre = new double[M + 1], Pim = new double[M + 1];
        for (int k = 0; k <= M; k++) {
            Pre[k] = Fre[k] * ks.re[k] - Fim[k] * ks.im[k];
            Pim[k] = Fre[k] * ks.im[k] + Fim[k] * ks.re[k];
        }
        double[] convFull = irfft(Pre, Pim, ks.fsize);

        int start = (ks.kernelLen - 1) / 2;
        double[] result = new double[n];
        System.arraycopy(convFull, start, result, 0, n);
        return result;
    }

    /**
     * Discretized N(mean, var) kernel on grid spacing dr, normalized so the
     * discrete Riemann sum equals 1 (probability-preserving).
     */
    static double[] gaussianKernel(double dr, double mean, double var, double nStd) {
        double std = Math.sqrt(var);
        double halfWidth = Math.max(nStd * std, dr);
        int nPts = Math.max(1, (int) Math.ceil(halfWidth / dr));
        int len = 2 * nPts + 1;
        double[] kernel = new double[len];
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            double offset = (i - nPts) * dr;
            double d = offset - mean;
            double val = Math.exp(-0.5 * d * d / var);
            kernel[i] = val;
            sum += val;
        }
        for (int i = 0; i < len; i++) kernel[i] /= sum;
        return kernel;
    }

    static double[] gaussianKernel(double dr, double mean, double var) {
        return gaussianKernel(dr, mean, var, 8.0);
    }

    /** exp(Simpson-integrated c(r,tau) dtau) over [tauO,tauN], per r grid point. */
    private static double[] reactionDecay(double[] rGrid, double tauO, double tauN,
                                           DoubleUnaryOperator lam, double mu) {
        double tauM = 0.5 * (tauO + tauN);
        int Nr = rGrid.length;
        double[] result = new double[Nr];
        for (int i = 0; i < Nr; i++) {
            double r = rGrid[i];
            double cO = SubstitutionModel.cFunc(r, tauO, lam, mu);
            double cM = SubstitutionModel.cFunc(r, tauM, lam, mu);
            double cN = SubstitutionModel.cFunc(r, tauN, lam, mu);
            result[i] = Math.exp(((tauN - tauO) / 6.0) * (cO + 4 * cM + cN));
        }
        return result;
    }

    /** Strang-split full-step propagator: half-reaction, FFT diffusion, half-reaction. */
    private static double[] propagate(double[] F, double tauO, double tauN, KernelSpectrum ks,
                                       double[] rGrid, DoubleUnaryOperator lam, double mu) {
        double tauH = 0.5 * (tauO + tauN);
        int Nr = F.length;

        double[] decay1 = reactionDecay(rGrid, tauO, tauH, lam, mu);
        double[] fh = new double[Nr];
        for (int i = 0; i < Nr; i++) fh[i] = F[i] * decay1[i];

        double[] fc = fftConvolveSame(fh, ks);

        double[] decay2 = reactionDecay(rGrid, tauH, tauN, lam, mu);
        double[] fOut = new double[Nr];
        for (int i = 0; i < Nr; i++) fOut[i] = fc[i] * decay2[i];
        return fOut;
    }

    /**
     * One step of dM/dtau = L(tau)[M] + S(tau), via the second-order
     * (exponential/Lawson-trapezoidal) Duhamel treatment of the source term.
     * Pass srcO = srcN = null for the homogeneous (no-source) case.
     */
    private static double[] step(double[] F, double tauO, double tauN,
                                  double[] srcO, double[] srcN, double dtau,
                                  KernelSpectrum ks, double[] rGrid,
                                  DoubleUnaryOperator lam, double mu) {
        double[] fNew = propagate(F, tauO, tauN, ks, rGrid, lam, mu);
        if (srcO != null) {
            double[] srcOProp = propagate(srcO, tauO, tauN, ks, rGrid, lam, mu);
            for (int i = 0; i < fNew.length; i++) {
                fNew[i] += (dtau / 2.0) * (srcOProp[i] + srcN[i]);
            }
        }
        return fNew;
    }

    /**
     * Same outputs as SubstitutionModel.solveFMeanNVarN, but phi0, sigma0Sq
     * are SCALARS (constant drift/variance -- plain Brownian motion with
     * drift), and the diffusion sub-step uses FFT convolution with the exact
     * Gaussian kernel instead of Crank-Nicolson finite differences.
     */
    public static SubstitutionModel.Result solveFMeanNVarNFFT(
            double phi0, double sigma0Sq, DoubleUnaryOperator lam,
            double mu, double alpha, double tTotal,
            double rMin, double rMax, int Nr, int Nt, DoubleUnaryOperator fT) {

        double[] rGrid = new double[Nr];
        double dr = (rMax - rMin) / (Nr - 1);
        for (int i = 0; i < Nr; i++) rGrid[i] = rMin + i * dr;

        double dtau = tTotal / Nt;
        double[] tauGrid = new double[Nt + 1];
        for (int j = 0; j <= Nt; j++) tauGrid[j] = j * dtau;

        if (fT == null) fT = r -> 1.0;
        final DoubleUnaryOperator fTerm = fT;

        // Corrected sign: -phi0*dtau (see class-level note above). The kernel
        // -- and its FFT spectrum -- are the same for every step in this
        // solve, so both are built ONCE here rather than inside the loop.
        double[] kernel = gaussianKernel(dr, -phi0 * dtau, sigma0Sq * dtau);
        KernelSpectrum ks = makeKernelSpectrum(kernel, Nr);

        // --- Step 1: F(r,t) = M0, homogeneous ---
        double[][] fArr = new double[Nt + 1][Nr];
        for (int i = 0; i < Nr; i++) fArr[0][i] = fTerm.applyAsDouble(rGrid[i]);
        for (int j = 0; j < Nt; j++) {
            fArr[j + 1] = step(fArr[j], tauGrid[j], tauGrid[j + 1], null, null, dtau,
                    ks, rGrid, lam, mu);
        }

        // --- Step 2: first moment of Z (source = c2 * M0) ---
        double[][] mz1 = new double[Nt + 1][Nr];
        for (int j = 0; j < Nt; j++) {
            double tauO = tauGrid[j], tauN = tauGrid[j + 1];
            double[] srcO = new double[Nr], srcN = new double[Nr];
            for (int i = 0; i < Nr; i++) {
                srcO[i] = SubstitutionModel.c2Func(rGrid[i], tauO, lam, mu, alpha) * fArr[j][i];
                srcN[i] = SubstitutionModel.c2Func(rGrid[i], tauN, lam, mu, alpha) * fArr[j + 1][i];
            }
            mz1[j + 1] = step(mz1[j], tauO, tauN, srcO, srcN, dtau, ks, rGrid, lam, mu);
        }

        // --- Step 3: second raw moment of Z (source = 2 * c2 * MZ1) ---
        double[][] mz2 = new double[Nt + 1][Nr];
        for (int j = 0; j < Nt; j++) {
            double tauO = tauGrid[j], tauN = tauGrid[j + 1];
            double[] srcO = new double[Nr], srcN = new double[Nr];
            for (int i = 0; i < Nr; i++) {
                srcO[i] = 2 * SubstitutionModel.c2Func(rGrid[i], tauO, lam, mu, alpha) * mz1[j][i];
                srcN[i] = 2 * SubstitutionModel.c2Func(rGrid[i], tauN, lam, mu, alpha) * mz1[j + 1][i];
            }
            mz2[j + 1] = step(mz2[j], tauO, tauN, srcO, srcN, dtau, ks, rGrid, lam, mu);
        }

        double[] fCurve = fArr[Nt];
        double[] meanNCurve = new double[Nr];
        double[] varNCurve = new double[Nr];
        for (int i = 0; i < Nr; i++) {
            double meanZ = mz1[Nt][i] / fCurve[i];
            double raw2 = mz2[Nt][i] / fCurve[i];
            double varZ = raw2 - meanZ * meanZ;
            meanNCurve[i] = meanZ;
            varNCurve[i] = meanZ + varZ;
        }

        return new SubstitutionModel.Result(rGrid, fCurve, meanNCurve, varNCurve);
    }

    // ---------------------------------------------------------------
    // Demo: compares against SubstitutionModel's Crank-Nicolson solver on
    // the same constant-drift problem, matching fft_diffusion.py's __main__.
    // ---------------------------------------------------------------
    public static void main(String[] args) {
        double phi0 = 0.3, sigma0Sq = 1.0;
        double mu = 0.3;
        DoubleUnaryOperator lam = r -> 0.25 + 0.02 * r;
        double alpha = 0.5;
        double tTotal = 2.0;
        double r0 = 5.0;
        double rMin = -10.0, rMax = 20.0;
        int Nr = 300, Nt = 400;

        DoubleUnaryOperator fT = SubstitutionModel.normalFT(r0, 1.0);

        java.util.function.DoubleBinaryOperator phiConst = (r, tau) -> phi0;
        java.util.function.DoubleBinaryOperator sigma2Const = (r, tau) -> sigma0Sq;

        long t0 = System.nanoTime();
        SubstitutionModel.Result cn = SubstitutionModel.solveFMeanNVarN(
                phiConst, sigma2Const, lam, mu, alpha, tTotal, rMin, rMax, Nr, Nt, fT);
        double tCn = (System.nanoTime() - t0) / 1e9;

        t0 = System.nanoTime();
        SubstitutionModel.Result fft = solveFMeanNVarNFFT(
                phi0, sigma0Sq, lam, mu, alpha, tTotal, rMin, rMax, Nr, Nt, fT);
        double tFft = (System.nanoTime() - t0) / 1e9;

        System.out.printf("%-18s%12s%12s%12s%n", "method", "time(s)", "mean(N)", "var(N)");
        System.out.printf("%-18s%12.4f%12.4f%12.4f%n", "Crank-Nicolson", tCn,
                SubstitutionModel.interp(r0, cn.rGrid, cn.meanNCurve),
                SubstitutionModel.interp(r0, cn.rGrid, cn.varNCurve));
        System.out.printf("%-18s%12.4f%12.4f%12.4f%n", "FFT convolution", tFft,
                SubstitutionModel.interp(r0, fft.rGrid, fft.meanNCurve),
                SubstitutionModel.interp(r0, fft.rGrid, fft.varNCurve));
        System.out.printf("%nspeedup (CN/FFT): %.2fx%n", tCn / tFft);

        double maxDiff = 0.0;
        for (int i = 0; i < Nr; i++) maxDiff = Math.max(maxDiff, Math.abs(cn.fCurve[i] - fft.fCurve[i]));
        System.out.printf("max |F_cn - F_fft| across grid: %.2e%n", maxDiff);
    }
}
