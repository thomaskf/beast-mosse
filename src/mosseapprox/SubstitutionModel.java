package mosseapprox;

import java.util.function.DoubleBinaryOperator; // (r, tau) -> value
import java.util.function.DoubleUnaryOperator;  // r -> value

/**
 * Java port of substitution_model.py.
 *
 * Computes F(r,t) -- the density of the substitution rate r at time t,
 * with the substitution-count dimension n fully marginalized out -- and
 * the mean and variance of N(t), the number of substitutions accumulated
 * by time t, as functions of r.
 *
 * Model recap
 * -----------
 * F_n(r,t): probability of observing the subtree after time t along an
 * edge, with n substitutions, given substitution rate r at time t. Q (the
 * substitution-count shift operator) is the generator of a pure Poisson
 * process, so:
 *
 *   - F(r,t) = sum_n F_n(r,t) [n marginalized out] solves a plain scalar
 *     reaction-diffusion PDE using only
 *         c(r,t) = -(lambda(r)+mu) + 2*lambda(r)*E(r,t)
 *
 *   - N(t) behaves, under this model's linearization (I + alpha*Q
 *     approximating the exact e^{alpha*Q}), exactly like a
 *     doubly-stochastic Poisson process with instantaneous rate
 *         c2(r,t) = r + 2*alpha*lambda(r)*E(r,t)
 *     so mean(N) = E[Z], var(N) = E[Z] + Var(Z), Z = integral c2(r,t) dt,
 *     computed with the SAME simple moment-hierarchy PDE used for the
 *     plain time-integral of r (no PGF/DFT machinery needed for just
 *     mean + variance).
 *
 * tau is elapsed time from the terminal/initial condition F_T(r) (tau=0)
 * to the point of interest (tau=T_total). Three real scalar PDE solves
 * total (F, first moment of Z, second moment of Z), independent of any
 * substitution-count truncation.
 */
public class SubstitutionModel {

    // ---------------------------------------------------------------
    // Tridiagonal (Thomas algorithm) solver:
    //   sub[i]*x[i-1] + main[i]*x[i] + sup[i]*x[i+1] = rhs[i]
    // ---------------------------------------------------------------
    static double[] thomasSolve(double[] sub, double[] main, double[] sup, double[] rhs) {
        int n = main.length;
        double[] cp = new double[n];
        double[] dp = new double[n];
        cp[0] = sup[0] / main[0];
        dp[0] = rhs[0] / main[0];
        for (int i = 1; i < n; i++) {
            double denom = main[i] - sub[i] * cp[i - 1];
            cp[i] = sup[i] / denom;
            dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / denom;
        }
        double[] x = new double[n];
        x[n - 1] = dp[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }
        return x;
    }

    // ---------------------------------------------------------------
    // Birth-death closed forms
    // ---------------------------------------------------------------
    static final double CRIT_EPS = 1e-8;

    /** Birth-death extinction probability by elapsed time tau, rate lambda(r). */
    static double eFunc(double r, double tau, DoubleUnaryOperator lam, double mu) {
        double l = lam.applyAsDouble(r);
        if (Math.abs(l - mu) < CRIT_EPS) {
            return tau > 0 ? mu * tau / (1 + mu * tau) : 0.0;
        }
        double eMuTau = Math.exp(mu * tau);
        double eLamTau = Math.exp(l * tau);
        return mu * (eMuTau - eLamTau) / (mu * eMuTau - l * eLamTau);
    }

    /** c(r,tau) = -(lambda(r)+mu) + 2*lambda(r)*E(r,tau), in closed form. */
    static double cFunc(double r, double tau, DoubleUnaryOperator lam, double mu) {
        double l = lam.applyAsDouble(r);
        if (Math.abs(l - mu) < CRIT_EPS) {
            return tau > 0 ? -2 * mu / (1 + mu * tau) : -2 * mu;
        }
        double eMuTau = Math.exp(mu * tau);
        double eLamTau = Math.exp(l * tau);
        double num = (l - mu) * (mu * eMuTau + l * eLamTau);
        double den = mu * eMuTau - l * eLamTau;
        return num / den;
    }

    /** c2(r,tau) = r + 2*alpha*lambda(r)*E(r,tau): total substitution rate. */
    static double c2Func(double r, double tau, DoubleUnaryOperator lam, double mu, double alpha) {
        return c2Func(r, tau, lam, mu, alpha, false);
    }

    /** logScale=true: r is log-rate, so the substitution intensity uses exp(r). */
    static double c2Func(double r, double tau, DoubleUnaryOperator lam, double mu, double alpha, boolean logScale) {
        double subRate = logScale ? Math.exp(r) : r;
        return subRate + 2 * alpha * lam.applyAsDouble(r) * eFunc(r, tau, lam, mu);
    }

    // ---------------------------------------------------------------
    // Result container
    // ---------------------------------------------------------------
    public static class Result {
        public final double[] rGrid;
        public final double[] fCurve;      // F(r, T_total)
        public final double[] meanNCurve;  // E[N(T_total) | r(0)=r]
        public final double[] varNCurve;   // Var[N(T_total) | r(0)=r]

        Result(double[] rGrid, double[] fCurve, double[] meanNCurve, double[] varNCurve) {
            this.rGrid = rGrid;
            this.fCurve = fCurve;
            this.meanNCurve = meanNCurve;
            this.varNCurve = varNCurve;
        }
    }

    /**
     * @param phi     (r, tau) -> drift of the r-diffusion
     * @param sigma2  (r, tau) -> variance rate of the r-diffusion
     * @param lam     r -> speciation rate lambda(r)
     * @param mu      extinction rate
     * @param alpha   average extra substitutions per surviving speciation event
     * @param tTotal  horizon length
     * @param rMin,rMax,Nr,Nt  spatial / temporal grid
     * @param fT      r -> terminal condition F_T(r) (pass null for all-ones)
     */
    public static Result solveFMeanNVarN(
            DoubleBinaryOperator phi, DoubleBinaryOperator sigma2, DoubleUnaryOperator lam,
            double mu, double alpha, double tTotal,
            double rMin, double rMax, int Nr, int Nt, DoubleUnaryOperator fT) {
        return solveFMeanNVarN(phi, sigma2, lam, mu, alpha, tTotal, rMin, rMax, Nr, Nt, fT, false);
    }

    /** logScale: r-grid holds log-rate; substitution intensity uses exp(r) (method A). */
    public static Result solveFMeanNVarN(
            DoubleBinaryOperator phi, DoubleBinaryOperator sigma2, DoubleUnaryOperator lam,
            double mu, double alpha, double tTotal,
            double rMin, double rMax, int Nr, int Nt, DoubleUnaryOperator fT, boolean logScale) {

        double[] rGrid = new double[Nr];
        double dr = (rMax - rMin) / (Nr - 1);
        for (int i = 0; i < Nr; i++) rGrid[i] = rMin + i * dr;

        double dtau = tTotal / Nt;
        double[] tauGrid = new double[Nt + 1];
        for (int j = 0; j <= Nt; j++) tauGrid[j] = j * dtau;

        if (fT == null) fT = r -> 1.0;
        final DoubleUnaryOperator fTerm = fT;

        // --- Step 1: F(r,t) = M0, homogeneous (no source, no n-dimension) ---
        double[][] fArr = new double[Nt + 1][Nr];
        for (int i = 0; i < Nr; i++) fArr[0][i] = fTerm.applyAsDouble(rGrid[i]);
        double[] zeros = new double[Nr];
        for (int j = 0; j < Nt; j++) {
            fArr[j + 1] = cnStep(fArr[j], tauGrid[j], tauGrid[j + 1], zeros, zeros,
                    rGrid, dr, dtau, phi, sigma2, lam, mu);
        }

        // --- Step 2: first moment of Z = integral c2 dt  (source = 1*c2*M0) ---
        // Initial condition M1(r, tau=0) = 0 for every r: `new double[...]` in
        // Java zero-fills automatically, so mz1[0] is already this all-zero row
        // -- there is no substitution accumulated yet at the terminal condition.
        double[][] mz1 = new double[Nt + 1][Nr];
        for (int j = 0; j < Nt; j++) {
            double tauO = tauGrid[j], tauN = tauGrid[j + 1];
            double[] srcO = new double[Nr], srcN = new double[Nr];
            for (int i = 0; i < Nr; i++) {
                srcO[i] = c2Func(rGrid[i], tauO, lam, mu, alpha, logScale) * fArr[j][i];
                srcN[i] = c2Func(rGrid[i], tauN, lam, mu, alpha, logScale) * fArr[j + 1][i];
            }
            mz1[j + 1] = cnStep(mz1[j], tauO, tauN, srcO, srcN, rGrid, dr, dtau, phi, sigma2, lam, mu);
        }

        // --- Step 3: second raw moment of Z  (source = 2*c2*MZ1) ---
        // Same deal: M2(r, tau=0) = 0, implicit in mz2[0] via Java's zero-fill.
        double[][] mz2 = new double[Nt + 1][Nr];
        for (int j = 0; j < Nt; j++) {
            double tauO = tauGrid[j], tauN = tauGrid[j + 1];
            double[] srcO = new double[Nr], srcN = new double[Nr];
            for (int i = 0; i < Nr; i++) {
                srcO[i] = 2 * c2Func(rGrid[i], tauO, lam, mu, alpha, logScale) * mz1[j][i];
                srcN[i] = 2 * c2Func(rGrid[i], tauN, lam, mu, alpha, logScale) * mz1[j + 1][i];
            }
            mz2[j + 1] = cnStep(mz2[j], tauO, tauN, srcO, srcN, rGrid, dr, dtau, phi, sigma2, lam, mu);
        }

        double[] fCurve = fArr[Nt];
        double[] meanNCurve = new double[Nr];
        double[] varNCurve = new double[Nr];
        for (int i = 0; i < Nr; i++) {
            double meanZ = mz1[Nt][i] / fCurve[i];
            double raw2 = mz2[Nt][i] / fCurve[i];
            double varZ = raw2 - meanZ * meanZ;
            meanNCurve[i] = meanZ;                 // E[N] = E[Z]
            varNCurve[i] = meanZ + varZ;            // Var[N] = E[Z] + Var[Z]
        }

        return new Result(rGrid, fCurve, meanNCurve, varNCurve);
    }

    /** One Crank-Nicolson step for the scalar reaction-diffusion operator. */
    private static double[] cnStep(double[] M, double tauO, double tauN,
                                    double[] srcO, double[] srcN,
                                    double[] rGrid, double dr, double dtau,
                                    DoubleBinaryOperator phi, DoubleBinaryOperator sigma2,
                                    DoubleUnaryOperator lam, double mu) {
        int Nr = rGrid.length;
        double[] subO = new double[Nr], mainO = new double[Nr], supO = new double[Nr];
        double[] subN = new double[Nr], mainN = new double[Nr], supN = new double[Nr];
        buildCoeffs(rGrid, tauO, dr, phi, sigma2, lam, mu, subO, mainO, supO);
        buildCoeffs(rGrid, tauN, dr, phi, sigma2, lam, mu, subN, mainN, supN);

        double[] rhs = M.clone();
        for (int i = 1; i < Nr - 1; i++) {
            rhs[i] += (dtau / 2) * (subO[i] * M[i - 1] + mainO[i] * M[i] + supO[i] * M[i + 1]);
            rhs[i] += (dtau / 2) * (srcO[i] + srcN[i]);
        }
        rhs[0] = 0.0;
        rhs[Nr - 1] = 0.0;

        double[] sub = new double[Nr], main = new double[Nr], sup = new double[Nr];
        for (int i = 0; i < Nr; i++) {
            sub[i] = -(dtau / 2) * subN[i];
            main[i] = 1.0 - (dtau / 2) * mainN[i];
            sup[i] = -(dtau / 2) * supN[i];
        }
        // Neumann BCs: M[0] = M[1],  M[Nr-1] = M[Nr-2]
        sub[0] = 0.0; main[0] = 1.0; sup[0] = -1.0;
        sub[Nr - 1] = -1.0; main[Nr - 1] = 1.0; sup[Nr - 1] = 0.0;

        return thomasSolve(sub, main, sup, rhs);
    }

    private static void buildCoeffs(double[] rGrid, double tau, double dr,
                                     DoubleBinaryOperator phi, DoubleBinaryOperator sigma2,
                                     DoubleUnaryOperator lam, double mu,
                                     double[] sub, double[] main, double[] sup) {
        int Nr = rGrid.length;
        for (int i = 0; i < Nr; i++) {
            double r = rGrid[i];
            double ph = phi.applyAsDouble(r, tau);
            double sg = sigma2.applyAsDouble(r, tau);
            double cc = cFunc(r, tau, lam, mu);
            sub[i] = -ph / (2 * dr) + sg / (2 * dr * dr);
            main[i] = -sg / (dr * dr) + cc;
            sup[i] = ph / (2 * dr) + sg / (2 * dr * dr);
        }
    }

    /**
     * Normal density N(r; r0, sig^2), for use as a terminal condition F_T(r)
     * -- e.g. the GLM-predicted tip substitution rate from the notes: a
     * normal distribution centered at r0 with standard deviation sig.
     */
    static DoubleUnaryOperator normalFT(double r0, double sig) {
        double norm = 1.0 / (sig * Math.sqrt(2 * Math.PI));
        return r -> {
            double z = (r - r0) / sig;
            return norm * Math.exp(-0.5 * z * z);
        };
    }

    /** Linear interpolation of `curve` (defined on rGrid) at point r0. */
    static double interp(double r0, double[] rGrid, double[] curve) {
        int n = rGrid.length;
        if (r0 <= rGrid[0]) return curve[0];
        if (r0 >= rGrid[n - 1]) return curve[n - 1];
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (rGrid[mid] <= r0) lo = mid; else hi = mid;
        }
        double t = (r0 - rGrid[lo]) / (rGrid[hi] - rGrid[lo]);
        return curve[lo] + t * (curve[hi] - curve[lo]);
    }

    // ---------------------------------------------------------------
    // Demo, matching the Python __main__ example
    // ---------------------------------------------------------------
    public static void main(String[] args) {
        double kappa = 1.0, theta = 5.0, sigma = 1.0;
        double mu = 0.3;
        DoubleUnaryOperator lam = r -> 0.25 + 0.02 * r;
        double alpha = 0.5;
        double tTotal = 2.0;
        double r0 = 5.0;
        double rMin = 0.0, rMax = 10.0;

        DoubleBinaryOperator phi = (r, tau) -> kappa * (theta - r);
        DoubleBinaryOperator sigma2 = (r, tau) -> sigma * sigma;

        // F_T(r): terminal condition (tip) as a normal distribution centered
        // at tipMean with standard deviation tipSig, e.g. from the GLM
        // prediction of tip substitution rate in the notes.
        double tipMean = 5.0, tipSig = 0.5;
        DoubleUnaryOperator fT = normalFT(tipMean, tipSig);

        Result res = solveFMeanNVarN(phi, sigma2, lam, mu, alpha, tTotal,
                rMin, rMax, 60, 80, fT);

        System.out.printf("F(r0=%.1f, T=%.1f)   = %.6f%n", r0, tTotal, interp(r0, res.rGrid, res.fCurve));
        System.out.printf("mean(N | r0=%.1f)     = %.4f%n", r0, interp(r0, res.rGrid, res.meanNCurve));
        System.out.printf("var(N | r0=%.1f)      = %.4f%n", r0, interp(r0, res.rGrid, res.varNCurve));
    }
}
