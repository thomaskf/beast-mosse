package mosse;

import org.apache.commons.math3.distribution.NormalDistribution;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;
import beast.base.inference.parameter.BooleanParameter;
import beast.base.inference.parameter.RealParameter;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse tip likelihood using regression model of tip rate given traits and parameters beta and epsilon")
public class MosseTipLikelihood extends CalculationNode {

	final public Input<RealParameter> betaInput = new Input<>("beta", "beta coefficients for each trait",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> meanSubstitutionInput = new Input<>("subst", "mean substitution value",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> epsilonInput = new Input<>("epsilon", "error term of regression model",
			Input.Validate.REQUIRED);
	final public Input<BooleanParameter> logScaleInput = new Input<>("logscale",
			"whether to use log scale for substitution rate (defaults to false)", Input.Validate.OPTIONAL);

	public RealParameter beta;
	public RealParameter meanSubstitution;
	public RealParameter epsilon;

	public MosseTipLikelihood() {

	}

	/**
	 * Returns P(tip rate | beta0, beta1, epsilon, trait0, trait1) ~ Gaussian(mean =
	 * beta0 * trait0 + beta1 * trait1 + ... + mean, sd = epsilon) within the tip
	 * rate interval (a,b)
	 *
	 * @param a      start value of tip rate interval
	 * @param b      end value of tip rate interval
	 * @return tip likelihood between intervals a and b
	 */
	public double getTipLikelihood(double a, double b, NormalDistribution normalDist) {
		double startProb = normalDist.cumulativeProbability(a);
		double endProb = normalDist.cumulativeProbability(b);
		return endProb - startProb;
	}

	/**
	 *
	 * @param traits        array of trait values
	 * @param numBins       number of bins for substitution rate discretization
	 * @param startSubsRate substitution rate lower bound for rate bins
	 * @param subsInterval  substitution rate interval for rate bins
	 * @return array of tip likelihoods
	 */
	public double[] getTipLikelihoods(double[] traits, int numBins, double startSubsRate, double subsInterval) {
		if (beta.getDimension() != traits.length) {
			throw new IllegalArgumentException("beta dimension not equal to trait dimension!");
		}
		double mean = meanSubstitution.getValue();
		for (int i = 0; i < traits.length; i++) {
			mean += beta.getValue(i) * traits[i];
		}
		double sd = epsilon.getValue();
		NormalDistribution normalDist = new NormalDistribution(mean, sd);
		double[] tipLikelihoods = new double[numBins];
		for (int i = 0; i < numBins; i++) {
			double x = startSubsRate + i * subsInterval;
			tipLikelihoods[i] = normalDist.density(x) * subsInterval;
		}
		// TODO make logscale consistent with TreeLikelihood

		return tipLikelihoods;
	}

	
	/**
	 *
	 * @param traits        array of trait values
	 * @param numBins       number of bins for substitution rate discretization
	 * @param startSubsRate substitution rate lower bound for rate bins
	 * @param subsInterval  substitution rate interval for rate bins
	 * @return array of tip likelihoods
	 */
	public double[] getTipLikelihoods_prob(double[] traits, int numBins, double startSubsRate, double subsInterval) {
		if (beta.getDimension() != traits.length) {
			throw new IllegalArgumentException("beta dimension not equal to trait dimension!");
		}
		double mean = meanSubstitution.getValue();
		for (int i = 0; i < traits.length; i++) {
			mean += beta.getValue(i) * traits[i];
		}
		double sd = epsilon.getValue();
		NormalDistribution normalDist = new NormalDistribution(mean, sd);
		double[] tipLikelihoods = new double[numBins];
		for (int i = 0; i < numBins; i++) {
			double a = startSubsRate + i * subsInterval;
			double b = startSubsRate + (i + 1) * subsInterval;
			// area between a and b under normal distribution
			tipLikelihoods[i] = getTipLikelihood(a, b, normalDist);
		}
		// TODO make logscale consistent with TreeLikelihood

		return tipLikelihoods;
	}

	/**
	 *
	 * @param traits        array of trait values
	 * @param numBins       number of bins for substitution rate discretization
	 * @param startSubsRate substitution rate lower bound
	 * @param endSubsRate   substitution rate upper bound
	 * @return array of tip likelihoods
	 */
	public double[] getTipLikelihoods2_prob(double[] traits, int numBins, double startSubsRate, double endSubsRate) {
		if (beta.getDimension() != traits.length) {
			throw new IllegalArgumentException("beta dimension not equal to trait dimension!");
		}
		double mean = meanSubstitution.getValue();
		for (int i = 0; i < traits.length; i++) {
			mean += beta.getValue(i) * traits[i];
		}
		double sd = epsilon.getValue();
		NormalDistribution normalDist = new NormalDistribution(mean, sd);
		double subsInterval = (endSubsRate - startSubsRate) / numBins;
		double[] tipLikelihoods = new double[numBins];
		for (int i = 0; i < numBins; i++) {
			double a = startSubsRate + i * subsInterval;
			double b = startSubsRate + (i + 1) * subsInterval;
			tipLikelihoods[i] = getTipLikelihood(a, b, normalDist);
		}
		return tipLikelihoods;
	}

	@Override
	public void initAndValidate() {
		beta = betaInput.get();
		meanSubstitution = meanSubstitutionInput.get();
		epsilon = epsilonInput.get();
	}
	
	public void printParams() {
		System.out.print("beta =");
		for (int i = 0; i< beta.getDimension(); i++) {
			System.out.print(" " + beta.getValue(i));
		}
		System.out.println();
		System.out.println("epsilon = " + epsilon.getValue());
		System.out.println("meanSubstitution = " + meanSubstitution.getValue());
	}
	
	public boolean betaOutOfRange(double bmin, double bmax) {
		RealParameter betas = betaInput.get();
		for (int i = 0; i < betas.getDimension(); i++) {
			if (betas.getValue(i) < bmin || betas.getValue(i) > bmax) {
				return true;
			}
		}
		return false;
	}
}