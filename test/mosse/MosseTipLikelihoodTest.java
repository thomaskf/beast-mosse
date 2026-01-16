package mosse;

/**
 * @author Kylie Chen
 */

import static junit.framework.Assert.assertEquals;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Test;

import beast.base.inference.parameter.RealParameter;

public class MosseTipLikelihoodTest {

    private static double DELTA = 1e-10;

    /**
     * tests the tip likelihood in a large area integrates to 1.0
     */
    @Test
    public void testMosseTipLikelihoodLargeIntervalIsOne() {
        double beta0 = 0.1;
        double beta1 = 0.2;
        double subst = 0.01;
        double epsilon = 0.01;

        double[] traits = {1.0, 1.0};
        double a = -5.0;
        double b = 5.0;

        Double[] betasArray = {beta0, beta1};
        RealParameter betas = new RealParameter(betasArray);

        MosseTipLikelihood tipLikelihood = new MosseTipLikelihood();
        tipLikelihood.initByName(
                "beta", betas,
                "subst", Double.toString(subst),
                "epsilon", Double.toString(epsilon),
                "logscale", "false"
        );
        tipLikelihood.initAndValidate();

		double mean = tipLikelihood.meanSubstitution.getValue();
		for (int i = 0; i < traits.length; i++) {
			int numBetas = tipLikelihood.beta.getDimension();
			if (numBetas != traits.length) {
				throw new IllegalArgumentException("beta dimension not equal to trait dimension!");
			}
			mean += tipLikelihood.beta.getValue(i) * traits[i];
		}
		double sd = tipLikelihood.epsilon.getValue();
		NormalDistribution normalDist = new NormalDistribution(mean, sd);
        double prob = tipLikelihood.getTipLikelihood(a, b, normalDist);

        assertEquals(1.0, prob, DELTA);
    }
}