package mosse;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.CalculationNode;

/**
 * This class implements a type of link function that is used by QuaSSE to
 * convert an x value (continuous-trait value) into a y value (macroevolutionary
 * parameter, e.g., birth rate)
 *
 * @author Fabio K. Mendes
 */

@Description("Logistic link function for converting x into y, "
		+ "where x is a continuous trait and y is a macroevolutionary" + "parameter.")
public class LogisticFunction extends CalculationNode implements LinkFn {

	final public Input<RealParameter> curveYBaseValueInput = new Input<>("curveYBaseValue", "Curve y base value.",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> curveMaxYInput = new Input<>("curveMaxY", "Curve maximum y value.",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> sigmoidMidpointInput = new Input<>("sigmoidMidpoint",
			"Midpoint of sigmoid curve.", Input.Validate.REQUIRED);
	final public Input<RealParameter> logisticGrowthRateInput = new Input<>("logisticGrowthRate",
			"Growth rate of logistic curve.", Input.Validate.REQUIRED);

	private double y0, y1, x0, r;
	private static final String LINKFUNCTION = "logistic";

	@Override
	public void initAndValidate() {
		y0 = curveYBaseValueInput.get().getValue();
		y1 = curveMaxYInput.get().getValue();
		r = logisticGrowthRateInput.get().getValue();
		x0 = sigmoidMidpointInput.get().getValue();
	}

	@Override
	public boolean refreshParams() {

		boolean refreshedSomething = false;

		if (curveYBaseValueInput.get().somethingIsDirty()) {
			y0 = curveYBaseValueInput.get().getValue();
			refreshedSomething = true;
		}

		if (curveMaxYInput.get().somethingIsDirty()) {
			y1 = curveMaxYInput.get().getValue();
			refreshedSomething = true;
		}

		if (logisticGrowthRateInput.get().somethingIsDirty()) {
			r = logisticGrowthRateInput.get().getValue();
			refreshedSomething = true;
		}

		if (sigmoidMidpointInput.get().somethingIsDirty()) {
			x0 = sigmoidMidpointInput.get().getValue();
			refreshedSomething = true;
		}

		return refreshedSomething;
	}

	@Override
	public double[] getY(double[] x, double[] y) {
		if (x.length != y.length) {
			throw new RuntimeException("Sizes of x (qu trait) and y (macroevol param) differ. Exiting...");
		}

		for (int i = 0; i < x.length; i++) {
			double curveMaxMinusBaseValue = y1 - y0;
			y[i] = y0 + curveMaxMinusBaseValue / (1.0 + Math.exp(r * (x0 - x[i])));
		}

		return y;
	}

	@Override
	public String getLinkFnName() {
		return LINKFUNCTION;
	}

	@Override
	public void printParams() {
		// y0, y1, x0, r;
		System.out.println("y0 = " + y0 + "; y1 = " + y1 + "; x0 = " + x0 + "; r = " + r);
	}

	@Override
	protected boolean requiresRecalculation() {
		refreshParams();
		return true;
	}
}
