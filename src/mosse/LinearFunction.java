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

@Description("Linear function for converting x into y, "
		+ "where x is a continuous trait and y is a macroevolutionary" + "parameter.")
public class LinearFunction extends CalculationNode implements LinkFn {

	final public Input<RealParameter> curveYBaseValueInput = new Input<>("curveYBaseValue", "Curve y base value.",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> curveMaxYInput = new Input<>("curveMaxY", "Curve maximum y value.",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> logisticGrowthRateInput = new Input<>("logisticGrowthRate",
			"Growth rate of logistic curve.", Input.Validate.REQUIRED);

	private double y0, y1, r;
	private static final String LINKFUNCTION = "linear";

	@Override
	public void initAndValidate() {
		y0 = curveYBaseValueInput.get().getValue();
		y1 = curveMaxYInput.get().getValue();
		r = logisticGrowthRateInput.get().getValue();
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

		return refreshedSomething;
	}

	@Override
	public double[] getY(double[] x, double[] y) {
		if (x.length != y.length) {
			throw new RuntimeException("Sizes of x (qu trait) and y (macroevol param) differ. Exiting...");
		}

		for (int i = 0; i < x.length; i++) {
			
			if (x[i] < 0.0) {
				y[i] = y0;
			} else {
				y[i] = y0 + r * x[i];
				if (y[i] > y1) {
					y[i] = y1;
				}
			}
		}

		return y;
	}

	@Override
	public String getLinkFnName() {
		return LINKFUNCTION;
	}

	@Override
	public void printParams() {
		// y0, y1, r;
		System.out.println("y0 = " + y0 + "; y1 = " + y1 + "; r = " + r);
	}

	@Override
	protected boolean requiresRecalculation() {
		refreshParams();
		return true;
	}
}
