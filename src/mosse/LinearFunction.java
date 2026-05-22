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
	final public Input<RealParameter> curveMaxYInput = new Input<>("curveMaxY", "Curve maximum y value (cap).",
			Input.Validate.REQUIRED);
	final public Input<RealParameter> linearGrowthRateInput = new Input<>("linearGrowthRate",
			"Slope (growth rate) of the linear function.", Input.Validate.REQUIRED);

	private double y0, y1, r;
	private double storedY0, storedY1, storedR;
	private static final String LINKFUNCTION = "linear";

	@Override
	public void initAndValidate() {
		y0 = curveYBaseValueInput.get().getValue();
		y1 = curveMaxYInput.get().getValue();
		r = linearGrowthRateInput.get().getValue();
	}

	@Override
	public boolean refreshParams() {

		boolean refreshedSomething = false;

		if (curveYBaseValueInput.get().somethingIsDirty()) {
			refreshedSomething = true;
		}
		if (curveMaxYInput.get().somethingIsDirty()) {
			refreshedSomething = true;
		}
		if (linearGrowthRateInput.get().somethingIsDirty()) {
			refreshedSomething = true;
		}
		// Always re-read all three values so the cache stays consistent even
		// if only a subset of the inputs is dirty in this proposal cycle.
		y0 = curveYBaseValueInput.get().getValue();
		y1 = curveMaxYInput.get().getValue();
		r  = linearGrowthRateInput.get().getValue();

		return refreshedSomething;
	}

	// store/restore the cached y0, y1, r so a rejected proposal does not leave
	// the cache pointing at the rejected values (refreshParams() is only invoked
	// when an input is dirty, so without explicit restore the cache can be stale
	// until the same input is proposed again).
	@Override
	public void store() {
		super.store();
		storedY0 = y0;
		storedY1 = y1;
		storedR  = r;
	}

	@Override
	public void restore() {
		super.restore();
		y0 = storedY0;
		y1 = storedY1;
		r  = storedR;
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
