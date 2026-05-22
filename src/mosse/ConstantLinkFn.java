package mosse;

import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.CalculationNode;

/*
 * Applies the same y (macroevol param) value to all x (qu trait) bins.
 * If you put a prior on the y value, it's the same as assuming y is
 * distributed according to that prior, and independent of x
 */
public class ConstantLinkFn extends CalculationNode implements LinkFn {

	final public Input<RealParameter> yValueInput = new Input<>("yV",
			"Constant value of dependent variable (quantitative trait).", Input.Validate.REQUIRED);

	double yValue;
	private double storedYValue;
	private static final String LINKFUNCTION = "constant";

	@Override
	public void initAndValidate() {
		yValue = yValueInput.get().getValue();
	}

	@Override
	public boolean refreshParams() {

		boolean refreshedSomething = false;

		if (yValueInput.get().somethingIsDirty()) {
			refreshedSomething = true;
		}
		// Always re-read so the cache stays consistent (refreshParams() is only
		// invoked when this Input is dirty, so without this the cache can diverge
		// from the parameter after a rejected proposal).
		yValue = yValueInput.get().getValue();

		return refreshedSomething;
	}

	// store/restore so a rejected proposal does not leave the cached yValue at
	// the rejected value. Same rationale as LinearFunction's store/restore.
	@Override
	public void store() {
		super.store();
		storedYValue = yValue;
	}

	@Override
	public void restore() {
		super.restore();
		yValue = storedYValue;
	}

	@Override
	public double[] getY(double[] x, double[] y) {
		if (x.length != y.length) {
			throw new RuntimeException("Sizes of x (qu trait) and y (macroevol param) differ. Exiting...");
		}

		for (int i = 0; i < x.length; i++) {
			y[i] = yValue;
		}

		return y;
	}

	@Override
	public String getLinkFnName() {
		return LINKFUNCTION;
	}

	@Override
	public void printParams() {
		System.out.println("yValue = " + yValue);
	}

	@Override
	protected boolean requiresRecalculation() {
		refreshParams();
		return true;
	}
}
