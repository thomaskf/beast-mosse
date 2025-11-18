package mosse;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.State;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.parameter.BooleanParameter;
import beast.base.evolution.tree.TreeDistribution;

import java.util.List;
import java.util.Random;
import java.lang.UnsupportedOperationException;


/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse tree model")
public class MosseDistribution extends TreeDistribution {

    final public Input<IntegerParameter> nxInput = new Input<>("nx", "number of bins for substitution rate", new IntegerParameter("1024"));
    final public Input<RealParameter> dxInput = new Input<>("dx", "distance between xs", new RealParameter("0.0001"));
    final public Input<RealParameter> driftInput = new Input<>("drift", "drift parameter", new RealParameter("0.0"));
    final public Input<RealParameter> diffusionInput = new Input<>("diffusion", "diffusion parameter", new RealParameter("0.001"));
    final public Input<RealParameter> dtInput = new Input<>("dt", "time interval dt", new RealParameter("0.01"));
    final public Input<IntegerParameter> widthInput = new Input<>("width", "width of the kernel for convolution", new IntegerParameter("10"));
    final public Input<IntegerParameter> resolutionInput = new Input<>("resolution", "scale factor for resolution of bins", new IntegerParameter("4"));
    final public Input<BooleanParameter> lowresolutionInput = new Input<>("lowresolution", "whether using low resolution", new BooleanParameter("false"));


    final public int FLAG_FFTW3_DEFAULT = 0;


    static {
        System.loadLibrary("test");
    }

    /**
     * initialize mosse C object
     * @param nx number of bins for substitution rate
     * @param dx distance between xs
     * @param array_nd plan dimensions for FFTW3 integration
     * @param flags flags for FFTW3 integration
     * @return mosse object pointer
     */
    private native long makeMosseFFT(int nx, double dx, int[] array_nd, int flags);

    /**
     * destroy mosse C object
     * @param obj_ptr mosse object pointer
     */
    private native void mosseFinalize(long obj_ptr);

    private native double[] doIntegrateMosse(long obj_ptr, double[] vars, double[] lambda, double[] mu,
            double drift, double diffusion, double[] Q, int nt, double dt, int pad_left, int pad_right);

    public void initAndValidate() {

    }

    public int getPadLeft(boolean lowResolution) {
        double mean = driftInput.get().getValue() * dtInput.get().getValue();
        double sd = Math.sqrt(diffusionInput.get().getValue() * dtInput.get().getValue());
        double dx = dxInput.get().getValue();
        int width = widthInput.get().getValue();
        int resolution = resolutionInput.get().getValue();
        int padLeft = 0;
        if (lowResolution) {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx / resolution);
        } else {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx / resolution) * resolution;
        }
        return Math.abs(padLeft);
    }

    public int getPadRight(boolean lowResolution) {
        double mean = driftInput.get().getValue() * dtInput.get().getValue();
        double sd = Math.sqrt(diffusionInput.get().getValue() * dtInput.get().getValue());
        double dx = dxInput.get().getValue();
        int width = widthInput.get().getValue();
        int resolution = resolutionInput.get().getValue();
        int padRight = 0;
        if (lowResolution) {
            padRight = (int) Math.ceil((mean + width * sd) / dx / resolution);
        } else {
            padRight = (int) Math.ceil((mean + width * sd) / dx / resolution) * resolution;
        }
        return Math.abs(padRight);
    }

    public void calculateBranchLogP(double branchTime, double[] vars, double[] lambda, double[] mu, double[] Q, double[] result, boolean lowResolution) {
        // double logP = 0.0;
        // getting parameter values
        int nx = nxInput.get().getValue();
        double dx = dxInput.get().getValue();
        int[] nd = {5};
        double drift = driftInput.get().getValue();
        double diffusion = diffusionInput.get().getValue();
        double dt = dtInput.get().getValue();
        int resolution = resolutionInput.get().getValue();
        
        // consider this special case: 
        // Consider branchTime = 0.3000001; dt = 0.01;
        // let x = branchTime/dt 
        // then x = 30.00001, and Math.ceil(x) would be 31
        // x need to be greater than say 30 + delta so that nt would be 31
        double delta = 0.1;
        int nt = (int) Math.ceil(branchTime / dt - delta);
        
        if (lowResolution) {
        	dx = dx * resolution;
        } else {
        	nx = nx * resolution;
        }

        int padLeft = getPadLeft(lowResolution);
        int padRight = getPadRight(lowResolution);

        double[] the_result = doIntegration(nx, dx, nd, FLAG_FFTW3_DEFAULT,
            vars, lambda, mu,
            drift, diffusion,
            Q, nt, dt,
            padLeft, padRight);

        System.arraycopy(the_result, 0, result, 0, the_result.length);
    }

    /**
     * perform integration along a single branch
     * @param nx number of bins for substitution rate
     * @param dx distance between xs
     * @param nd plan dimensions for FFT3 integration
     * @param flags flags for FFTW3 integration
     * @param vars array of tips or partials
     * @param lambda array of birth-rates
     * @param mu array of death-rates
     * @param drift drift parameter
     * @param diffusion diffusion parameter
     * @param Q exponentiated form of the Q matrix
     * @param nt number of time steps
     * @param dt_max maximum time step
     * @param pad_left padding size left of the kernel (zero padding)
     * @param pad_right padding size right of the kernel (zero padding)
     * @param lq log-scaling
     * @return partial probabilities
     */
    public double[] doIntegration(int nx, double dx, int[] nd, int flags,
                              double[] vars, double[] lambda, double[] mu,
                              double drift, double diffusion,
                              double[] Q, int nt, double dt_max,
                              int pad_left, int pad_right) {
    	
        // make mosse fft object pointer
        long ptr = makeMosseFFT(nx, dx, nd, flags);
        
        // integrate using C propagate x and propagate t
        double[] result = doIntegrateMosse(ptr, vars, lambda, mu, drift, diffusion, Q, nt, dt_max, pad_left, pad_right);
        
        mosseFinalize(ptr); // destroy obj pointer
        return result; // return non logged results
    }

    @Override
    public List<String> getArguments() {
        return null;
    }

    @Override
    public List<String> getConditions() {
        return null;
    }

    @Override
    public void sample(State state, Random random) {
        throw new UnsupportedOperationException();
    }

}
