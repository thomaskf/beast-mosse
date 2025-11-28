package mosse;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.State;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
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

    final public int FLAG_FFTW3_DEFAULT = 0;
    
    protected int resolution;
    protected int nx;
    protected double dx;
    protected int padLeft_h; // padLeft for high resolution
    protected int padLeft_l; // padLeft for low resolution
    protected int padRight_h; // padRight for high resolution
    protected int padRight_l; // padRight for low resolution
    
    protected double drift;
    protected double diffusion;
    protected double dt;
    protected double mean;
    protected double sd;
    protected int width;

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
        if (resolutionInput.get() != null) {
            resolution = resolutionInput.get().getValue();
        } else {
            resolution = resolutionInput.defaultValue.getValue();
        }
        if (nxInput.get() != null) {
            nx = nxInput.get().getValue();
        } else {
            nx = nxInput.defaultValue.getValue();
        }
        if (dxInput.get() != null) {
            dx = dxInput.get().getValue();
        } else {
            dx = dxInput.defaultValue.getValue();
        }
        
        drift = driftInput.get().getValue();
        diffusion = diffusionInput.get().getValue();
        dt = dtInput.get().getValue();
        
        mean = drift * dt;
        sd = Math.sqrt(diffusion * dt);
        width = widthInput.get().getValue();
        
        padLeft_h = getPadLeft(false); // padLeft for high resolution
        padLeft_l = getPadLeft(true); // padLeft for low resolution
        padRight_h = getPadRight(false); // padRight for high resolution
        padRight_l = getPadRight(true); // padRight for low resolution
    }

    public int getPadLeft(boolean lowResolution) {
        int padLeft = 0;
        if (lowResolution) {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx / resolution);
        } else {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx / resolution) * resolution;
        }
        return Math.abs(padLeft);
    }

    public int getPadRight(boolean lowResolution) {
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
        int[] nd = {5};
        
        // consider this special case: 
        // Consider branchTime = 0.3000001; dt = 0.01;
        // let x = branchTime/dt 
        // then x = 30.00001, and Math.ceil(x) would be 31
        // x need to be greater than say 30 + delta so that nt would be 31
        double delta = 0.1;
        int nt = (int) Math.ceil(branchTime / dt - delta);
        double[] the_result;
        
        if (lowResolution) {
	        the_result = doIntegration(nx, dx * resolution, nd, FLAG_FFTW3_DEFAULT,
	            vars, lambda, mu,
	            drift, diffusion,
	            Q, nt, dt,
	            padLeft_l, padRight_l);
        } else {
	        the_result = doIntegration(nx * resolution, dx, nd, FLAG_FFTW3_DEFAULT,
		        vars, lambda, mu,
		        drift, diffusion,
		        Q, nt, dt,
		        padLeft_h, padRight_h);
        }

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
