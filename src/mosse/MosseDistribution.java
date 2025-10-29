package mosse;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.State;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.Parameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.parameter.BooleanParameter;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeDistribution;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.lang.UnsupportedOperationException;


/**
 * @author Kylie Chen
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

        System.out.println(".");
        
        System.out.println("resolution = " + resolution);
        System.out.println("nx = " + nx);
        System.out.println("dx = " + dx);
        System.out.print("nd =");
        for (int i = 0; i < nd.length; i++)
        	System.out.print(" " + nd[i]);
        System.out.println();
        
        // show vars
        System.out.println("vars.length = " + vars.length);
        int x = vars.length / 5;
        for (int s = 0; s < vars.length; s+= x) {
	        System.out.print("vars[" + s + ":" + s+4 + "] =");
	        for (int i = 0; i < 5; i++)
	        	System.out.print(" " + vars[s+i]);
	        System.out.println();
        }
        
        // show lambda
        System.out.println("lambda.length = " + lambda.length);
        System.out.print("lambda[0:4]:");
        for (int i = 0; i < 5; i++)
        	System.out.print(" " + lambda[i]);
        System.out.println();
        
        // show mu
        System.out.println("mu.length = " + mu.length);
        System.out.print("mu[0:4]:");
        for (int i = 0; i < 5; i++)
        	System.out.print(" " + mu[i]);
        System.out.println();
        
        System.out.println("drift = " + drift);
        System.out.println("diffusion = " + diffusion);
        System.out.println("Q.length = " + Q.length);
        System.out.println("nt = " + nt);
        System.out.println("dt = " + dt);
        System.out.println("padLeft = " + padLeft);
        System.out.println("padRight = " + padRight);
        
        System.out.println(".");

        double[] the_result = doIntegration(nx, dx, nd, FLAG_FFTW3_DEFAULT,
            vars, lambda, mu,
            drift, diffusion,
            Q, nt, dt,
            padLeft, padRight);

        System.arraycopy(the_result, 0, result, 0, the_result.length);
    }

    /**
     * calculate the log probability on a single branch
     * @param array input non logged partials from doIntegration()
     * @param nx number of bins for substitution rate
     * @param ncol number of columns
     * @param dx distance between xs
     * @param ans array for storing logged result
     * @return log probability
     */
    public double calculateBranchLogP(double[] array, int nx, int ncol, double dx, double[][] ans) {
    	
        double logP = logCompensation(nx, ncol, dx, array, ans);
        return logP; // return log compensated result
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
    	
    	System.out.println("calling the function makeMosseFFT...");
    	System.out.print("nx = " + nx + "; dx = " + dx + "; nd =");
    	for (int i = 0; i < nd.length; i++)
    		System.out.print(" " + nd[i]);
    	System.out.println("; flags = " + flags);
    	
        // make mosse fft object pointer
        long ptr = makeMosseFFT(nx, dx, nd, flags);
        
        System.out.println("calling the function doIntegrateMosse...");
        System.out.println("vars.length = " + vars.length + "; lambda.length = " + lambda.length + "; mu.length = " + mu.length + 
        		"; drift = " + drift + "; diffusion = " + diffusion + "; Q.length = " + Q.length + "; nt = " + nt + 
        		"; dt_max = " + dt_max + "; pad_left = " + pad_left + "; pad_right = " + pad_right);

        // integrate using C propagate x and propagate t
        double[] result = doIntegrateMosse(ptr, vars, lambda, mu, drift, diffusion, Q, nt, dt_max, pad_left, pad_right);
        
        mosseFinalize(ptr); // destroy obj pointer
        return result; // return non logged results
    }

    /**
     *
     * @param nrow number of rows
     * @param ncol number of columns
     * @param dx distance between xs
     * @param result non-logged partial probabilities
     * @param ans logged partial probabilities
     * @return logged partial probabilities
     */
    public double logCompensation(int nrow, int ncol, double dx, double[] result, double[][] ans) {
        double logP = 0.0;
        int count = 0;
        for (int j = 0; j < ncol; j++) {
            for (int i = 0; i < nrow; i++) {
                ans[i][j] = result[count];
                count++;
            }
        }
        if (ncol > 1 ) {
            double sum = 0.0;
            // sum except first col
            for (int i = 0; i < nrow; i++) {
                for (int j = 1; j < ncol; j++) {
                    sum += ans[i][j] * dx;
                }
            }
            double q = sum;
            // update ans except first col
            for (int i = 0; i < nrow; i++) {
                for (int j = 1; j < ncol; j++) {
                    ans[i][j] = ans[i][j] / q;
                }
            }
            logP = Math.log(q);
        } else {
            logP = 0.0;
        }
        return logP;
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
