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

    public int getPadLeft() {
        double mean = driftInput.get().getValue() * dtInput.get().getValue();
        double sd = Math.sqrt(diffusionInput.get().getValue() * dtInput.get().getValue());
        double dx = dxInput.get().getValue();
        int width = widthInput.get().getValue();
        int resolution = resolutionInput.get().getValue();
        boolean lowResolution = lowresolutionInput.get().getValue();
        int padLeft = 0;
        if (lowResolution) {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx);
        } else {
            padLeft = (int) Math.ceil(-(mean - width * sd) / dx / resolution) * resolution;
        }
        System.out.println("mean = " + mean + "; width = " + width + "; sd = " + sd + "; dx = " + dx + "; resolution = " + resolution + ";padleft = " + padLeft);
        return Math.abs(padLeft);
    }

    public int getPadRight() {
        double mean = driftInput.get().getValue() * dtInput.get().getValue();
        double sd = Math.sqrt(diffusionInput.get().getValue() * dtInput.get().getValue());
        double dx = dxInput.get().getValue();
        int width = widthInput.get().getValue();
        int resolution = resolutionInput.get().getValue();
        boolean lowResolution = lowresolutionInput.get().getValue();
        int padRight = 0;
        if (lowResolution) {
            padRight = (int) Math.ceil((mean + width * sd) / dx);
        } else {
            padRight = (int) Math.ceil((mean + width * sd) / dx / resolution) * resolution;
        }
        return Math.abs(padRight);
    }

    public double calculateBranchLogP(double branchTime, double[] vars, double[] lambda, double[] mu, double[] Q, double[] result) {
        // double logP = 0.0;
        // getting parameter values
        int nx = nxInput.get().getValue();
        double dx = dxInput.get().getValue();
        int[] nd = {5};
        double drift = driftInput.get().getValue();
        double diffusion = diffusionInput.get().getValue();
        double dt = dtInput.get().getValue();
        int nt = (int) Math.ceil(branchTime / dt);

        int padLeft = getPadLeft();
        int padRight = getPadRight();
        double[] lq = {0.0}; // log scaling for vars

        double[] the_result = doIntegration(nx, dx, nd, FLAG_FFTW3_DEFAULT,
            vars, lambda, mu,
            drift, diffusion,
            Q, nt, dt,
            padLeft, padRight, lq);
        
        System.arraycopy(the_result, 0, result, 0, the_result.length);
        
        /*
        int ncol = vars.length / nx; // 5 dimensions
        double[][] ans = new double[nx][ncol];
        logP = calculateBranchLogP(result, nx, ncol, dx, ans, lq);
        
        System.out.println("logP = " + logP);
        System.out.println("After log compensation");
        System.out.println("*result's length = " + result.length);
        System.out.print("*result:");
        int kk = 0;
        for (int i = 0; i < 5; i++) {
        	for (int j = 0; j < nx; j++) {
        		System.out.print(" " + result[kk]);
        		kk++;
        	}
        	System.out.println();
        }
        System.out.println();
        */
        return lq[0];
    }

    /**
     * calculate the log probability on a single branch
     * @param array input non logged partials from doIntegration()
     * @param nx number of bins for substitution rate
     * @param ncol number of columns
     * @param dx distance between xs
     * @param ans array for storing logged result
     * @param lq log-scaling for vars
     * @return log probability
     */
    public double calculateBranchLogP(double[] array, int nx, int ncol, double dx, double[][] ans, double[] lq) {
    	
    	/*
    	System.out.println("The ans from intergation:");
    	for (int i = 0; i < nx; i++) {
    		for (int j = 0; j < ncol; j++) {
    			int index = j * nx + i;
    			System.out.print(" " + array[index]);
    		}
    		System.out.println();
    	}
    	*/
    	
    	System.out.println("[calculateBranchLogP] lq = " + lq[0]);
    	
        double logP = logCompensation(nx, ncol, dx, array, ans) + lq[0];
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
                              int pad_left, int pad_right, double[] lq) {
        // make mosse fft object pointer
    	
    	System.out.print("Before calling makeMosseFFT, nx = " + nx + "; dx = " + dx + "; flags = " + flags + "; nd = {");
    	for (int i = 0; i < nd.length; i++)
    		System.out.print(" " + nd[i]);
    	System.out.println("}");
    	
        long ptr = makeMosseFFT(nx, dx, nd, flags);
        // integrate using C propagate x and propagate t
        
        // normalize the values of vars
        // ignore the first nx entries (i.e. first row)
        double vsum = 0.0;
        for (int i = nx; i < vars.length; i++)
        	vsum += (vars[i] * dx);
        for (int i = nx; i < vars.length; i++)
        	vars[i] /= vsum;
        // lq[0] = Math.log(vsum); // log scaling
        System.out.println("[doIntegration] vsum = " + vsum + "; lq = " + lq[0]);
        System.out.println("vars.length = " + vars.length);
        /*
        for (int i = 0; i < vars.length; i++)
        	if (vars[i] > 0.0)
        		System.out.println("vars[" + i + "] = " + vars[i]);
        */
        System.out.println("lambda.length = " + lambda.length);
        System.out.print("lambda =");
        for (int i = 0; i < 3; i++)
        	System.out.print(" " + lambda[i]);
        System.out.println();

        System.out.println("mu.length = " + mu.length);
        System.out.print("mu =");
        for (int i = 0; i < 3; i++)
        	System.out.print(" " + mu[i]);
        System.out.println();

        System.out.println("drift = " + drift);
        
        System.out.println("diffusion = " + diffusion);

        System.out.println("Q.length = " + Q.length);
        System.out.print("Q =");
        for (int i = 0; i < 3; i++)
        	System.out.print(" " + Q[i]);
        System.out.println();

        System.out.println("nt = " + nt);
        System.out.println("dt_max = " + dt_max);
        System.out.println("pad_left = " + pad_left);
        System.out.println("pad_right = " + pad_right);

        double[] result = doIntegrateMosse(ptr, vars, lambda, mu, drift, diffusion, Q, nt, dt_max, pad_left, pad_right);
        
        // log compensation
        // ignore the first nx entries (i.e. first row)
        vsum = 0.0;
        for (int i = nx; i < result.length; i++)
        	vsum += (result[i] * dx);
        for (int i = nx; i < result.length; i++)
        	result[i] /= vsum;
        lq[0] = Math.log(vsum); // log scaling
        
        System.out.println("result's length = " + result.length);
        
        System.out.print("result:");
        int kk = 0;
        for (int i = 0; i < 5; i++) {
        	for (int j = 0; j < nx; j++) {
        		System.out.print(" " + result[kk]);
        		kk++;
        	}
        	System.out.println();
        }
        System.out.println();

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
        System.out.println("[logCompensation] ncol = " + ncol + "; nrow = " + nrow);
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
        System.out.println("[logCompensation] logP = " + logP);
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
