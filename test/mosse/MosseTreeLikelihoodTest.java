package mosse;

import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.Frequencies;
import beast.base.evolution.substitutionmodel.GTR;
import beast.base.evolution.substitutionmodel.GeneralSubstitutionModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;

import org.junit.Test;
import test.beast.evolution.substmodel.GTRTest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;


/**
 * @author Kylie Chen
 */

public class MosseTreeLikelihoodTest {

    private static double DELTA = 1e-7;

    /**
     * returns an Alignment of nucleotide sequences
     * @param numLeaves number of taxa (or leaves in tree)
     * @param names name of each taxa
     * @param sequences nucleotide sequences for each taxa
     * @return an Alignment of nucleotide sequences
     */
    public Alignment getAlignment(int numLeaves, String[] names, String[] sequences) {
        List<Sequence> seqList = new ArrayList<>();
        
        assert(names.length == sequences.length);

        for (int i = 0; i < numLeaves; i++) {
            seqList.add(new Sequence(names[i], sequences[i]));
        }

        Alignment alignment = new Alignment(seqList, "nucleotide");

        return alignment;
    }

    /**
     * tests initialization of MosseTreeLikelihood does not throw any errors
     * using a simple two taxa tree (t0: 0.5, t1: 0.5) with tips "A" and "C"
     */
    @Test
    public void testMosseLikelihood() {
//        int numLeaves = 2;
//        String[] names = {"t0", "t1"};
//        String[] sequences = {"AG", "CG"};
//        String newick = "(t0: 0.4, t1: 0.4);";
//        int numTraits = 1;
//        String trait1Values = "t0=0.15, t1=0.1";

        int numLeaves = 4;
        String[] names = {"t0", "t1", "t2", "t3"};
//        String[] sequences = {"A", "C", "G", "T"};
        String[] sequences = {"AG", "CT", "GG", "TC"};
        String newick = "((t0:0.2,t1:0.2):0.2,(t2:0.3,t3:0.3):0.1);";
        int numTraits = 1;
        String trait1Values = "t0=0.15, t1=0.1, t2=0.25, t3=0.2";
        
//        int numLeaves = 3;
//        String[] sequences = {"AG", "CT", "GG"};
//        String newick = "((t0:0.2,t1:0.2):0.2,t2:0.4);";
//        int numTraits = 1;
//        String trait1Values = "t0=0.15, t1=0.1, t2=0.25";

        Alignment alignment = getAlignment(numLeaves, names, sequences);
        
        // Parameters
        Double[] betasArray = {1.0};
        // Double[] betasArray = {0.1, 0.2};
        double meanSubst = 0.0; // mean substitution rate
        double epsilon = 0.01;
        double startSubsRate = 0.0001;
        int numBins = 1024;

        // Parameters for lambda and mu functions
        Double[] y0 = new Double[] { 0.1 };
        Double[] y1 = new Double[] { 0.2 };
        Double[] x0 = new Double[] { 0.0 };
        Double[] r = new Double[] { 2.5 };
        Double[] yValue = new Double[] { 0.03 }; // constant
        
        // Parameters for Mosse distribution
        double dx = 0.0001;          // distance between xs
        double drift = 0.0;          // drift parameter
        double diffusion = 0.001;    // diffusion parameter
        double dt = 0.01;            // time interval dt
        int width = 5;              
        int resolution = 4;
        // boolean lowresolution = false;

        // Tree and Models Construction
        
        Tree tree = new Tree(newick);

        JukesCantor JC = new JukesCantor();
        // JC.initAndValidate(); (no need, as this function has been called when the object JC is constructed

        SiteModel siteModel = new SiteModel();
        siteModel.initByName(
                "mutationRate", "1.0",
                "gammaCategoryCount", 1,
                "substModel", JC);


        MosseTipLikelihood tipModel = new MosseTipLikelihood();
        tipModel.initByName(
                "beta", new RealParameter(betasArray),
                "subst", Double.toString(meanSubst),
                "epsilon", Double.toString(epsilon),
                "logscale", "false"
        );

        // trait 0
//        TraitSet trait0 = new TraitSet();
//        trait0.initByName(
//                "traitname", "trait0",
//                "taxa", new TaxonSet(alignment),
//                "value", trait0Values);
//        traitsList.add(trait0);
        // trait 1
        TraitSet trait1 = new TraitSet();
        trait1.initByName(
                "traitname", "trait1",
                "taxa", new TaxonSet(alignment),
                "value", trait1Values);
        
        List<TraitSet> traitsList = new ArrayList<>(numTraits);
        traitsList.add(trait1);

        // lambda and mu functions
        // logistic
        RealParameter y0rp = new RealParameter(y0);
        RealParameter y1rp = new RealParameter(y1);
        RealParameter x0rp = new RealParameter(x0);
        RealParameter rrp = new RealParameter(r);
        LogisticFunction logFunc = new LogisticFunction();
        logFunc.initByName( "curveYBaseValue",
                y0rp, "curveMaxY", y1rp,
                "sigmoidMidpoint", x0rp,
                "logisticGrowthRate", rrp);

        // constant
        RealParameter yValueRP = new RealParameter(yValue);
        ConstantLinkFn constFunc = new ConstantLinkFn();
        constFunc.initByName("yV", yValueRP);

        MosseDistribution mosseDist = new MosseDistribution();
        
        mosseDist.initByName(
                "tree", tree,
                "nx", Integer.toString(numBins),           // number of bins for substitution rate
                "dx", Double.toString(dx),                 // distance between xs
                "drift", Double.toString(drift),           // drift parameter
                "diffusion", Double.toString(diffusion),   // diffusion parameter
                "dt", Double.toString(dt),                 // time interval dt
                "width", Integer.toString(width),          
                "resolution", Integer.toString(resolution) //, 
//                "lowresolution", Boolean.toString(lowresolution)
        );

        MosseTreeLikelihood likelihood = new MosseTreeLikelihood();
        likelihood.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "tipModel", tipModel,
                "treeModel", mosseDist,
                "traits", traitsList,
                "startSubsRate", Double.toString(startSubsRate),
                "numRateBins", Integer.toString(numBins),
                "lambdaFunc", logFunc,
                "muFunc", constFunc,
                "resolution", Integer.toString(resolution)
        );

        // using observed root
        double result = likelihood.calculateLogP();
        System.out.println("testMosseLikelihood logP = " + result);

    }

    public void testMosseLikelihoodRoot() {
        // test root node treatments
    }
}
