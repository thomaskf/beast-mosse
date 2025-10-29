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
 * @author Thomas Wong
 */

public class MosseTreeLikelihoodTest2 {

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
        int numLeaves = 15;
        String[] names = {"sp1","sp2","sp5","sp6","sp7","sp8","sp9","sp10","sp11","sp12","sp13","sp14","sp15","sp16","sp17"};
        String[] sequences = {"A","A","A","G","T","A","C","A","C","G","A","G","G","T","A"};
        String newick = "(sp2:13.77320255,(sp1:12.76688384,((((sp12:1.170387028,sp13:1.170387028):0.9837720325,sp9:2.154159061):5.451401092,((sp5:4.311645343,(sp14:0.8910055279,sp15:0.8910055279):3.420639815):2.536663776,((sp16:0.3011866125,sp17:0.3011866125):4.264383667,(sp6:3.95083843,sp7:3.95083843):0.6147318498)0:2.282738839):0.7572510339):2.554739141,((sp10:2.059478202,sp11:2.059478202):0.4198789018,sp8:2.479357104):7.68094219):2.60658455):1.006318707);";
        
        // Trait information
        int numTraits = 15;
        String traitValues = "sp1=0.01070713, sp2=0.00970083, sp5=0.0104502, sp6=0.00918406, sp7=0.01006519, sp8=0.01041169, sp9=0.01002527, sp10=0.01027695, sp11=0.00945112, sp12=0.01050648, sp13=0.00998902, sp14=0.00937548, sp15=0.00881356, sp16=0.00981287, sp17=0.01024662";

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

        // trait
        TraitSet trait = new TraitSet();
        trait.initByName(
                "traitname", "trait1",
                "taxa", new TaxonSet(alignment),
                "value", traitValues);
        
        List<TraitSet> traitsList = new ArrayList<>(numTraits);
        traitsList.add(trait);

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
