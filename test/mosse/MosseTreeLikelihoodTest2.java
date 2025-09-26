package mosse;

import beast.base.inference.parameter.RealParameter;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.Frequencies;
import beast.base.evolution.substitutionmodel.GTR;
import beast.base.evolution.substitutionmodel.GeneralSubstitutionModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.substitutionmodel.ComplexSubstitutionModel;
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

public class MosseTreeLikelihoodTest2 {

    private static double DELTA = 1e-7;

    /**
     * returns an Alignment of nucleotide sequences
     * @param numLeaves number of taxa (or leaves in tree)
     * @param sequences nucleotide sequences for each taxa
     * @return an Alignment of nucleotide sequences
     */
    public Alignment getAlignment(int numLeaves, String[] sequences) {
        List<Sequence> seqList = new ArrayList<>();

        for (int i = 0; i < numLeaves; i++) {
            String taxonID = "t" + i;
            seqList.add(new Sequence(taxonID, sequences[i]));
        }

        Alignment alignment = new Alignment(seqList, "nucleotide");

        return alignment;
    }

    /**
     * returns an Alignment of nucleotide sequences
     * @param numLeaves number of taxa (or leaves in tree)
     * @param sequences nucleotide sequences for each taxa
     * @return an Alignment of nucleotide sequences
     */
    public Alignment getAlignment(int numLeaves, String[] sequences, String[] taxonIDs) {
        List<Sequence> seqList = new ArrayList<>();

        for (int i = 0; i < numLeaves; i++) {
            seqList.add(new Sequence(taxonIDs[i], sequences[i]));
        }

        Alignment alignment = new Alignment(seqList, "nucleotide");

        return alignment;
    }
    
    
    /**
     * tests initialization of MosseTreeLikelihood does not throw any errors
     */
    @Test
    public void testMosseLikelihood() {
        int numLeaves = 15;
        String[] taxonIDs = {"sp1","sp2","sp5","sp6","sp7","sp8","sp9","sp10","sp11","sp12","sp13","sp14","sp15","sp16","sp17"};
        String[] sequences = {"A","A","A","G","T","A","C","A","C","G","A","G","G","T","A"};
        Alignment alignment = getAlignment(numLeaves, sequences, taxonIDs);
        String newick = "(sp2:13.77320255,(sp1:12.76688384,((((sp12:1.170387028,sp13:1.170387028):0.9837720325,sp9:2.154159061):5.451401092,((sp5:4.311645343,(sp14:0.8910055279,sp15:0.8910055279):3.420639815):2.536663776,((sp16:0.3011866125,sp17:0.3011866125):4.264383667,(sp6:3.95083843,sp7:3.95083843):0.6147318498)0:2.282738839):0.7572510339):2.554739141,((sp10:2.059478202,sp11:2.059478202):0.4198789018,sp8:2.479357104):7.68094219):2.60658455):1.006318707);";
        
        // Trait information
        int numTraits = 15;
        String traitValues = "sp1=0.01070713, sp2=0.00970083, sp5=0.0104502, sp6=0.00918406, sp7=0.01006519, sp8=0.01041169, sp9=0.01002527, sp10=0.01027695, sp11=0.00945112, sp12=0.01050648, sp13=0.00998902, sp14=0.00937548, sp15=0.00881356, sp16=0.00981287, sp17=0.01024662";
        
        // Parameters
        Double[] betasArray = {1.0};
        double meanSubst = 0.0; // mean substitution rate
        double epsilon = 0.001;
        double startSubsRate = 0.0001;
        int numBins = 1024;

        // Parameters for lambda and mu functions
        Double[] x0 = new Double[] { 0.0 };
        Double[] y1 = new Double[] { 0.1 };
        Double[] y0 = new Double[] { 0.1 };
        Double[] r = new Double[] { 0.01 };
        Double[] yValue = new Double[] { 0.01 }; // constant
        
        // Parameters for Mosse distribution
        double dx = 0.0001;          // distance between xs
        double drift = 0.0;          // drift parameter
        double diffusion = 0.001;    // diffusion parameter
        double dt = 0.01;            // time interval dt
        int width = 5;              // diffusion parameter
        int resolution = 1;          // time interval dt

        // Tree and Models Construction
        
        Tree tree = new Tree(newick);
        
        // Q Matrix
        Double[] Q_matrix = new Double[] { 0.2, 0.1, 0.1, 0.1, 0.3, 0.4, 0.2, 0.2, 0.2, 0.3, 0.4, 0.2 };
        Double[] rate_matrix = new Double[Q_matrix.length];
        
        
        // Compute the corresponding rate matrix
        for (int i = 0; i < Q_matrix.length; i++)
        		rate_matrix[i] = Q_matrix[i] / 0.25;
        
        RealParameter rates = new RealParameter(rate_matrix);
        
        Frequencies freqs = new Frequencies();
        freqs.initByName("frequencies", new RealParameter(new Double[]{0.25,0.25,0.25,0.25}));

        ComplexSubstitutionModel nonrev = new ComplexSubstitutionModel();
        nonrev.initByName(
            "rates",  rates,        // all off-diagonals; diagonals are set so rows sum to 0
            "frequencies", freqs    // equilibrium frequencies
            // "eigenSystem", "beast.base.evolution.substitutionmodel.DefaultEigenSystem" // optional
        );
        
        SiteModel siteModel = new SiteModel();
        siteModel.initByName(
                "mutationRate", "1.0",
                "gammaCategoryCount", 1,
                "substModel", nonrev);


        MosseTipLikelihood tipModel = new MosseTipLikelihood();
        tipModel.initByName(
                "beta", new RealParameter(betasArray),
                "subst", Double.toString(meanSubst),
                "epsilon", Double.toString(epsilon),
                "logscale", "false"
        );
        // tipModel.initAndValidate(); (no need, as this function has been called inside initByName())

        // trait
        TraitSet trait = new TraitSet();
        trait.initByName(
                "traitname", "trait",
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
                "dx", Double.toString(dx),                 // distrance between xs
                "drift", Double.toString(drift),           // drift parameter
                "diffusion", Double.toString(diffusion),   // diffusion parameter
                "dt", Double.toString(dt),                 // time interval dt
                "width", Integer.toString(width),          // diffusion parameter
                "resolution", Integer.toString(resolution) // time interval dt
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
                "muFunc", constFunc
                );

        // using observed root
        double result = likelihood.calculateLogP();
        System.out.println("testMosseLikelihood logP = " + result);

    }

    public void testMosseLikelihoodRoot() {
        // test root node treatments
    }
}
