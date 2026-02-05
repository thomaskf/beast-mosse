package mosse;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;


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
//         int numLeaves = 15;
//         String[] names = {"sp1","sp2","sp5","sp6","sp7","sp8","sp9","sp10","sp11","sp12","sp13","sp14","sp15","sp16","sp17"};
//         String[] sequences = {"A","A","A","G","T","A","C","A","C","G","A","G","G","T","A"};
//         String newick = "(sp2:13.77320255,(sp1:12.76688384,((((sp12:1.170387028,sp13:1.170387028):0.9837720325,sp9:2.154159061):5.451401092,((sp5:4.311645343,(sp14:0.8910055279,sp15:0.8910055279):3.420639815):2.536663776,((sp16:0.3011866125,sp17:0.3011866125):4.264383667,(sp6:3.95083843,sp7:3.95083843):0.6147318498)0:2.282738839):0.7572510339):2.554739141,((sp10:2.059478202,sp11:2.059478202):0.4198789018,sp8:2.479357104):7.68094219):2.60658455):1.006318707);";
//         int numTraits = 1;
//         String traitValues = "sp1=0.01070713, sp2=0.00970083, sp5=0.0104502, sp6=0.00918406, sp7=0.01006519, sp8=0.01041169, sp9=0.01002527, sp10=0.01027695, sp11=0.00945112, sp12=0.01050648, sp13=0.00998902, sp14=0.00937548, sp15=0.00881356, sp16=0.00981287, sp17=0.01024662";

    	int numLeaves = 20;
        String[] names = {"sp1","sp2","sp3","sp4","sp5","sp6","sp7","sp8","sp9","sp10","sp11","sp12","sp13","sp14","sp15","sp16","sp17","sp18","sp19","sp20"};
        String[] sequences = {"t","t","t","t","t","t","t","t","t","t","t","t","t","t","t","t","t","t","t","t"};
        String newick = "((sp1:0.9410095548820001,((sp4:0.43828346558200015,((sp6:0.23995376277536223,(sp12:0.059211836322,sp13:0.059211836322):0.18074192645336223):0.14593396528993902,(((sp14:0.04861332138200014,(sp19:0.0035636480020000416,sp20:0.0035636480020000416):0.0450496733800001):0.13675683785009132,((sp7:0.008316409561679644,sp8:0.008316409561679644):0.0887005381275639,(sp17:0.012301279822000133,sp18:0.012301279822000133):0.08471566786724341):0.08835321154284792):0.13527593549993097,sp5:0.32064609473202244):0.06524163333327881):0.052395737516698904):0.024399052287571943,sp3:0.4626825178695721):0.478327037012428):0.15909207830000005,(sp2:0.1550889664224112,((sp15:0.029907699592000014,sp16:0.029907699592000014):0.0977520085900001,(sp10:0.06324471327385377,(sp9:0.057467097563870626,sp11:0.057467097563870626):0.0057776157099831416):0.06441499490814635):0.027429258240411092):0.945012666759589):0.0;";
        String traitValues = "sp1=0.0387876829138803988,sp2=-0.0261650640045382014,sp3=0.0000780613211212051,sp4=0.0640955976744835010,sp5=0.0631755463800323935,sp6=0.0175265726505994986,sp7=0.0285494149210851998,sp8=0.0034181730917631800,sp9=0.0381431935689817980,sp10=0.0429503413943981976,sp11=0.0322004948111379030,sp12=0.0470716865963385980,sp13=0.0547267264548012972,sp14=0.0392330272674960984,sp15=-0.0039068533176126897,sp16=0.0182309457961658990,sp17=0.0801855614646413001,sp18=0.0790165878526876003,sp19=0.0521209102564738000,sp20=0.0470196223928652998";
        int numTraits = 1;

        Alignment alignment = getAlignment(numLeaves, names, sequences);

        // Parameters
        Double[] betasArray = { 0.0 }; // 3.632238571467065E-4 };
        double meanSubst = 0.050033792386930856;
        double epsilon = 6.98738781637296E-6;

        // Double[] betasArray = { 1.0 };
        // double meanSubst = 0.0; // mean substitution rate
        // double epsilon = 0.001;
        
        int numBins = 1024;

        // Parameters for lambda and mu functions
        Double[] y0 = new Double[] { 0.0 };
        Double[] y1 = new Double[] { 8.096057330537827 };
        Double[] x0 = new Double[] { 0.0 };
        Double[] r = new Double[] { 171.92056820137893 };
        Double[] yValue = new Double[] { 6.7718723223281545 }; // constant

        // Parameters for Mosse distribution
        double dx = 0.0001;          // distance between xs
        double drift = 0.0;          // drift parameter
        double diffusion = 2.3124558018078874E-8;    // diffusion parameter
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
//                "startSubsRate", Double.toString(startSubsRate),
//                "numRateBins", Integer.toString(numBins),
                "lambdaFunc", logFunc,
                "muFunc", constFunc
//                "resolution", Integer.toString(resolution)
        );

        // using observed root
        double result = likelihood.calculateLogP();
        System.out.println("testMosseLikelihood logP = " + result);

    }

    public void testMosseLikelihoodRoot() {
        // test root node treatments
    }
}
