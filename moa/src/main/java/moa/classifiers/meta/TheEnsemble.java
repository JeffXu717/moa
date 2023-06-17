package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;

import java.util.Random;

public class TheEnsemble extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;
    @Override
    public String getPurposeString() {
        return "The 4th assignment for COMPX523-23A";
    }

    // Actually only work for Hoeffding Tree
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'e',
            "Classifier (HoeffdingTree) to train.", Classifier.class, "trees.HoeffdingTree");


    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public IntOption winLengthToUpdateEnsembleOption = new IntOption("winLengthUpdateEnsemble", 'l',
            "Every l instances ensemble will be updated.", 1000, 1, Integer.MAX_VALUE);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',
                                                          "Seed for generating base learner with random hyperparams", -1);

    public FlagOption systemTimeAsRandomSeedOption = new FlagOption("systemTimeAsRandomSeed", 't',
            "Using system time as random seed, not fixed one.");

    protected Classifier[] ensemble;

    protected double[] predictPerformanceArray;

    protected long[] instancesTestedArray;

    protected Classifier candidateLearner;

    protected double predictPerformanceOfCandidate;

    protected long instancesTestedOfCandidate;

    protected long instancesProcessedFromBeginning;

    protected Classifier baseLearnerTemplate;

    // parameters to vary the hyperparameters
    // of the base Hoeffding Trees.
    // Grace period
    protected int gpMin = 10;
    protected int gpMax = 200;
    protected int gpStep = 10;
    // Split Confidence
    protected float scMin = 0;
    protected float scMax = 1;
    protected float scStep = 0.05f;
    // Tie Threshold
    protected float tMin = 0;
    protected float tMax = 1;
    protected float tStep = 0.05f;

    // region utils
    protected int randomIntVal(int min, int max, int step)
    {
        return min + step * this.classifierRandom.nextInt((max - min) / step + 1);
    }

    protected float randomFloatVal(float min, float max, float step)
    {
        return min + step * this.classifierRandom.nextInt(Math.round((max - min) / step) + 1);
    }

    protected void varyHyperParametersOfHT(HoeffdingTree ht)
    {
        ht.gracePeriodOption.setValue(randomIntVal(gpMin, gpMax, gpStep));
        ht.splitConfidenceOption.setValue(randomFloatVal(scMin, scMax, scStep));
        ht.tieThresholdOption.setValue(randomFloatVal(tMin, tMax, tStep));
    }

    protected double calPredictPerformance(double predictPerformance, long instancesTested, DoubleVector vote, Instance inst)
    {
        if (vote.maxIndex() == inst.classValue())
        {
            predictPerformance = (predictPerformance * instancesTested + 1)
                    / (instancesTested + 1);
        }
        else
        {
            predictPerformance = predictPerformance * instancesTested
                    / (instancesTested + 1);
        }
        return predictPerformance;
    }

    public Classifier genRandomHyperParamsHT()
    {
        Classifier newHT = baseLearnerTemplate.copy();
        // not just copy but randomize hyperparameters
        varyHyperParametersOfHT((HoeffdingTree) newHT);
        return newHT;
    }
    // endregion

    @Override
    public void resetLearningImpl() {
        //reset random seed
        if (this.systemTimeAsRandomSeedOption.isSet())
        {
            this.randomSeed = (int) System.currentTimeMillis();
        }
        else
        {
            this.randomSeed = this.randomSeedOption.getValue();
        }
        this.classifierRandom = new Random(this.randomSeed);

        // reset total counter
        this.instancesProcessedFromBeginning = 0;

        // reset predict performances
        this.predictPerformanceArray = new double[this.ensembleSizeOption.getValue()];

        // reset ensemble
        this.instancesTestedArray = new long[this.ensembleSizeOption.getValue()];
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        baseLearnerTemplate = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearnerTemplate.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            // reset counter to 0
            this.instancesTestedArray[i] = 0;

            // reset predict performance, every learner has the same weight at the beginning
            this.predictPerformanceArray[i] = 0;

            // reset learner in ensemble
            this.ensemble[i] = genRandomHyperParamsHT();
        }

        // reset candidate
        this.instancesTestedOfCandidate = 0;
        this.predictPerformanceOfCandidate = 0;
        this.candidateLearner = genRandomHyperParamsHT();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        // update predict perfarmance of the candidate
        DoubleVector voteCandidate = new DoubleVector(this.candidateLearner.getVotesForInstance(inst));
        if (voteCandidate.sumOfValues() > 0.0)
        {
            this.predictPerformanceOfCandidate = calPredictPerformance(this.predictPerformanceOfCandidate,
                    this.instancesTestedOfCandidate, voteCandidate, inst);
            this.instancesTestedOfCandidate++;
        }

        // update predict perfarmances of ensemble
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                // calculate predict performances of ensemble
                this.predictPerformanceArray[i] = calPredictPerformance(this.predictPerformanceArray[i],
                        this.instancesTestedArray[i], vote, inst);
                this.instancesTestedArray[i] = this.instancesTestedArray[i] + 1;
            }
        }

        //  receives one instance and use it for training in all base learners and the candidate
        this.candidateLearner.trainOnInstance(inst);
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i].trainOnInstance(inst);
        }

        this.instancesProcessedFromBeginning++;
        // Test - then - train - then - checkUpdate
        if (this.instancesProcessedFromBeginning % this.winLengthToUpdateEnsembleOption.getValue() == 0)
        {
            // Time to update ensemble
            int minIndex = 0;
            double minPredictPerformance = this.predictPerformanceArray[0];
            for (int i = 1; i < this.predictPerformanceArray.length; i++)
            {
                if (this.predictPerformanceArray[i] < minPredictPerformance)
                {
                    minIndex = i;
                    minPredictPerformance = this.predictPerformanceArray[i];
                }
            }

            // campare min(p(e)) with p(c)
            if (this.predictPerformanceOfCandidate > minPredictPerformance)
            {
                // replace worst one in ensemble with the candidate
                this.ensemble[minIndex] = this.candidateLearner;
                this.instancesTestedArray[minIndex] = this.instancesTestedOfCandidate;
                this.predictPerformanceArray[minIndex] = this.predictPerformanceOfCandidate;
            }
            // change a new candidate
            this.instancesTestedOfCandidate = 0;
            this.predictPerformanceOfCandidate = 0;
            this.candidateLearner = genRandomHyperParamsHT();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                // predict performance as weight
                vote.scaleValues(this.predictPerformanceArray[i]);
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0)};
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }
}
