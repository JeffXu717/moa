package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

public class TheEnsemble extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {
    @Override
    public String getPurposeString() {
        return "The 4th assignment for COMPX523-23A";
    }


    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'e',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");


    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public IntOption winLengthToUpdateEnsembleOption = new IntOption("winLengthUpdateEnsemble", 'l',
            "Every l instances ensemble will be updated.", 1000, 1, Integer.MAX_VALUE);

    protected Classifier[] ensemble;

    protected Classifier candidateClassifier;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        // not just copy but randomize hyperparameters
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        //  receives one instance and use it for training in all base learners
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        // Missing weights
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return false;
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
