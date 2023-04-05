/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * k Nearest Neighbor.<p>
 *
 * Valid options are:<p>
 *
 * -k number of neighbours <br> -m max instances <br> 
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class kNNwithNearestCentroid extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	// For checking regression with mean value or median value
	public FlagOption medianOption = new FlagOption("median",'m',"median or mean");

	public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

        public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                "KDTree search algorithm for nearest neighbour search"
            }, 0);


	int C = 0;

    @Override
    public String getPurposeString() {
        return "Rocchio classifier: kNN with nearest centroid.";
    }

    protected Instances window;
	protected Instances centroids;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
		this.window = null;
    }


	private Map<Integer, InstanceGroup> classValInstanceGroupMap;

	private void addInst(Instance inst, int instClassVal)
	{
		this.window.add(inst);
		// ? How to cal centroid
	}

	private void deleteInst()
	{
		this.window.delete(0);
		// ? How to cal centroid
	}

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		int instClassVal = (int) inst.classValue();
		if (inst.classValue() > C)
			C = (int) instClassVal;
		if (this.window == null) { //Init window
			this.window = new Instances(inst.dataset());
			this.classValInstanceGroupMap = new HashMap<>();
		}
		if (!this.classValInstanceGroupMap.containsKey(instClassVal))
		{
			this.classValInstanceGroupMap.put(instClassVal, new InstanceGroup());
		}

		if (this.limitOption.getValue() <= this.window.numInstances()) { // If out of limit
			deleteInst();
		}
		addInst(inst, instClassVal); // add inst to window
    }

	@Override
    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[C+1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.centroids);
			} else {
				search = new KDTree();
				search.setInstances(this.centroids);
			}	
			if (this.window.numInstances()>0) {	
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(1, this.centroids.numInstances()));
				//================== Regression ====================
				if(inst.classAttribute().isNumeric()){
					double[] result = new double[1];
					// For storing the sum of class values of all the k nearest neighbours
					double sum = 0;
					// For storing the number of the nearest neighbours
					int num = neighbours.numInstances();
					//================== Median ====================
					if(medianOption.isSet()){
						// For storing every neighbour's class value
						double[] classValues = new double[num];

						for(int i=0;i<num;i++){
							classValues[i] = neighbours.instance(i).classValue();
						}
						// Sort the class values
						Arrays.sort(classValues);
						// Assign the median value into result
						if(classValues.length%2==1){
							result[0] = classValues[num/2];
						}else{
							result[0] = (classValues[num/2 - 1] + classValues[num/2]) / 2;
						}
						return result;
						//=============== End of Median ============
					}else{
						//================== Mean ==================
						for(int i=0;i<num;i++){
							sum += neighbours.instance(i).classValue();
						}
						// Calculate the mean of all k nearest neighbours' class values
						result[0] = sum / num;
						return result;
						//=============== End of Mean ==============
					}
					//============= End of Regression ==============
				}else{
					// vote for neighbours
					for (int i = 0; i < neighbours.numInstances(); i++) {
						v[(int) neighbours.instance(i).classValue()]++;
					}
				}
			}
		} catch(Exception e) {
			//System.err.println("Error: kNN search failed.");
			//e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}

class InstanceGroup
{
	int numInGroup;
	Instance centroid;

	public int getNumInGroup()
	{
		return numInGroup;
	}

	public void setNumInGroup(int numInGroup)
	{
		this.numInGroup = numInGroup;
	}

	public Instance getCentroid()
	{
		return centroid;
	}

	public InstanceGroup() {
		numInGroup = 0;
	}
}