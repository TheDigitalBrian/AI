package edu.uab.cis.learning.decisiontree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A decision tree classifier.
 *
 * @param <LABEL> The type of label that the classifier predicts.
 * @param <FEATURE_NAME> The type used for feature names.
 * @param <FEATURE_VALUE> The type used for feature values.
 */
public class DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE> {
    
    LABEL leafNode;
    HashMap<FEATURE_VALUE, DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE>> branchNodes = 
            new HashMap<FEATURE_VALUE, DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE>>();
    HashMap<LABEL, Integer> labels = new HashMap<LABEL, Integer>();
    int splitOnFeature = -1;

    /**
     * Trains a decision tree classifier on the given training examples.
     *
     * <ol> <li>If all examples have the same label, a leaf node is
     * created.</li> <li>If no features are remaining, a leaf node is
     * created.</li> <li>Otherwise, the feature F with the highest information
     * gain is identified. A branch node is created where for each possible
     * value V of feature F: <ol> <li>The subset of examples where F=V is
     * selected.</li> <li>A decision (sub)tree is recursively created for the
     * selected examples. None of these subtrees nor their descendants are
     * allowed to branch again on feature F.</li> </ol> </li> </ol>
     *
     * @param trainingData The training examples, where each example is a set of
     * features and the label that should be predicted for those features.
     */
    public DecisionTree(Collection<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData) {
        ArrayList<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> data =
                (ArrayList<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>) trainingData;
        LABEL tempKey = null;

        // Use map to record how many times each label appears
        // Key = label, Value = label count
        for (int i = 0; i < data.size(); i++) {
            tempKey = data.get(i).getLabel();

            if (labels.containsKey(tempKey)) {
                labels.put(tempKey, labels.get(tempKey) + 1);
            } else {
                labels.put(tempKey, 1);
            }
        }

        // If only one key/value pair, then only one label
        // Return label as leaf node
        if (labels.size() == 1) {
            for (LABEL leaf : labels.keySet()) {
                leafNode = leaf;
            }
        // If there are no features left, 
        // return most frequent label as leaf node
        } else if (data.get(0).getFeatureNames().isEmpty()) {
            int max = -1;

            for (Map.Entry<LABEL, Integer> entry : labels.entrySet()) {
                int value = entry.getValue();
                if (value > max) {
                    leafNode = entry.getKey();
                    max = value;
                }
            }
        // Otherwise, develop branch nodes
        } else {
            Map<Integer, HashMap<Integer, ArrayList<Object>>> map =
                    new HashMap<Integer, HashMap<Integer, ArrayList<Object>>>();
            int i = 0;

            // Convert feature name/value pairs into map entries
            // This map helps calculate information gain
            for (LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> features : data) {
                int j = 0;

                for (FEATURE_NAME name : features.getFeatureNames()) {
                    HashMap<Integer, ArrayList<Object>> featureMap;

                    if (!map.containsKey(j)) {
                        featureMap = new HashMap<Integer, ArrayList<Object>>();
                    } else {
                        featureMap = map.get(j);
                    }

                    ArrayList<Object> pair = new ArrayList<Object>();
                    pair.add(features.getFeatureValue(name));
                    pair.add(features.getLabel());
                    featureMap.put(i, pair);
                    map.put(j, featureMap);
                    j++;
                }
                i++;
            }

            double min = 100;
            int featureCounter = 0;

            for (Map.Entry<Integer, HashMap<Integer, ArrayList<Object>>> feature :
                    map.entrySet()) {
                HashMap<FEATURE_VALUE, HashMap<LABEL, Integer>> labelCount =
                        new HashMap<FEATURE_VALUE, HashMap<LABEL, Integer>>();
                HashMap<FEATURE_VALUE, Integer> labelCountTotal =
                        new HashMap<FEATURE_VALUE, Integer>();
                HashMap<Integer, ArrayList<Object>> featureValues = feature.getValue();
                double total = 0;
                
                for (Map.Entry<Integer, ArrayList<Object>> fValue : featureValues.entrySet()) {
                    FEATURE_VALUE value = (FEATURE_VALUE) fValue.getValue().get(0);
                    LABEL label = (LABEL) fValue.getValue().get(1);
                    HashMap<LABEL, Integer> m;
                    
                    if (labelCount.containsKey(value)) {
                        m = labelCount.get(value);
                        if (m.containsKey(label)) {
                            m.put(label, m.get(label) + 1);
                        } else {
                            m.put(label, 1);
                        }
                        labelCount.put(value, m);
                    } else {
                        m = new HashMap<LABEL, Integer>();
                        m.put(label, 1);
                        labelCount.put(value, m);
                    }
                    int count;
                    if (labelCountTotal.containsKey(value)) {
                        count = labelCountTotal.get(value);
                        labelCountTotal.put(value, count + 1);
                    } else {
                        count = 1;
                        labelCountTotal.put(value, count);
                    }
                    total += 1;
                }
                double sum = 0;
                
                for (Map.Entry<FEATURE_VALUE, HashMap<LABEL, Integer>> lCount :
                        labelCount.entrySet()) {
                    FEATURE_VALUE v = lCount.getKey();
                    double to = (double) labelCountTotal.get(v);
                    double lSum = 0;

                    for (Map.Entry<LABEL, Integer> l : lCount.getValue().entrySet()) {
                        LABEL la = l.getKey();
                        double lc = (double) l.getValue();
                        double prob = -(lc / to) * (Math.log(lc / to) / Math.log(2));
                        lSum += prob;
                    }

                    double product = lSum * (to / total);
                    sum += product;
                }

                if (sum < min) {
                    min = sum;
                    splitOnFeature = featureCounter;
                }
                featureCounter++;
            }

            HashMap<FEATURE_VALUE, HashMap<Integer, LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>> hm =
                    new HashMap<FEATURE_VALUE, HashMap<Integer, LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>>();

            for (int x = 0; x < data.size(); x++) {
                LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lf = data.get(x);
                HashMap<FEATURE_NAME, FEATURE_VALUE> hm2 =
                        new HashMap<FEATURE_NAME, FEATURE_VALUE>();
                FEATURE_VALUE splitFeature = null;
                int count = 0;

                for (FEATURE_NAME fn : lf.getFeatureNames()) {
                    FEATURE_VALUE fv = lf.getFeatureValue(fn);

                    if (count == splitOnFeature) {
                        splitFeature = fv;
                    } else {
                        hm2.put(fn, fv);
                    }
                    count++;
                }

                LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lf2 =
                        new LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>(lf.getLabel(), hm2);
                HashMap<Integer, LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> hm3;
                
                if (hm.containsKey(splitFeature)) {
                    hm3 = hm.get(splitFeature);
                } else {
                    hm3 = new HashMap<Integer, LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>();
                }
                
                hm3.put(hm3.size(), lf2);
                hm.put(splitFeature, hm3);
            }

            for (FEATURE_VALUE f : hm.keySet()) {
                HashMap<Integer, LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> h = hm.get(f);
                List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> newData =
                        new ArrayList<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>();

                for (LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> l : h.values()) {
                    newData.add(l);
                }
                
                DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE> d =
                        new DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE>(newData);
                branchNodes.put(f, d);
            }
        }
    }

    /**
     * Predicts a label given a set of features.
     *
     * <ol> <li>For a leaf node where all examples have the same label, that
     * label is returned.</li> <li>For a leaf node where the examples have more
     * than one label, the most frequent label is returned.</li> <li>For a
     * branch node based on a feature F, E is inspected to determine the value V
     * that it has for feature F. <ol> <li>If the branch node has a subtree for
     * V, then example E is recursively classified using the subtree.</li>
     * <li>If the branch node does not have a subtree for V, then the most
     * frequent label for the examples at the branch node is returned.</li>
     * </ol> <li> </ol>
     *
     * @param features The features for which a label is to be predicted.
     * @return The predicted label.
     */
    public LABEL classify(Features<FEATURE_NAME, FEATURE_VALUE> features) {
        //System.out.println(splitOnFeature);
        if (!(leafNode == null)) {
            return leafNode;
        } else {
            HashMap<FEATURE_NAME, FEATURE_VALUE> hm =
                    new HashMap<FEATURE_NAME, FEATURE_VALUE>();
            DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE> dt = branchNodes.get(null);
            int counter = 0;

            for (FEATURE_NAME fn : features.getFeatureNames()) {
                if (counter == splitOnFeature) {
                    FEATURE_VALUE fv = features.getFeatureValue(fn);
                    if (!branchNodes.containsKey(fv)) {
                        int max = -1;

                        for (Map.Entry<LABEL, Integer> entry : labels.entrySet()) {
                            int value = entry.getValue();
                            if (value > max) {
                                leafNode = entry.getKey();
                                max = value;
                            }
                        }
                        return leafNode;
                    }
                    dt = branchNodes.get(fv);
                } else {
                    hm.put(fn, features.getFeatureValue(fn));
                }
                counter++;
            }
            return dt.classify(new Features(hm));
        }
    }
}
