package edu.uab.cis.learning.decisiontree;

import java.util.Map;

import com.google.common.collect.Maps;

/**
 * A mapping from feature names to feature values that is also associated with a
 * label.
 *
 * @param <LABEL> The type used as the representation of the label.
 * @param <FEATURE_NAME> The type used for feature names.
 * @param <FEATURE_VALUE> The type used for feature values.
 */
public class LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>
        extends Features<FEATURE_NAME, FEATURE_VALUE> {

    private LABEL label;

    /**
     * Creates a labeled set of features from a label and a name-to-value
     * mapping.
     *
     * @param label The label for the set of features.
     * @param features A mapping from feature names to feature values.
     */
    public LabeledFeatures(LABEL label, Map<FEATURE_NAME, FEATURE_VALUE> features) {
        super(features);
        this.label = label;
    }

    /**
     * Creates a labeled set of features from a label and an array of feature
     * values.
     *
     * The name of each feature will be its integer index in the list.
     *
     * @param label The label for the set of features.
     * @param featureValues The array of feature values.
     * @return A labeled set of features where indexes are mapped to feature
     * values.
     */
    @SafeVarargs
    public static <LABEL_TYPE, FEATURE_VALUE_TYPE> LabeledFeatures<LABEL_TYPE, Integer, FEATURE_VALUE_TYPE> of(LABEL_TYPE label, FEATURE_VALUE_TYPE... featureValues) {
        Map<Integer, FEATURE_VALUE_TYPE> features = Maps.newHashMap();
        for (int i = 0; i < featureValues.length; ++i) {
            features.put(i, featureValues[i]);
        }
        return new LabeledFeatures<LABEL_TYPE, Integer, FEATURE_VALUE_TYPE>(label, features);
    }

    /**
     * A convenience specialization of {@link #of(Object, Object...)} for
     * String-valued features.
     */
    public static <LABEL_TYPE> LabeledFeatures<LABEL_TYPE, Integer, String> ofStrings(
            LABEL_TYPE label,
            String... featureValues) {
        return LabeledFeatures.of(label, featureValues);
    }

    /**
     * A convenience specialization of {@link #of(Object, Object...)} for
     * Integer-valued features.
     */
    public static <LABEL_TYPE> LabeledFeatures<LABEL_TYPE, Integer, Integer> ofIntegers(
            LABEL_TYPE label,
            Integer... featureValues) {
        return LabeledFeatures.of(label, featureValues);
    }

    /**
     * @return The label associated with this set of features.
     */
    public LABEL getLabel() {
        return this.label;
    }
}
