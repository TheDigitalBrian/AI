package edu.uab.cis.learning.decisiontree;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Maps;

/**
 * A mapping from feature names to feature values.
 *
 * @param <FEATURE_NAME> The type used for feature names.
 * @param <FEATURE_VALUE> The type used for feature values.
 */
public class Features<FEATURE_NAME, FEATURE_VALUE> {

    private Map<FEATURE_NAME, FEATURE_VALUE> features;

    /**
     * Creates a new set of features from a name-to-value mapping
     *
     * @param features A mapping from feature names to feature values.
     */
    public Features(Map<FEATURE_NAME, FEATURE_VALUE> features) {
        this.features = features;
    }

    /**
     * Creates a new set of features from an array of feature values.
     *
     * The name of each feature will be its integer index in the list.
     *
     * @param featureValues The array of feature values.
     * @return A set of features where indexes are mapped to feature values.
     */
    @SafeVarargs
    public static <FEATURE_VALUE_TYPE> Features<Integer, FEATURE_VALUE_TYPE> of(
            FEATURE_VALUE_TYPE... featureValues) {
        Map<Integer, FEATURE_VALUE_TYPE> features = Maps.newHashMap();
        for (int i = 0; i < featureValues.length; ++i) {
            features.put(i, featureValues[i]);
        }
        return new Features<Integer, FEATURE_VALUE_TYPE>(features);
    }

    /**
     * @return The set of feature names.
     */
    public Set<FEATURE_NAME> getFeatureNames() {
        return Collections.unmodifiableSet(this.features.keySet());
    }

    /**
     * @param featureName The name of a feature.
     * @return The value associated with the named feature.
     */
    public FEATURE_VALUE getFeatureValue(FEATURE_NAME featureName) {
        return this.features.get(featureName);
    }
}
