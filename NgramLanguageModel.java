package edu.uab.cis.probability.ngram;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A probabilistic n-gram language model.
 *
 * @param <T> The type of items in the sequences over which the language model
 * estimates probabilities.
 */
public class NgramLanguageModel<T> {

    // size of n for n-gram
    int nSize;
    // number of unique one-grams
    double v;
    Representation rep;
    Smoothing smo;
    // list of number of times each conditional statement appears
    Map<List<T>, Integer> conditionals = new HashMap<List<T>, Integer>();
    // probability associated with each n-gram
    Map<List<T>, Double> nGrams = new HashMap<List<T>, Double>();

    enum Smoothing {

        /**
         * Do not apply smoothing. An n-gram w<sub>1</sub>,...,w<sub>n</sub>
         * will have its joint probability P(w<sub>1</sub>,...,w<sub>n</sub>)
         * estimated as #(w<sub>1</sub>,...,w<sub>n</sub>) / N, where N
         * indicates the total number of all 1-grams observed during training.
         *
         * Note that we have defined only the joint probability of an n-gram
         * here. Deriving the conditional probability from the definition above
         * is left as an exercise.
         */
        NONE,
        /**
         * Apply Laplace smoothing. An n-gram w<sub>1</sub>,...,w<sub>n</sub>
         * will have its conditional probability
         * P(w<sub>n</sub>|w<sub>1</sub>,...,w<sub>n-1</sub>) estimated as (1 +
         * #(w<sub>1</sub>,...,w<sub>n</sub>)) / (V +
         * #(w<sub>1</sub>,...,w<sub>n-1</sub>)), where # indicates the number
         * of times an n-gram was observed during training and V indicates the
         * number of <em>unique</em> 1-grams observed during training.
         *
         * Note that Laplace smoothing defines only the conditional probability
         * of an n-gram, not the joint probability.
         */
        LAPLACE
    }

    enum Representation {

        /**
         * Calculate probabilities in the normal range, [0,1].
         */
        PROBABILITY,
        /**
         * Calculate log-probabilities instead of probabilities. In every case
         * where probabilities would have been multiplied, take advantage of the
         * fact that log(P(x)*P(y)) = log(P(x)) + log(P(y)) and add
         * log-probabilities instead. This will improve efficiency since
         * addition is faster than multiplication, and will avoid some numerical
         * underflow problems that occur when taking the product of many small
         * probabilities close to zero.
         */
        LOG_PROBABILITY
    }

    /**
     * Creates an n-gram language model.
     *
     * @param n The number of items in an n-gram.
     * @param representation The type of representation to use for
     * probabilities.
     * @param smoothing The type of smoothing to apply when estimating
     * probabilities.
     */
    public NgramLanguageModel(int n, Representation representation, Smoothing smoothing) {
        nSize = n;
        rep = representation;
        smo = smoothing;
    }

    /**
     * Trains the language model with the n-grams from a sequence of items.
     *
     * This typically involves collecting counts of n-grams that occurred in the
     * sequence.
     *
     * @param sequence The sequence on which the model should be trained.
     */
    public void train(List<T> sequence) {
        int index;
        int iTemp;
        double dTemp;
        List<T> sub;
        List<T> subOfSub;
        List<T> curItem;

        // if smoothing is not Laplace
        if (smo.equals(Smoothing.NONE)) {
            // first, generate conditionals and add to proper map
            for (int i = 1; i <= sequence.size() + 1; i++) {
                // if length of subsequence is less than value of n, start index at 0
                // otherwise start index at difference of i and value of n
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }

                // get conditional
                sub = sequence.subList(index, i - 1);

                // for every possible conditional based on subsequence, increment value in map
                for (int j = 0; j < sub.size(); j++) {
                    subOfSub = sub.subList(j, sub.size());
                    // if conditional is part of map, get value
                    // if not, initialize at 0
                    if (conditionals.containsKey(subOfSub)) {
                        iTemp = conditionals.get(subOfSub);
                    } else {
                        iTemp = 0;
                    }
                    // increment value and place in map
                    iTemp++;
                    conditionals.put(subOfSub, iTemp);
                }
            }

            // generate n-grams
            for (int i = 1; i <= sequence.size(); i++) {
                // if length of subsequence is less than value of n, start index at 0
                // otherwise start index at difference of i and value of n
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }

                // get n-gram
                sub = sequence.subList(index, i);

                // for every possible n-gram based on subsequence, add to probability
                for (int j = 0; j < sub.size(); j++) {
                    // if n-gram is not a one-gram, get conditional
                    // if it is, no conditional available
                    if (j != sub.size() - 1) {
                        subOfSub = sub.subList(j, sub.size() - 1);
                    } else {
                        subOfSub = sequence;
                    }
                    curItem = sub.subList(j, sub.size());

                    //if n-gram is part of map, get probability
                    // if not, initialize probability
                    if (nGrams.containsKey(curItem)) {
                        dTemp = nGrams.get(curItem);
                    } else {
                        dTemp = 0.0;
                    }

                    // if one-gram, probability is 1/size
                    // if not, probability is 1/(number of times conditional appeared)
                    // THIS RESULTS IN A TOTAL PROBABILITY OF n/#
                    if (subOfSub.equals(sequence)) {
                        dTemp += (1.0 / (double) sequence.size());
                    } else {
                        dTemp += (1.0 / (double) conditionals.get(subOfSub));
                    }
                    nGrams.put(curItem, dTemp);
                }
            }
            // if smoothing is Laplace
        } else if (smo.equals(Smoothing.LAPLACE)) {
            v = 0.0;
            // first, generate conditionals and add to proper map
            for (int i = 1; i <= sequence.size() + 1; i++) {
                // if length of subsequence is less than value of n, start index at 0
                // otherwise start index at difference of i and value of n
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }

                // get conditional
                sub = sequence.subList(index, i - 1);

                // for every possible conditional based on subsequence, increment value in map
                for (int j = 0; j < sub.size(); j++) {
                    // if conditional is part of map, get value
                    // if not, initialize at 0
                    subOfSub = sub.subList(j, sub.size());
                    if (conditionals.containsKey(subOfSub)) {
                        iTemp = conditionals.get(subOfSub);
                    } else {
                        iTemp = 0;
                        // when initializing, if it is a one-gram, increment unique one-gram counter
                        if (subOfSub.size() == 1) {
                            v += 1.0;
                        }
                    }
                    // increment value and place in map
                    iTemp++;
                    conditionals.put(subOfSub, iTemp);
                }
            }

            // generate n-grams
            for (int i = 1; i <= sequence.size(); i++) {
                // if length of subsequence is less than value of n, start index at 0
                // otherwise start index at difference of i and value of n
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }

                // get n-gram
                sub = sequence.subList(index, i);

                // for every possible n-gram based on subsequence, add to probability
                for (int j = 0; j < sub.size(); j++) {
                    // if n-gram is not a one-gram, get conditional
                    // if it is, no conditional available
                    if (j != sub.size() - 1) {
                        subOfSub = sub.subList(j, sub.size() - 1);
                    } else {
                        subOfSub = sequence;
                    }
                    curItem = sub.subList(j, sub.size());

                    //if n-gram is part of map, get probability
                    // if not, initialize probability
                    if (nGrams.containsKey(curItem)) {
                        dTemp = nGrams.get(curItem);
                    } else {
                        // if one-gram, probability starts at 1/(size + v)
                        // if not, probability starts at 1/(number of times conditional appeared + v)
                        if (subOfSub.equals(sequence)) {
                            dTemp = (1.0 / ((double) sequence.size() + v));
                        } else {
                            dTemp = (1.0 / ((double) conditionals.get(subOfSub) + v));
                        }
                    }

                    // if one-gram, add 1/(size + v) to probability
                    // if not, add 1/(number of times conditional appeared + v) to probability
                    // THIS RESULTS IN A TOTAL PROBABILITY OF (1 + n)/(# + v)
                    if (subOfSub.equals(sequence)) {
                        dTemp += (1.0 / ((double) sequence.size() + v));
                    } else {
                        dTemp += (1.0 / ((double) conditionals.get(subOfSub) + v));
                    }
                    nGrams.put(curItem, dTemp);
                }
            }
        } else {
            System.out.println("What'chu talkin' 'bout, Willis?");
        }
    }

    /**
     * Return the estimated n-gram probability of the sequence:
     *
     * P(w<sub>0</sub>,...,w<sub>k</sub>) = ???<sub>i=0,...,k</sub>
     * P(w<sub>i</sub>|w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ...,
     * w<sub>i-1</sub>)
     *
     * For example, a 3-gram language model would calculate the probability of
     * the sequence [A,B,B,C,A] as:
     *
     * P(A,B,B,C,A) = P(A)*P(B|A)*P(B|A,B)*P(C|B,B)*P(A|B,C)
     *
     * The exact calculation of the conditional probabilities in this expression
     * depends on the smoothing method. See {@link Smoothing}.
     *
     * The result is in the range [0,1] with {@link Representation#PROBABILITY}
     * and in the range (-???,0] with {@link Representation#LOG_PROBABILITY}.
     *
     * @param sequence The sequence of items whose probability is to be
     * estimated.
     * @return The estimated probability of the sequence.
     */
    public double probability(List<T> sequence) {
        int index;
        double prob;

        // if smoothing is none and representation is not log, get each n-gram and multiply each probability together
        if (smo.equals(Smoothing.NONE) && rep.equals(Representation.PROBABILITY)) {
            prob = 1.0;
            for (int i = 1; i <= sequence.size(); i++) {
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }
                // if there is no n-gram, multiply probability by zero
                if (!nGrams.containsKey(sequence.subList(index, i))) {
                    prob *= 0.0;
                } else {
                    prob *= nGrams.get(sequence.subList(index, i));
                }
            }
            return prob;
            // if smoothing is Laplace and representation is not log, get each n-gram and multiply each probability together
        } else if (smo.equals(Smoothing.LAPLACE) && rep.equals(Representation.PROBABILITY)) {
            prob = 1.0;
            for (int i = 1; i <= sequence.size(); i++) {
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }
                // if there is no n-gram, multiply the probability by 1/(# + v)
                // this is the probability for any absent n-gram in Laplace smoothing
                if (!nGrams.containsKey(sequence.subList(index, i))) {
                    if(!conditionals.containsKey(sequence.subList(index, i - 1))) {
                        prob *= (1.0 / v);
                    } else {
                        prob *= (1.0 / ((double) conditionals.get(sequence.subList(index, i - 1)) + v));
                    }
                } else {
                    prob *= nGrams.get(sequence.subList(index, i));
                }
            }
            return prob;
            // if smoothing is none and representation is log, get each n-gram and add the log of each probability together
        } else if (smo.equals(Smoothing.NONE) && rep.equals(Representation.LOG_PROBABILITY)) {
            prob = 0.0;
            for (int i = 1; i <= sequence.size(); i++) {
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }
                // if there is no n-gram, add Log(0) to probability
                if (!nGrams.containsKey(sequence.subList(index, i))) {
                    prob += Math.log(0.0);
                } else {
                    prob += Math.log(nGrams.get(sequence.subList(index, i)));
                }
            }
            return prob;
            // if smoothing is none and representation is log, get each n-gram and add the log of each probability together
        } else if (smo.equals(Smoothing.LAPLACE) && rep.equals(Representation.LOG_PROBABILITY)) {
            prob = 0.0;
            for (int i = 1; i <= sequence.size(); i++) {
                if (i - nSize < 0) {
                    index = 0;
                } else {
                    index = i - nSize;
                }
                // if there is no n-gram, add Log(1/(# + v)) to probability
                if (!nGrams.containsKey(sequence.subList(index, i))) {
                    if(!conditionals.containsKey(sequence.subList(index, i - 1))) {
                        prob += Math.log(1.0 / v);
                    } else {
                        prob += Math.log(1.0 / ((double) conditionals.get(sequence.subList(index, i - 1)) + v));
                    }
                } else {
                    prob += Math.log(nGrams.get(sequence.subList(index, i)));
                }
            }
            return prob;
        } else {
            System.out.println("You should not be here.");
            return 0.0;
        }
    }
}