from math import log2
from utils import load_test_set, load_training_set, preprocess_text
import numpy
from random import randrange

# Some basic thoughts
    # I'm going to try to avoid getting caught up in making this code too general and just focus on getting the assignment done
    # Some light optimization/ storing of intermediate values makes sense, but I'll try to not go overboard...

def classify_review(document, ):
    pass


def main():
    # from the assignment description...
    train_tot_pos = 12500
    train_tot_neg = 12500

    # programmer's choice...
    train_pos_perc = 0.2
    train_neg_perc = 0.2
    test_pos_perc = 0.2
    test_neg_perc = 0.2
    # sanity check...
    if train_pos_perc + train_neg_perc > 1.0:
        print(f"ERROR: Bad training percentages: ({train_pos_perc +  train_neg_perc})")
        return
    if test_pos_perc + test_neg_perc > 1.0:
        print(f"ERROR: Bad testing percentages: ({test_pos_perc +  test_neg_perc})")
        return

    (pos_train, neg_train, vocab) = load_training_set(train_pos_perc, train_neg_perc)
    (pos_test,  neg_test)         = load_test_set(test_pos_perc, test_neg_perc)

    prior_prob_pos = len(pos_train) / (len(pos_train) + len(neg_train))
    prior_prob_neg = len(neg_train) / (len(pos_train) + len(neg_train))

    # We'll establish the convention here that in the vocab_counts dict,
    # the first entry for a given word will be the POSITIVE counts, and the second entry will be the NEGATIVE counts
    # i.e. 
    #   vocab_counts[word1][0] = # occurences in postive training set
    #   vocab_counts[word1][1] = # occurences in negative training set
    vocab_counts = {}
    vocab_total_pos = 0
    vocab_total_neg = 0
    for word in vocab:
        vocab_counts[word] = [0,0]

    for doc in pos_train:
        for word in doc:
            vocab_counts[word][0] += 1
            vocab_total_pos += 1

    for doc in neg_train:
        for word in doc:
            vocab_counts[word][1] += 1
            vocab_total_neg += 1

    #print(f"pos_train: {pos_train}\n\n\n\n\n\n")
    #print(f"neg_train: {neg_train}\n\n\n\n")
    #print(f"vocab: {vocab}\n\n\n\n\n\n")
    print(f"Vocab total pos: {vocab_total_pos}")
    print(f"Vocab total neg: {vocab_total_neg}")
    print(f"Prior probability positive: {prior_prob_pos}")
    print(f"Prior probability negative: {prior_prob_neg}")
    print(f"Pos train: {len(pos_train)}")
    print(f"Neg train: {len(neg_train)}")
    print(f"Pos test: {len(pos_test)}")
    print(f"Neg test: {len(neg_test)}")

    #for word in vocab:
    #    print(f"{word}: # pos: {vocab_counts[word][0]}, # neg: {vocab_counts[word][1]}")

    # try laplace smoothing???

    num_corr = 0
    num_incorr = 0
    num_test = 0
    print("NAIVE IMPLEMENTATION:")
    print(f"\tPOSITIVE TEST:")
    for doc in pos_test:
        num_test += 1
        used = {}
        pos_prob = numpy.longdouble(prior_prob_pos) # Pr(y_0)
        neg_prob = numpy.longdouble(prior_prob_neg) # Pr(y_1)
        for word in doc:
            if True: #word not in used: # avoid double counting...
                used[word] = True 
                if word in vocab_counts:
                    if vocab_counts[word][0] > 0:
                        pos_prob *= numpy.longdouble(vocab_counts[word][0] / vocab_total_pos) # Product: Pr(w_k | y_0) 
                    if vocab_counts[word][1] > 0:
                        neg_prob *= numpy.longdouble(vocab_counts[word][1] / vocab_total_neg) # Product: Pr(w_k | y_1)
                # if the word wasn't in our training set, ignore it (https://campuswire.com/c/G78D7CCD1/feed/174)
        #print(f"pos prob: {pos_prob}, neg_prob: {neg_prob}")
        if pos_prob > neg_prob:
            #print("Classification: POSITIVE")
            num_corr += 1
        elif pos_prob < neg_prob: 
            #print("Classification: NEGATIVE")
            num_incorr += 1
        else: # probabiltiies most likely got smashed down to 0...flip a coin
            #print(f"\tHeyo: {pos_prob}, {neg_prob}")
            if randrange(0,2) == 0:
                num_corr += 1
            else: 
                num_incorr += 1
    print(f"\tPositive score: {num_corr / num_test}")

    num_corr = 0
    num_incorr = 0
    num_test = 0
    print(f"\tNEGATIVE TEST:")
    for doc in neg_test:
        num_test += 1
        used = {}
        pos_prob = numpy.longdouble(prior_prob_pos) # Pr(y_0)
        neg_prob = numpy.longdouble(prior_prob_neg) # Pr(y_1)
        for word in doc:
            if True: #word not in used: # avoid double counting...
                used[word] = True 
                if word in vocab_counts:
                    if vocab_counts[word][0] > 0: # if the word wasn't in our training set, ignore it (https://campuswire.com/c/G78D7CCD1/feed/174)
                        pos_prob *= numpy.longdouble(vocab_counts[word][0] / vocab_total_pos) # Product: Pr(w_k | y_0) 
                    if vocab_counts[word][1] > 0:
                        neg_prob *= numpy.longdouble(vocab_counts[word][1] / vocab_total_neg) # Product: Pr(w_k | y_1)
        #print(f"pos prob: {pos_prob}, neg_prob: {neg_prob}")
        if pos_prob > neg_prob:
            #print("Classification: POSITIVE")
            num_incorr += 1
        elif pos_prob < neg_prob: 
            #print("Classification: NEGATIVE")
            num_corr += 1
        else: # probabiltiies most likely got smashed down to 0...flip a coin
            #print(f"\tHeyo: {pos_prob}, {neg_prob}")
            if randrange(0,2) == 0:
                num_incorr += 1
            else: 
                num_corr += 1
    print(f"\tNegative score: {num_corr / num_test}")
    print("Log TRICK:")
    print(f"\tPOSITIVE TEST:")

    num_corr = 0
    num_incorr = 0
    num_test = 0
    for doc in pos_test:
        num_test += 1
        used = {}
        pos_prob = numpy.longdouble(log2(prior_prob_pos)) # Pr(y_0)
        neg_prob = numpy.longdouble(log2(prior_prob_neg)) # Pr(y_1)
        for word in doc:
            if word not in used: # avoid double counting...
                used[word] = True 
                if word in vocab_counts:
                    if vocab_counts[word][0] > 0:
                        pos_prob += log2(numpy.longdouble(vocab_counts[word][0] / vocab_total_pos)) # Product: Pr(w_k | y_0) 
                    if vocab_counts[word][1] > 0:
                        neg_prob += log2(numpy.longdouble(vocab_counts[word][1] / vocab_total_neg)) # Product: Pr(w_k | y_1)
                # if the word wasn't in our training set, ignore it (https://campuswire.com/c/G78D7CCD1/feed/174)
        #print(f"pos prob: {pos_prob}, neg_prob: {neg_prob}")
        if pos_prob > neg_prob:
            #print("Classification: POSITIVE")
            num_corr += 1
        elif pos_prob < neg_prob: 
            #print("Classification: NEGATIVE")
            num_incorr += 1
        else: # probabiltiies most likely got smashed down to 0...flip a coin
            #print(f"\tHeyo: {pos_prob}, {neg_prob}")
            if randrange(0,2) == 0:
                num_corr += 1
            else: 
                num_incorr += 1
    print(f"\tPositive score: {num_corr / num_test}")

    num_corr = 0
    num_incorr = 0
    num_test = 0
    print(f"\tNEGATIVE TEST:")
    for doc in neg_test:
        num_test += 1
        used = {}
        pos_prob = numpy.longdouble(log2(prior_prob_pos)) # Pr(y_0)
        neg_prob = numpy.longdouble(log2(prior_prob_neg)) # Pr(y_1)
        for word in doc:
            if word not in used: # avoid double counting...
                used[word] = True 
                if word in vocab_counts:
                    if vocab_counts[word][0] > 0: # if the word wasn't in our training set, ignore it (https://campuswire.com/c/G78D7CCD1/feed/174)
                        pos_prob += log2(numpy.longdouble(vocab_counts[word][0] / vocab_total_pos)) # Product: Pr(w_k | y_0) 
                    if vocab_counts[word][1] > 0:
                        neg_prob += log2(numpy.longdouble(vocab_counts[word][1] / vocab_total_neg)) # Product: Pr(w_k | y_1)
        #print(f"pos prob: {pos_prob}, neg_prob: {neg_prob}")
        if pos_prob > neg_prob:
            #print("Classification: POSITIVE")
            num_incorr += 1
        elif pos_prob < neg_prob: 
            #print("Classification: NEGATIVE")
            num_corr += 1
        else: # probabiltiies most likely got smashed down to 0...flip a coin
            #print(f"\tHeyo: {pos_prob}, {neg_prob}")
            if randrange(0,2) == 0:
                num_incorr += 1
            else: 
                num_corr += 1
    print(f"\tNegative score: {num_corr / num_test}")



if __name__ == "__main__":
    main()