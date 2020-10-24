# MarkovModel
> Implementation of a Markov Model in Python following the Scikit-learn API.

This library can be thought of as an unsupervised machine learning method for
dealing with Markov Processes.

More specifically, it expects as input a list of Markov Chains. It then uses
this data to estimate the most likely Transition Matrix to have generated the
given dataset.

Once the Transition Matrix has been estimated, it can then use it in order to
predict the transition probability for one given node to another node from the
Markov Chain. The Transition Probability Matrix is calculated using Monte Carlo
simulations.

You can assess the quality of the generated model by checking its score, which
is the MacFadden's Pseudo-R2 in relation to the null model.

## Basic usage

``` python
>>> markov_chain_list = [[0,1,2],[0,1,1,2]]
>>> markov_model = MarkovModel(n_states=3, start_state=0, end_state=2, target_state=2) 
>>> markov_model.fit(markov_chain_list)
>>> markov_model.transition_matrix
array([[0.        , 1.        , 0.        ],
       [0.        , 0.33333333, 0.66666667],
       [0.        , 0.        , 1.        ]])
>>> markov_model.predict(0) # Probability of state 0 reaching state 2
0.87
>>> markov_model.score(markov_chain_list)
0.6792859981553997
```

## Caveat

Take note that this is a work in progress!

This has been adapted from a particular use case: analyzing user interaction in a website.
So there's still a lot of room for improvement and making this more generally applicable.

If you are interested in contributing feel free to message me.
