# %%
import numpy as np
import pandas as pd
import concurrent.futures
from itertools import chain, repeat
from toolz import curry, sliding_window
from src.data.load_clean_data import get_events_df

import time

# %%
df, cat_dict = get_events_df("./data/events_v5.csv", nrows=500_000,
                             usecols=['person',
                                      'timestamp',
                                      'session',
                                      'event'])
event_dict = cat_dict['event']
event_dict[30] = 'session-start'
event_dict[31] = 'session-end'

# %%
df.tail()

# %%
print(df['timestamp'].min())
print(df['timestamp'].max())


# %%
class MarkovModel(object):
    """docstring for MarkovModel."""

    def __init__(self, n_states=None, transition_matrix=None):
        super(MarkovModel, self).__init__()
        if n_states is not None and transition_matrix is None:
            self.n_states = n_states
            self.transition_matrix = np.zeros((self.n_states, self.n_states))
            self.prediction_matrix = np.zeros(self.n_states)
        elif transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.n_states = len(self.transition_matrix)
            self.prediction_matrix = np.zeros(self.n_states)
        else:
            raise ValueError("Either `n_states` or `transition_matrix`"
                             "are required for building a MarkovModel.")

    def fit(self, start_state=0,
              target_state=1, max_chain_length=1_000,
              n_training_chains=10_000, end_state='end_state', data=None):

        if end_state == 'end_state':
            end_state = self.n_states - 1

        # Check if self.transition_matrix isn't already "trained"
        if np.allclose(self.transition_matrix,
                       np.zeros((self.n_states, self.n_states))):
            events_list = get_all_users_session_journeys(data,
                                                         n_states=self.n_states,
                                                         end_state=end_state)

            tm = generate_transition_matrix(list(events_list),
                                       n_states=self.n_states,
                                       end_state=end_state)

            # Fix end_event transition to itself with prob 1.0
            # due to domain application logic:
            # after a user leaves the site there should be no
            # more transitions
            tm = list(tm)
            tm[end_state] = np.insert(np.zeros(self.n_states-1),
                                      end_state, 1.0)

            self.transition_matrix = np.array(list(tm))

        # Simulate markov chains starting from start_state
        aux_chains = simulate_chain_list(start_state,
                                         self.transition_matrix,
                                         max_chain_length,
                                         n_training_chains)

        median_chain_length = np.median(list(map(len, list(aux_chains))))
        # Having the mean chain length is important for setting a
        # more reasonable maximum chain lenght for the simulations.

        simulate_chain_list_from_state = \
            simulate_chain_list(transition_matrix=self.transition_matrix,
                                max_chain_length=median_chain_length,
                                n_simulations=n_training_chains)
        # Simulate markov chains for each possible state
        with concurrent.futures.ProcessPoolExecutor() as executor:
            training_chains = \
                map(simulate_chain_list_from_state,
                             range(self.n_states))

        # Calculate probability of seeing the `target_state` in each
        # markov chain for each possible state
        probability_to_target_state = probability_to_state(target_state)
        proba = map(probability_to_target_state, training_chains)

        self.prediction_matrix = np.array(list(proba))

        return self

    def predict(self, x):
        return self.prediction_matrix[x]


# %%
@curry
def is_in(x, l):
    return x in set(l)

# %%
@curry
def simulate_chain(chain, transition_matrix, max_chain_length=100):
    # Return if reaches max_chain_length
    if len(chain) == max_chain_length:
        return chain

    current_state = chain[-1]
    transition_prob = transition_matrix[current_state]
    next_state = np.random.choice(np.arange(len(transition_matrix)),
                                  p=transition_prob)

    new_chain = np.append(chain, next_state)

    # Returns if reaches ending state
    if current_state == next_state \
       and transition_matrix[next_state][next_state] == 1.0:
        return chain

    return simulate_chain(new_chain, transition_matrix, max_chain_length)


# %%
def test_simulte_chain():
    tm = np.array([[0.5, 0.45, 0.05], [0.2, 0.79, 0.01], [0.0, 0.0, 1.0]])
    chain = simulate_chain([0], list(tm), 10)
    has_2 = is_in(2, chain)
    print(chain)
    print(has_2)


# %%
test_simulte_chain()

# %%
@curry
def simulate_chain_list(initial_state, transition_matrix,
                        max_chain_length, n_simulations):
    multi_chain_generator = simulate_chain(transition_matrix=transition_matrix,
                                           max_chain_length=max_chain_length)
    chains = map(multi_chain_generator,
                 repeat(np.array([initial_state]), n_simulations))
    return chains


# %%
def test_simulate_chain_list():
    tm = np.array([[0.5, 0.45, 0.05], [0.2, 0.79, 0.01], [0.0, 0.0, 1.0]])
    # chains = map(lambda x: simulate_chain([x], list(tm), 20), np.repeat(0, 40))
    # print(np.mean(list(map(len, list(chains)))))
    chains = simulate_chain_list(0, tm, 100, 1_000)
    print(np.mean(list(map(len, list(chains)))))
    # assert mean_chain_length < 100
    chains = simulate_chain_list(0, tm, 10, 5)
    print(list(chains))


# %%
test_simulate_chain_list()


# %%
@curry
def probability_to_state(state, chain_list):
    """
        Given a list of Markov Chains, checks wether the
        given `state` is present in each one of the lists.

        Then, based on this count, estimates the probability
        of finding the target state.
    """

    state_is_in = is_in(state)
    bool_list = list(map(state_is_in, chain_list))
    proba = bool_list.count(True)/len(bool_list)

    return proba


# %%
def markov_mean(m):
    return np.sum(m, axis=0)/np.sum(m)


# %%
def zero_division_guard(f):
    def decorated_f(row):
        return row if sum(row) == 0.0 else f(row)
    return decorated_f


# %%
@zero_division_guard
def calculate_probability(row):
    return row/np.sum(row)


# %%
def generate_transition_matrix(states, n_states, end_state=None):
    """
    Retrieves transition matrix from given states.

    Given a list of states, a zip containing the original list and a shifted
    version will be created. Each pair in this zip shall denote the present
    and the next event that occured after it.
    With that, a transition matrix can be calculated iteratively.

    Parameters
    ----------
    states : array
        Array of shape (n_states), where n_states is the number of states
        expected for this model

    Returns
    -------
    transition_matrix : ndarray
        Result of shape (n_states, n_states).
    """

    transition_matrix = np.zeros((n_states, n_states))
    present_next_event = sliding_window(2, states)

    for idx in list(present_next_event):
        transition_matrix[idx] += 1

    # TODO:
    # Treat NaNs before continuing
    transition_matrix = map(calculate_probability,
                            transition_matrix)

    # If any of the rows (states) have a sum probability of 0.0
    # assume this state points to the end_state with probability 1.0
    if end_state is not None:
        if end_state == 'self':
            transition_matrix = map(lambda idx, row:
                                    np.insert(np.zeros(n_states-1), idx, 1.0)
                                    if sum(row) == 0.0
                                    else np.array(row),
                                    enumerate(transition_matrix))
        elif type(end_state) is int:
            transition_matrix = map(lambda row:
                                    np.insert(np.zeros(n_states-1),
                                              end_state, 1.0)
                                    if sum(row) == 0.0
                                    else np.array(row),
                                    transition_matrix)
        else:
            # TODO: Accept an array as well with different "end estates"?
            raise TypeError("Parameter end_state must be None,"
                            "a string `self` or an integer.")

    return transition_matrix


# %%
# Test generate_transition_matrix
def test_3_events_list_5_total():
    # Possible events: 0,1,2,3,4
    events_list = [1, 1, 2]

    transition = generate_transition_matrix(events_list, n_states=5, end_state=4)
    transition = list(transition)

    expected = np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0]])
    # expected = np.reshape(expected, (5,5))

    np.testing.assert_array_almost_equal(expected, transition)
    for row in expected:
        assert (np.sum(row) == 1.0)


# %%
# Run test
test_3_events_list_5_total()


# %%
def add_beginning_end_events(session_event_pairs,
                             beginning_event, end_event):
    """
    Retrieves events list, adding beginning and end of session.

    The input is a pair (tuple) of tuples. Each pair is a tuple holding
    information about the session (first position) and an event (second
    position).
    The first element of this pair of tuples corresponds to a
    (session, event) at time t, and the second at a time t+1.
    Comparing the session value from both pairs we can know when a session
    has ended.
    If the sesison hasn't ended yet (the session element from both pairs
    holds the same value) then we return the event corresponding to the
    first pair. Otherwise (the session element from both pairs holds
    different values) we return not only the event from the first tuple
    but also indicators that the session has ended and a new one should
    start next.

    Parameters
    ----------
    session_event_pairs : tuple

    Returns
    -------
    tuple
        Tuple with corresponding events.
    """
    first_pair, second_pair = session_event_pairs
    if first_pair[0] == second_pair[0]:
        return (first_pair[1])
    else:
        return (first_pair[1], end_event, beginning_event)


# %%
@curry
def get_time_sorted_events_list(df, n_states):
    """
    Given a dataframe returns a list of events, ordered by timestamp.
    """
    end_event = n_states - 1
    beginning_event = n_states - 2

    events_list = list([beginning_event])

    events_list.extend(df.reset_index()
                         .set_index("timestamp")
                         .sort_index()
                         ['event'].values)

    events_list.append(end_event)

    return events_list


# %%
def get_mean_transition_matrix(m):
    """
    Calculate mean transition matrix from given matrices.

    Parameters
    ----------
    m : iterator
        An iterator that resolves into an array of shape (n_states),
        where n_states is the number of states expected for this model.

    Returns
    -------
    mean_matrix : iterator
        Resolves into result of shape (n_states, n_states).
    """
    zip_rows = zip(*m)
    mean_row = curry(np.mean, axis=0)
    mean_matrix = map(mean_row, zip_rows)
    return mean_matrix


# %%
@curry
def get_idxs(df, k):
    """
        From an index sorted df, get left and right indexes given search key k.
    """
    return (df.index.searchsorted(k, side="left"),
            df.index.searchsorted(k, side="right"))


@curry
def get_view(df, idx):
    return df.iloc[idx[0]:idx[1]]


# %%
@curry
def get_user_session_journey(df, n_states, end_state='end_state'):
    unique_sessions = df['session'].unique()
    session_df = df.reset_index(drop=True).set_index('session').sort_index()

    # Two more events to account for the artificial events
    # representing the beginning and the end of a session
    if end_state == 'end_state':
        # By convetion, the 'end_state' is the last state
        end_state = n_states - 1

    get_idxs_session_df = get_idxs(session_df)
    idxs = map(get_idxs_session_df, unique_sessions)

    get_view_session_df = get_view(session_df)
    views = map(get_view_session_df, idxs)

    n_time_sorted_events_list = get_time_sorted_events_list(n_states=n_states)

    events_lists = map(n_time_sorted_events_list, views)

    chained_events = chain(*events_lists)
    # Instead of getting multiple transition matrices
    # and then average them, get only one transition
    # matrix
    # transition_matrices = map(lambda l:
    #                           generate_transition_matrix(
    #                             l,
    #                             n_states,
    #                             end_state),
    #                           events_lists)

    # # This fails when one of the columns' sum equals 0.0:
    # # mean_matrix = np.mean(np.array([*transition_matrices]), axis=0)
    # mean_matrix = get_mean_transition_matrix(transition_matrices)
    transition_matrix = generate_transition_matrix(list(chain(*events_lists)),
                                              n_states=n_states,
                                              end_state=end_state)

    return chained_events


# %%
def test_user_4_sessions_2_events_expected_result():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2],
            #"event":   [5, 1, 2, 6,
                       # 5, 1, 2, 6,
                       # 5, 2, 2, 6,
                       # 5, 2, 2, 6],
            # accounting for
            # sessiong beginning and ending
            "timestamp": [pd.Timestamp(1513393355.5, unit='s'),
                          pd.Timestamp(1513493355.5, unit='s'),
                          pd.Timestamp(1514393355.5, unit='s'),
                          pd.Timestamp(1514493355.5, unit='s'),
                          pd.Timestamp(1515393355.5, unit='s'),
                          pd.Timestamp(1515493355.5, unit='s'),
                          pd.Timestamp(1516393355.5, unit='s'),
                          pd.Timestamp(1516493355.5, unit='s')]}

    df = pd.DataFrame.from_dict(user)
    n_states = 5+2 # 2 artificial states: start_session and end_session
    end_state = 6

    events_list = get_user_session_journey(df, n_states=n_states,
                                                 end_state=end_state)
    transition_matrix = generate_transition_matrix(list(events_list),
                                              n_states=n_states,
                                              end_state=end_state)

    # Fix end_event transition to itself with prob 1.0
    # due to domain application logic:
    # after a user leaves the site there should be no
    # more transitions
    transition_matrix = list(transition_matrix)
    transition_matrix[end_state] = np.insert(np.zeros(n_states-1),
                                             end_state, 1.0)


    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 2/6, 0.0, 0.0, 0.0, 4/6],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 1/2, 1/2, 0.0, 0.0, 0.0, 0.0], # start_event
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) # end_event

    np.testing.assert_array_almost_equal(expected, list(transition_matrix))


def test_user_4_sessions_2_events_sum_to_one():
    user = {"person": [1, 1, 1, 1, 1, 1, 1, 1],
            "session": [0, 0, 1, 1, 2, 2, 3, 3],
            "event": [1, 2, 1, 2, 2, 2, 2, 2],
            "timestamp": [pd.Timestamp(1513393355.5, unit='s'),
                          pd.Timestamp(1513493355.5, unit='s'),
                          pd.Timestamp(1514393355.5, unit='s'),
                          pd.Timestamp(1514493355.5, unit='s'),
                          pd.Timestamp(1515393355.5, unit='s'),
                          pd.Timestamp(1515493355.5, unit='s'),
                          pd.Timestamp(1516393355.5, unit='s'),
                          pd.Timestamp(1516493355.5, unit='s')]}

    df = pd.DataFrame.from_dict(user)

    n_states = 5+2 # 2 artificial states: start_session and end_session
    end_state = 6

    events_list = get_user_session_journey(df, n_states=n_states,
                                                 end_state=end_state)
    transition_matrix = generate_transition_matrix(list(events_list),
                                              n_states=n_states,
                                              end_state=end_state)

    # Fix end_event transition to itself with prob 1.0
    # due to domain application logic:
    # after a user leaves the site there should be no
    # more transitions
    transition_matrix = list(transition_matrix)
    transition_matrix[end_state] = np.insert(np.zeros(n_states-1),
                                             end_state, 1.0)

    for row in list(transition_matrix):
        assert np.sum(row) == 1.0


def test_one_user_4_sessions_2_events_expected_result_all_users():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2],
            #"event":   [5, 1, 2, 6,
                       # 5, 1, 2, 6,
                       # 5, 2, 2, 6,
                       # 5, 2, 2, 6],
            # accounting for
            # sessiong beginning and ending
            "timestamp": [pd.Timestamp(1513393355.5, unit='s'),
                          pd.Timestamp(1513493355.5, unit='s'),
                          pd.Timestamp(1514393355.5, unit='s'),
                          pd.Timestamp(1514493355.5, unit='s'),
                          pd.Timestamp(1515393355.5, unit='s'),
                          pd.Timestamp(1515493355.5, unit='s'),
                          pd.Timestamp(1516393355.5, unit='s'),
                          pd.Timestamp(1516493355.5, unit='s')]}

    df = pd.DataFrame.from_dict(user)

    n_states = 5+2 # 2 artificial states: start_session and end_session
    end_state = 6

    events_list = get_all_users_session_journeys(df, n_states=n_states,
                                                 end_state=end_state)
    transition_matrix = generate_transition_matrix(list(events_list),
                                              n_states=n_states,
                                              end_state=end_state)

    # Fix end_event transition to itself with prob 1.0
    # due to domain application logic:
    # after a user leaves the site there should be no
    # more transitions
    transition_matrix = list(transition_matrix)
    transition_matrix[end_state] = np.insert(np.zeros(n_states-1),
                                             end_state, 1.0)

    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 2/6, 0.0, 0.0, 0.0, 4/6],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 1/2, 1/2, 0.0, 0.0, 0.0, 0.0],  # start_event
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # end_event

    np.testing.assert_array_almost_equal(expected, list(transition_matrix))


def test_two_users_4_sessions_2_events_expected_result_all_users():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2, 2, 2],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3,
                         0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2,
                         1, 2, 1, 2, 2, 2, 2, 2],
            #"event":   [5, 1, 2, 6,
                       # 5, 1, 2, 6,
                       # 5, 2, 2, 6,
                       # 5, 2, 2, 6],
            # accounting for
            # sessiong beginning and ending
            "timestamp": [pd.Timestamp(1513393355.5, unit='s'),  # person 1
                          pd.Timestamp(1513493355.5, unit='s'),  # timestamps
                          pd.Timestamp(1514393355.5, unit='s'),
                          pd.Timestamp(1514493355.5, unit='s'),
                          pd.Timestamp(1515393355.5, unit='s'),
                          pd.Timestamp(1515493355.5, unit='s'),
                          pd.Timestamp(1516393355.5, unit='s'),
                          pd.Timestamp(1516493355.5, unit='s'),
                          pd.Timestamp(1513393355.5, unit='s'),  # person 2
                          pd.Timestamp(1513493355.5, unit='s'),  # timestamps
                          pd.Timestamp(1514393355.5, unit='s'),
                          pd.Timestamp(1514493355.5, unit='s'),
                          pd.Timestamp(1515393355.5, unit='s'),
                          pd.Timestamp(1515493355.5, unit='s'),
                          pd.Timestamp(1516393355.5, unit='s'),
                          pd.Timestamp(1516493355.5, unit='s')]}

    df = pd.DataFrame.from_dict(user)

    n_states = 5+2 # 2 artificial states: start_session and end_session
    end_state = 6

    events_list = get_all_users_session_journeys(df, n_states=n_states,
                                                 end_state=end_state)

    transition_matrix = generate_transition_matrix(list(events_list),
                                              n_states=n_states,
                                              end_state=end_state)

    # Fix end_event transition to itself with prob 1.0
    # due to domain application logic:
    # after a user leaves the site there should be no
    # more transitions
    transition_matrix = list(transition_matrix)
    transition_matrix[end_state] = np.insert(np.zeros(n_states-1),
                                             end_state, 1.0)

    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 2/6, 0.0, 0.0, 0.0, 4/6],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 1/2, 1/2, 0.0, 0.0, 0.0, 0.0], # start_event
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) # end_event

    np.testing.assert_array_almost_equal(expected, list(transition_matrix))


# %%
test_user_4_sessions_2_events_sum_to_one()

# %%
test_user_4_sessions_2_events_expected_result()


# %%
def get_all_users_session_journeys(df, n_states, end_state):
    person_df = df.set_index('person').sort_index()
    unique_persons = person_df.index.unique()

    get_idxs_person_df = get_idxs(person_df)
    person_idxs = map(get_idxs_person_df, unique_persons)

    get_view_person_df = get_view(person_df)
    person_views = map(get_view_person_df, person_idxs)

    n_states_journeys = get_user_session_journey(n_states=n_states,
                                                 end_state=end_state)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        users_journeys = map(n_states_journeys, person_views)

    chained_journeys = chain(*users_journeys)

    return chained_journeys


# %%
test_one_user_4_sessions_2_events_expected_result_all_users()

# %%
test_two_users_4_sessions_2_events_expected_result_all_users()

# %%
def test_markov_proba():
    tm = \
        np.array([[0.5, 0.3, 0.05, 0.15],
                  [0.2, 0.59, 0.01, 0.2],
                  [0.01, 0.01, 0.0, 0.98],
                  [0.0, 0.0, 0.0, 1.0]])
    start_state = 0
    max_chain_length = 1_000
    n_training_chains = 10_000
    n_states = 4
    target_state = 2

    markov_model = MarkovModel(transition_matrix=tm)

    np.testing.assert_almost_equal(tm, markov_model.transition_matrix)
    assert markov_model.n_states == n_states

    markov_model.fit(start_state=start_state,
                       target_state=target_state,
                       max_chain_length=max_chain_length,
                       n_training_chains=n_training_chains)
    print(markov_model.prediction_matrix)


# %%
start = time.time()
test_markov_proba()
end = time.time()
print(end-start)

# %%
start = time.time()
markov_model = MarkovModel(n_states=32).fit(data=df, start_state=30,
                                              target_state=18,
                                              # conversion
                                              max_chain_length=1_000,
                                              n_training_chains=20_000)
end = time.time() - start

# %%
# Sequential: 500_000 => 6.45
end/60

# %%
# Threds: 500_00 => 8.27
end/60

# %%
# Process: 500_000 => 5.25 min
end/60

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df_journey_mm = pd.DataFrame(markov_model.transition_matrix,
                          columns=cat_dict['event'].values(),
                          index=cat_dict['event'].values())

# %%
fig, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(df_journey_mm, annot=True,
            linewidth=0.1, linecolor="white",
            ax=ax)
plt.show(fig)

# %%
for (idx, val) in enumerate(markov_model.prediction_matrix):
    print("{}:\t\t{:.3f}".format(event_dict[idx], val))
