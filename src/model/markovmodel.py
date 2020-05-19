# %%
import numpy as np
import pandas as pd
import concurrent
from itertools import chain, tee
from src.data.load_clean_data import get_events_df

import time

# %%
df, cat_dict = get_events_df("./data/events_v5.csv", nrows=500_000,
                             usecols=['person',
                                      'timestamp',
                                      'session',
                                      'event'])
df.set_index('person', inplace=True)
df.sort_index(inplace=True)
event_dict = cat_dict['event']
event_dict[30] = 'session-start'
event_dict[31] = 'session-end'


# %%
class MarkovModel(object):
    """docstring for MarkovModel."""

    def __init__(self, n_states=None, transition_matrix=None):
        super(MarkovModel, self).__init__()
        if n_states is not None and transition_matrix is None:
            self.transition_matrix = np.zeros((n_states, n_states))
            self.prediction_matrix = np.zeros(n_states)
        elif transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.n_states = len(transition_matrix)
            self.prediction_matrix = np.zeros(self.n_states)
        else:
            raise ValueError("Either n_states or transition_matrix"
                             "are required for building a MarkovModel.")

    def train(self, x):
        # Check if self.transition_matrix isn't already "trained"
        if np.allclose(self.transition_matrix,
                       np.zeros((self.n_state, self.n_states))):
            # If interested in the transition matrix for only one user
            if 'person' in x.columns:
                self.transition_matrix = \
                    get_user_session_journey(x, n_states=n_states)
            # If interested in the transition matrix for all users
            # TODO: is this checking really necessary?
            elif 'session' in x.columns:
                self.transition_matrix = get_mean_transition_matrix(
                                            get_all_users_session_journeys(x))

        self.prediction_matrix = np.zeros(len(self.transition_matrix))

        return self

    def predict(self, x):
        return self.prediction_matrix[x]


# %%
# TODO: change session end transition probability. Make it point to itself

# %%
def markov_mean(m):
    if np.sum(m) > 0.0:
        return np.sum(m, axis=0)/np.sum(m)
    return m[0]


# %%
def calculate_probability(row):
    if np.count_nonzero(row) > 0:
        return row/sum(row)
    return row


# %%
def get_transition_matrix(events, n_unique_events=30):
    """
    Retrieves transition matrix from given events.

    Given a list of events, a zip containing the original list and a shifted
    version will be created. Each pair in this zip shall denote the present
    and the next event that occured after it.
    With that, a transition matrix can be calculated iteratively.

    Parameters
    ----------
    events : array
        Array of shape (n_states), where n_states is the number of states
        expected for this model

    Returns
    -------
    transition_matrix : ndarray
        Result of shape (n_states, n_states).
    """

    transition_matrix = np.zeros((n_unique_events, n_unique_events))
    present_next_event = zip(events[:-1], events[1:])

    for idx in present_next_event:
        transition_matrix[idx] += 1

    transition_matrix = map(calculate_probability,
                            transition_matrix)

    return transition_matrix


# %%
# Test get_transition_matrix
def test_3_events_list_5_total():
    # Possible events: 0,1,2,3,4
    events_list = [1, 1, 2]

    transition = get_transition_matrix(events_list, n_unique_events=5)
    transition = np.array(list(transition))

    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])

    np.testing.assert_array_almost_equal(transition, expected)
    for row in expected:
        assert (np.sum(row) == 1.0) or (np.sum(row) == 0.0)


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
def get_time_sorted_events_list(df, n_unique_events):
    """
    Given a dataframe returns a list of events, ordered by timestamp.
    """
    end_event = n_unique_events - 1
    beginning_event = n_unique_events - 2

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
    mean_matrix : ndarray
        Result of shape (n_states, n_states).
    """
    zip_rows = zip(*m)
    mean_matrix = map(markov_mean, zip_rows)
    return mean_matrix


# %%
def get_user_session_journey(df, n_unique_events=30):
    unique_sessions = df['session'].unique()
    session_df = df.reset_index(drop=True).set_index('session').sort_index()

    # Two more events to account for the artificial events
    # representing the beginning and the end of a session
    n_unique_events += 2

    idxs = map(lambda s:
               (session_df.index.searchsorted(s, side="left"),
                session_df.index.searchsorted(s, side="right")),
               unique_sessions)

    events_lists = map(lambda idx:
                       get_time_sorted_events_list(
                          session_df.iloc[idx[0]:idx[1]],
                          n_unique_events),
                       idxs)
    # print(*events_lists)
    transition_matrices = map(lambda l:
                              get_transition_matrix(
                                l,
                                n_unique_events),
                              events_lists)

    # Wrong!
    # This fails when one of the columns' sum equals 0.0
    # mean_matrix = np.mean(np.array([*transition_matrices]), axis=0)
    mean_matrix = get_mean_transition_matrix(transition_matrices)

    return mean_matrix


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
    mean_matrix = get_user_session_journey(df, 5)

    expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1/4, 0.0, 0.0, 0.0, 3/4],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1/2, 1/2, 0.0, 0.0, 0.0, 0.0], # beginning_event
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) # end_event

    np.testing.assert_allclose(list(mean_matrix), expected)


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
    mean_matrix = get_user_session_journey(df, 5)

    for row in mean_matrix:
        assert (np.sum(row) == 1.0) or (np.sum(row) == 0.0)


# %%
test_user_4_sessions_2_events_sum_to_one()

# %%
test_user_4_sessions_2_events_expected_result()


# %%
def get_all_users_session_journeys(df):
    unique_persons = df.index.unique()

    person_idxs = map(lambda p: (df.index.searchsorted(p, side='left'),
                                 df.index.searchsorted(p, side='right')),
                      unique_persons)
    person_views = map(lambda idx: df.iloc[idx[0]:idx[1]], person_idxs)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        users_journeys = executor.map(get_user_session_journey, person_views)

    return users_journeys


# %%
start = time.time()
users_journeys = get_all_users_session_journeys(df)
mean_journey = get_mean_transition_matrix(users_journeys)

end = time.time() - start

# %%
end/60

# %%
users_journeys

# %%
df_journey = pd.DataFrame(mean_journey,
                          columns=cat_dict['event'].values(),
                          index=cat_dict['event'].values())


# %%
df_journey.head(30)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(df_journey, annot=True,
            linewidth=0.1, linecolor="white",
            ax=ax)
plt.show(fig)

# %%
fig.savefig("markov_50k.png")

# %%
a = np.array([[0.5, 0.5], [0.2, 0.8]])
b = np.array([[0.7, 0.3], [0.7, 0.3]])
c = np.array([a, b])
c = [*map(lambda i: c[:, i], range(0, len(a)))]
d = list(map(lambda x: np.sum(x, axis=0)/np.sum(x), c))
print(c)
print(d)

# %%
c = np.array([a, b])
print(c[:, 0])

# %%
a = np.array([[0.5, 0.5], [0.7, 0.3], [0.0, 0.0]])
b = np.sum(a)
c = np.sum(a, axis=0)
d = c/b
print(b)
print(c)
print(d)
# %%
from random import random
from functools import reduce



# %%
a = map(lambda x: [x, 1.0 - x], [random() for _ in range(2)])
a = map(lambda x: x, [[0.5, 0.5], [0.7, 0.3]])
b = map(lambda x: [0.3, 0.7], range(2))
a2 = map(lambda x: x, [[0.1, 0.9], [0.9, 0.1]])
b2 = map(lambda x: [0.2, 0.8], range(2))
c = map(lambda x: x, [a, b, a2, b2])
d = zip(*c)
e = map(markov_mean, d)

# %%
# [list(x) for x in d]
[*e]

# %%
[*zip([1,2,3],[4,5,6],[7,8,9])]
[*concatenate([1,2,3],[4,5,6])]




# %%
from itertools import islice

# %%
a = map(lambda x: [x, 1.0 - x], [random() for _ in range(2)])
b = map(lambda x: [0.3, 0.7], range(2))
c = map(lambda x: x, [a, b])
print(c)

# %%
n = 2
reshaped_matrices = map(lambda i: islice(c, i),
                        range(0, n))

# %%
mean_matrix = [*map(lambda row:
                    np.sum(row),
                  # np.sum(row, axis=0)/np.sum(row),
                  # if np.sum(row) > 0.0
                  # else row[0],
                  reshaped_matrices)]

# %%
print([list(list(x)) for x in mean_matrix])

# %%
c = [*map(lambda i: c[:, i], range(0, len(a)))]
d = list(map(lambda x: np.sum(x, axis=0)/np.sum(x), c))
print(c)
print(d)
# %%
    reshaped_matrices = np.array([*map(lambda i: m_array[:, i],
                                  range(0, n))])

    mean_matrix = np.array([*map(lambda row:
                                 np.sum(row, axis=0)/np.sum(row)
                                 if np.sum(row) > 0.0
                                 else row[0],
                           reshaped_matrices)])
