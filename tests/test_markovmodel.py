import numpy as np
import pandas as pd
from mm.markovmodel import (is_in,
                            simulate_chain,
                            simulate_chain_list,
                            generate_transition_matrix,
                            get_user_session_journey,
                            get_all_users_session_journeys,
                            pseudo_r2,
                            calculate_likelihood,
                            MarkovModel)


def test_simulte_chain():
    tm_seq = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    chain_seq = simulate_chain([0], list(tm_seq), 50)
    has_2_seq = is_in(2, chain_seq)

    assert type(has_2_seq) is bool
    assert len(chain_seq) == 3

    tm_rand = np.array([[0.5, 0.45, 0.05], [0.2, 0.79, 0.01], [0.0, 0.0, 1.0]])
    chain_rand = simulate_chain([0], list(tm_rand), 50)
    has_2_rand = is_in(2, chain_rand)

    assert type(has_2_rand) is bool
    assert len(chain_rand) <= 50


def test_simulate_chain_list():
    tm = np.array([[0.5, 0.45, 0.05], [0.2, 0.79, 0.01], [0.0, 0.0, 1.0]])
    chains = simulate_chain_list(0, tm, 100, 1_000)
    mean_chain_length = np.mean(list(map(len, list(chains))))

    assert mean_chain_length < 100


def test__generate_transition_matrix__3_events_list_5_total():
    # Possible events: 0,1,2,3,4
    events_list = [1, 1, 2]

    transition = generate_transition_matrix(events_list,
                                            n_states=5,
                                            end_state=4)
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


def test__generate_transition_matrix__user_4_sessions_2_events():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2],
            # "event":  [5, 1, 2, 6,
            #            5, 1, 2, 6,
            #            5, 2, 2, 6,
            #            5, 2, 2, 6],
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
    n_states = 5+2  # 2 artificial states: start_session and end_session
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
                         [0.0, 1/2, 1/2, 0.0, 0.0, 0.0, 0.0],  # start_event
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # end_event

    np.testing.assert_array_almost_equal(expected, list(transition_matrix))

    for row in list(transition_matrix):
        assert np.sum(row) == 1.0


def test___get_all_users_session_journeys__1_user_4_ses_2_even():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2],
            # "event":  [5, 1, 2, 6,
            #            5, 1, 2, 6,
            #            5, 2, 2, 6,
            #            5, 2, 2, 6],
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

    n_states = 5+2  # 2 artificial states: start_session and end_session
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


def test___get_all_users_session_journeys__2_user_4_ses_2_even():
    user = {"person":   [1, 1, 1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2, 2, 2],
            "session":  [0, 0, 1, 1, 2, 2, 3, 3,
                         0, 0, 1, 1, 2, 2, 3, 3],
            "event":    [1, 2, 1, 2, 2, 2, 2, 2,
                         1, 2, 1, 2, 2, 2, 2, 2],
            # "event":  [5, 1, 2, 6,
            #            5, 1, 2, 6,
            #            5, 2, 2, 6,
            #            5, 2, 2, 6],
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

    n_states = 5+2  # 2 artificial states: start_session and end_session
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


def test_markovmdel():
    tm = np.array([[0.5, 0.3, 0.05, 0.15],
                   [0.2, 0.59, 0.01, 0.2],
                   [0.01, 0.01, 0.0, 0.98],
                   [0.0, 0.0, 0.0, 1.0]])
    start_state = 0
    max_chain_length = 1_000
    n_training_chains = 10_000
    n_states = 4
    target_state = 2

    markov_model = MarkovModel(transition_matrix=tm)

    markov_model.fit(start_state=start_state,
                     target_state=target_state,
                     max_chain_length=max_chain_length,
                     n_training_chains=n_training_chains)

    np.testing.assert_almost_equal(tm, markov_model.transition_matrix)

    assert markov_model.n_states == n_states

    assert np.sum(markov_model.prediction_matrix) > 1.0


def test_pseudo_r2():
    tm_sequential = [[0.1, 0.8, 0.1],
                     [0.1, 0.1, 0.8],
                     [0.8, 0.1, 0.1]]

    tm_random = [[0.33, 0.33, 0.33],
                 [0.33, 0.33, 0.33],
                 [0.33, 0.33, 0.33]]

    sequential_chain = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    random_chain = [0, 0, 2, 1, 1, 2, 0, 0, 2]

    r2_seq = pseudo_r2(tm_sequential, tm_random, sequential_chain)
    r2_rand = pseudo_r2(tm_sequential, tm_random, random_chain)

    assert r2_seq > 0.0
    assert r2_rand < 0.0


def test_calculate_likelihood():
    tm_sequential = [[0.1, 0.8, 0.1],
                     [0.1, 0.1, 0.8],
                     [0.8, 0.1, 0.1]]

    sequential_chain = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    random_chain = [0, 0, 2, 1, 1, 2, 0, 0, 2]

    l_seq = calculate_likelihood(tm_sequential, sequential_chain)
    l_rand = calculate_likelihood(tm_sequential, random_chain)

    assert l_seq > l_rand
