from pomdp import POMDP
import tiger_world as domain

pomdp = POMDP(
    domain.get_states,
    domain.get_actions,
    domain.get_transition_probabilities,
    domain.get_reward,
    domain.get_observations,
    domain.get_observation_probabilities
)
