import gin
import tensorflow as tf

from tf_agents.agents import ReinforceAgent, PPOAgent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork


@gin.configurable
def create_agent(name,
                 obs_spec,
                 act_spec,
                 ts_spec,
                 step_counter,

                 # gin parameters
                 fc_layer_params,
                 value_fc_layer_params,
                 lstm_size,
                 learning_rate,
                 gamma,
                 entropy_coefficient,
                 decay_steps,
                 use_learning_schedule,
                 value_estimation_loss_coef,
                 normalize_returns,
                 gradient_clipping,
                 decay_rate):
    if value_fc_layer_params is not None:
        value_net = ValueNetwork(
            obs_spec,
            fc_layer_params=value_fc_layer_params)
    else:
        value_net = None

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
    learning_rate = lr_schedule if use_learning_schedule else learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    actor_net = ActorDistributionNetwork(obs_spec,
                                         act_spec,
                                         fc_layer_params=fc_layer_params)

    if name == "reinforce":
        agent = ReinforceAgent(ts_spec,
                               act_spec,
                               actor_network=actor_net,
                               value_network=value_net,
                               optimizer=optimizer,
                               normalize_returns=normalize_returns,
                               train_step_counter=step_counter,
                               gamma=gamma,
                               entropy_regularization=entropy_coefficient,
                               value_estimation_loss_coef=value_estimation_loss_coef,
                               gradient_clipping=gradient_clipping)
    elif name == "rnn_reinforce":
        actor_net = ActorDistributionRnnNetwork(obs_spec,
                                                act_spec,
                                                lstm_size=lstm_size,
                                                input_fc_layer_params=fc_layer_params,
                                                output_fc_layer_params=fc_layer_params)
        agent = ReinforceAgent(ts_spec,
                               act_spec,
                               actor_network=actor_net,
                               value_network=value_net,
                               optimizer=optimizer,
                               normalize_returns=normalize_returns,
                               train_step_counter=step_counter,
                               gamma=gamma,
                               entropy_regularization=entropy_coefficient,
                               value_estimation_loss_coef=value_estimation_loss_coef,
                               gradient_clipping=gradient_clipping)
    elif name == "ppo":
        agent = PPOAgent(ts_spec,
                         act_spec,
                         actor_net=actor_net,
                         value_net=value_net,
                         optimizer=optimizer,
                         train_step_counter=step_counter,
                         greedy_eval=True,
                         discount_factor=gamma,
                         entropy_regularization=entropy_coefficient,
                         num_epochs=10)
    elif name == "rnn_ppo":
        actor_net = ActorDistributionRnnNetwork(obs_spec,
                                                act_spec,
                                                lstm_size=lstm_size,
                                                input_fc_layer_params=fc_layer_params,
                                                output_fc_layer_params=fc_layer_params)
        agent = PPOAgent(ts_spec,
                         act_spec,
                         actor_net=actor_net,
                         value_net=value_net,
                         optimizer=optimizer,
                         train_step_counter=step_counter,
                         greedy_eval=True,
                         discount_factor=gamma,
                         entropy_regularization=entropy_coefficient,
                         num_epochs=10)
    else:
        raise NotImplementedError("{} agent is not implemented".format(name))

    agent.initialize()
    return agent
