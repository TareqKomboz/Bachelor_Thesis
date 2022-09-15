import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.utils import common


class TrainingDriver:
    def __init__(self, agent, environment, replay_buffer, replay_observer, num_iterations, clear_buffer):
        self.train_losses = common.create_variable('train_losses', shape=(num_iterations,), dtype=tf.float32)
        self.train_returns = common.create_variable('returns', shape=(num_iterations,), dtype=tf.float32)
        self.eval_performance = common.create_variable('performance', shape=(num_iterations, 2), dtype=tf.float32)
        self.eval_rewards = common.create_variable(
            'eval_rewards',
            shape=(environment.batch_size, environment.episode_length),
            dtype=tf.float32
        )
        self.driver = DynamicEpisodeDriver(environment, agent.collect_policy, replay_observer)
        self.eval_driver = DynamicEpisodeDriver(environment, agent.policy, replay_observer)
        self.eval_policy = agent.policy
        self.train_env = environment
        self.replay_buffer = replay_buffer
        agent.train = common.function(agent.train)
        self.agent = agent
        self.step = common.create_variable("step", 0, dtype=tf.int64)
        self.step_eval = common.create_variable("eval_step", 0, dtype=tf.int64)

        self.clear_buffer = clear_buffer

    @tf.function
    def train_step(self):
        self.driver.run()
        iterator = iter(self.replay_buffer.as_dataset(
            single_deterministic_pass=False,
            sample_batch_size=self.train_env.batch_size,
            num_steps=self.train_env.episode_length
        ))
        experience, _ = next(iterator)
        loss = self.agent.train(experience=experience).loss
        avg_return = tf.divide(tf.reduce_sum(experience.reward), self.train_env.batch_size)
        normalized_avg_return = tf.divide(avg_return, tf.cast(self.train_env.episode_length, tf.float32))
        self.train_returns[self.step].assign(normalized_avg_return)
        self.train_losses[self.step].assign(loss)

        if self.clear_buffer:
            self.replay_buffer.clear()

        self.step.assign_add(1)
        return normalized_avg_return, loss

    @tf.function
    def quick_eval(self):
        time_step = self.train_env.reset()
        self.eval_rewards[:, 0].assign(time_step.reward)
        policy_state = self.eval_policy.get_initial_state(self.train_env.batch_size)

        for i in range(1, tf.cast(self.train_env.episode_length, tf.int32)):
            action_step = self.eval_policy.action(time_step, policy_state=policy_state)
            policy_state = action_step.state
            time_step = self.train_env.step(action_step.action)
            self.eval_rewards[:, i].assign(time_step.reward)

        avg_performance = tf.reduce_mean(self.eval_rewards)
        final_performance = tf.reduce_mean(self.eval_rewards[:, -1])
        self.eval_performance[self.step_eval, 0].assign(final_performance)
        self.eval_performance[self.step_eval, 1].assign(avg_performance)
        self.step_eval.assign_add(1)
        return final_performance, avg_performance

    def get_summary(self):
        return self.train_returns[:self.step], self.train_losses[:self.step], self.eval_performance[:self.step_eval]

