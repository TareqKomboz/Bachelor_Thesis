import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.utils import common


class TrainingDriver:
    def __init__(self, agent, environment, replay_buffer, replay_observer, number_training_iterations, clear_buffer):
        self.train_losses = common.create_variable('train_losses', shape=(number_training_iterations,), dtype=tf.float32)
        self.train_rewards = common.create_variable('rewards', shape=(number_training_iterations,), dtype=tf.float32)
        self.evaluation_performances = common.create_variable('performance', shape=(number_training_iterations, 2), dtype=tf.float32)
        self.evaluation_rewards = common.create_variable(
            'evaluation_rewards',
            shape=(environment.batch_size, environment.get_episode_length()),
            dtype=tf.float32
        )
        self.driver = DynamicEpisodeDriver(environment, agent.collect_policy, replay_observer)
        self.evaluation_driver = DynamicEpisodeDriver(environment, agent.policy, replay_observer)
        self.evaluation_policy = agent.policy
        self.train_environment = environment
        self.replay_buffer = replay_buffer
        agent.train = common.function(agent.train)
        self.agent = agent
        self.step = common.create_variable("step", 0, dtype=tf.int64)
        self.step_evaluation = common.create_variable("evaluation_step", 0, dtype=tf.int64)  # counts how often quick_eval was called; always <= step

        self.clear_buffer = clear_buffer

    @tf.function
    def train_step(self):
        # collect experience (one episode) and train with experience
        self.driver.run()
        iterator = iter(self.replay_buffer.as_dataset(
            single_deterministic_pass=False,
            sample_batch_size=self.train_environment.batch_size,
            num_steps=self.train_environment.get_episode_length()
        ))
        experience, _ = next(iterator)
        loss = self.agent.train(experience=experience).loss

        # [saving average reward per action and loss from training] for each training iteration
        average_return_over_all_batches = tf.divide(tf.reduce_sum(experience.reward), self.train_environment.batch_size)
        average_reward_over_all_batches_and_all_actions = tf.divide(average_return_over_all_batches, tf.cast(self.train_environment.get_episode_length(), tf.float32))
        self.train_rewards[self.step].assign(average_reward_over_all_batches_and_all_actions)
        self.train_losses[self.step].assign(loss)

        if self.clear_buffer:
            self.replay_buffer.clear()

        self.step.assign_add(1)

    @tf.function
    def quick_evaluation(self):
        time_step = self.train_environment.reset()
        self.evaluation_rewards[:, 0].assign(time_step.reward)
        policy_state = self.evaluation_policy.get_initial_state(self.train_environment.batch_size)

        for i in range(1, tf.cast(self.train_environment.get_episode_length(), tf.int32)):
            action_step = self.evaluation_policy.action(time_step, policy_state=policy_state)
            policy_state = action_step.state
            time_step = self.train_environment.step(action_step.action)
            self.evaluation_rewards[:, i].assign(time_step.reward)

        average_final_reward_over_all_batches = tf.reduce_mean(self.evaluation_rewards[:, -1])
        average_evaluation_reward_over_all_batches_and_all_actions = tf.reduce_mean(self.evaluation_rewards)
        self.evaluation_performances[self.step_evaluation, 0].assign(average_final_reward_over_all_batches)
        self.evaluation_performances[self.step_evaluation, 1].assign(average_evaluation_reward_over_all_batches_and_all_actions)
        self.step_evaluation.assign_add(1)

    def get_summary(self):
        return self.train_rewards[:self.step], self.train_losses[:self.step], self.evaluation_performances[:self.step_evaluation]
