import gym
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
import cma
import numpy as np
import time
import os

MAX_INT = (1 << 31) - 1


class SelfAttention(nn.Module):
    """A simple self-attention solution."""

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []

        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        # Expect input_data to be of shape (b, t, k).
        b, t, k = input_data.size()

        # Linear transforms.
        queries = self._fc_q(input=input_data)  # (b, t, q)
        keys = self._fc_k(input=input_data)  # (b, t, q)

        # Attention matrix.
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot


class FCStack(nn.Module):
    """Fully connected layers."""

    def __init__(self, input_dim, num_units, activation, output_dim):
        super(FCStack, self).__init__()
        self._activation = activation
        self._layers = []
        dim_in = input_dim
        for i, n in enumerate(num_units):
            layer = nn.Linear(dim_in, n)
            self._layers.append(layer)
            setattr(self, "_fc{}".format(i + 1), layer)  # Why though?
            dim_in = n
        output_layer = nn.Linear(dim_in, output_dim)
        self._layers.append(output_layer)

    def forward(self, input_data):
        x_input = input_data
        for layer in self._layers[:-1]:
            x_output = layer(x_input)
            if self._activation == "tanh":
                x_input = torch.tanh(x_output)
            elif self._activation == "elu":
                x_input = F.elu(x_output)
            else:
                x_input = F.relu(x_output)
        x_output = self._layers[-1](x_input)
        return x_output


class LSTMStack(nn.Module):
    """LSTM layers."""

    def __init__(self, input_dim, num_units, output_dim):
        super(LSTMStack, self).__init__()
        self._layers = []
        self._hidden_layers = len(num_units) if len(num_units) else 1
        self._hidden_size = num_units[0] if len(num_units) else output_dim
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )
        if len(num_units):
            self._lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self._hidden_size,
                num_layers=self._hidden_layers,
            )
            self._layers.append(self._lstm)
            fc = nn.Linear(
                in_features=self._hidden_size,
                out_features=output_dim,
            )
            self._layers.append(fc)
        else:
            self._lstm = nn.LSTMCell(
                input_size=input_dim,
                hidden_size=self._hidden_size,
            )
            self._layers.append(self._lstm)

    def forward(self, input_data):
        x_input = input_data
        x_output, self._hidden = self._layers[0](x_input.view(1, 1, -1), self._hidden)
        x_output = torch.flatten(x_output, start_dim=0, end_dim=-1)
        if len(self._layers) > 1:
            x_output = self._layers[-1](x_output)
        return x_output

    def reset(self):
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )


class MLPSolution:
    """Model with an underlying MLP or LSTM"""

    def __init__(
        self,
        input_dim,
        num_hiddens,
        activation,
        output_dim,
        output_activation,
        use_lstm,
        l2_coefficient,
    ):
        self._use_lstm = use_lstm
        self._output_dim = output_dim
        self._output_activation = output_activation
        if "roulette" in self._output_activation:
            assert self._output_dim == 1
            self._n_grid = int(self._output_activation.split("_")[-1])
            self._theta_per_grid = 2 * np.pi / self._n_grid
        self._l2_coefficient = abs(l2_coefficient)
        if self._use_lstm:
            self._fc_stack = LSTMStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
            )
        else:
            self._fc_stack = FCStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
                activation=activation,
            )
        self._layers = self._fc_stack._layers

    def output(self, inputs):
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy(inputs).float()
            fc_output = self._fc_stack(inputs)

            if self._output_activation == "tanh":
                output = torch.tanh(fc_output).squeeze().numpy()
            elif self._output_activation == "softmax":
                output = F.softmax(fc_output, dim=-1).squeeze().numpy()
            else:
                output = fc_output.squeeze().numpy()

            return output

    def reset(self):
        if hasattr(self._fc_stack, "reset"):
            self._fc_stack.reset()


class Solution:
    def __init__(
        self,
        image_size,
        query_dim,
        output_dim,
        output_activation,
        num_hiddens,
        l2_coefficient,
        patch_size,
        patch_stride,
        top_k,
        data_dim,
        activation,
        normalize_positions=False,
        use_lstm_controller=False,
    ):
        self._layers = []
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._l2_coefficient = l2_coefficient
        self._normalize_positions = normalize_positions

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self._patch_centers = self._patch_centers()

        self._attention = SelfAttention(
            data_dim=data_dim * self._patch_size ** 2,
            dim_q=query_dim,
        )
        self._layers.extend(self._attention._layers)

        self._model = MLPSolution(
            input_dim=self._top_k * 2,
            num_hiddens=num_hiddens,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            l2_coefficient=l2_coefficient,
            use_lstm=use_lstm_controller,
        )
        self._layers.extend(self._model._layers)

    def _patch_centers(self):
        """Determines centers of patches based on image_size, patch_size and patch_stride"""
        # Images are squares, so patch rows == cols
        n = int((self._image_size - self._patch_size) / self._patch_stride + 1)
        offset = self._patch_size // 2  # Floor division
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * self._patch_stride
            for j in range(n):
                patch_center_col = offset + j * self._patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        return torch.tensor(patch_centers).float()

    def get_output(self, inputs):
        with torch.no_grad():
            # ob.shape = (h, w, c)
            ob = self._transform(inputs).permute(1, 2, 0)
            # print(ob.shape)
            h, w, c = ob.size()
            patches = ob.unfold(0, self._patch_size, self._patch_stride).permute(
                0, 3, 1, 2
            )
            patches = patches.unfold(2, self._patch_size, self._patch_stride).permute(
                0, 2, 1, 4, 3
            )
            patches = patches.reshape((-1, self._patch_size, self._patch_size, c))

            # flattened_patches.shape = (1, n, p * p * c)
            flattened_patches = patches.reshape((1, -1, c * self._patch_size ** 2))
            # attention_matrix.shape = (1, n, n)
            attention_matrix = self._attention(flattened_patches)
            # patch_importance_matrix.shape = (n, n)
            patch_importance_matrix = torch.softmax(attention_matrix.squeeze(), dim=-1)
            # patch_importance.shape = (n,)
            patch_importance = patch_importance_matrix.sum(dim=0)
            # extract top k important patches
            ix = torch.argsort(patch_importance, descending=True)
            top_k_ix = ix[: self._top_k]

            centers = self._patch_centers[top_k_ix]

            centers = centers.flatten(0, -1)
            if self._normalize_positions:
                centers = centers / self._image_size

            return self._model.output(centers)

    def reset(self):
        self._model.reset()

    def get_params(self):
        params = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        offset = 0
        for i, layer in enumerate(self._layers):
            weights_to_set = {}
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                weight = weight_dict[k].numpy()
                weight_size = weight.size
                weights_to_set[k] = torch.from_numpy(
                    params[offset:(offset + weight_size)].reshape(weight.shape))
                offset += weight_size
            self._layers[i].load_state_dict(state_dict=weights_to_set)

    def get_params_from_layer(self, layer_index):
        params = []
        layer = self._layers[layer_index]
        weight_dict = layer.state_dict()
        for k in sorted(weight_dict.keys()):
            params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params_to_layer(self, params, layer_index):
        weights_to_set = {}
        weight_dict = self._layers[layer_index].state_dict()
        offset = 0
        for k in sorted(weight_dict.keys()):
            weight = weight_dict[k].numpy()
            weight_size = weight.size
            weights_to_set[k] = torch.from_numpy(
                params[offset:(offset + weight_size)].reshape(weight.shape))
            offset += weight_size
        self._layers[layer_index].load_state_dict(state_dict=weights_to_set)

    def get_num_params_per_layer(self):
        num_params_per_layer = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            num_params = 0
            for k in sorted(weight_dict.keys()):
                weights = weight_dict[k].numpy()
                num_params += weights.size
            num_params_per_layer.append(num_params)
        return num_params_per_layer

    def _save_to_file(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def save(self, log_dir, iter_count, best_so_far):
        filename = os.path.join(log_dir, 'model_{}.npz'.format(iter_count))
        self._save_to_file(filename=filename)
        if best_so_far:
            filename = os.path.join(log_dir, 'best_model.npz')
            self._save_to_file(filename=filename)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_l2_penalty(self):
        if not hasattr(self, '_l2_coefficient'):
            raise ValueError('l2_coefficient not specified.')
        params = self.get_params()
        return self._l2_coefficient * np.sum(params ** 2)


class GymTask:
    """OpenAI gym tasks."""

    def __init__(self):
        self._env = None
        self._render = False
        self._logger = None

    def create_task(self, **kwargs):
        raise NotImplementedError()

    def seed(self, seed):
        self._env.seed(int(seed))

    def reset(self):
        return self._env.reset()

    def step(self, action, evaluate):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def _process_reward(self, reward, done, evaluate):
        return reward

    def _process_action(self, action):
        return action

    def _process_observation(self, observation):
        return observation

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        return done

    def _show_gui(self):
        if hasattr(self._env, "render"):
            self._env.render()

    def roll_out(self, solution, evaluate):
        ob = self.reset()
        ob = self._process_observation(ob)
        if hasattr(solution, "reset"):
            solution.reset()

        start_time = time.time()

        rewards = []
        done = False
        step_cnt = 0
        while not done:
            action = solution.get_output(inputs=ob)
            action = self._process_action(action)
            ob, r, done, _ = self.step(action, evaluate)
            ob = self._process_observation(ob)

            if self._render:
                self._show_gui()

            step_cnt += 1
            done = self._overwrite_terminate_flag(r, done, step_cnt, evaluate)
            step_reward = self._process_reward(r, done, evaluate)
            rewards.append(step_reward)

        time_cost = time.time() - start_time
        actual_reward = np.sum(rewards)
        if hasattr(self, "_logger") and self._logger is not None:
            self._logger.info(
                "Roll-out time={0:.2f}s, steps={1}, reward={2:.2f}".format(
                    time_cost, step_cnt, actual_reward
                )
            )

        return actual_reward


class CarRacingTask(GymTask):
    """Gym CarRacing-v0 task."""

    def __init__(self):
        super(CarRacingTask, self).__init__()
        self._max_steps = 0
        self._neg_reward_cnt = 0
        self._neg_reward_cap = 0
        self._action_high = np.array([1.0, 1.0, 1.0])
        self._action_low = np.array([-1.0, 0.0, 0.0])

    def _process_action(self, action):
        return (
            action * (self._action_high - self._action_low) / 2.0
            + (self._action_high + self._action_low) / 2.0
        )

    def reset(self):
        ob = super(CarRacingTask, self).reset()
        self._neg_reward_cnt = 0
        return ob

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        if evaluate:
            return done
        if reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0
        too_many_out_of_tracks = 0 < self._neg_reward_cap < self._neg_reward_cnt
        too_many_steps = 0 < self._max_steps <= step_cnt
        return done or too_many_out_of_tracks or too_many_steps

    def create_task(self, **kwargs):
        self._env = gym.make("CarRacing-v0")
        if "render" in kwargs:
            self._render = kwargs["render"]
        if "out_of_track_cap" in kwargs:
            self._neg_reward_cap = kwargs["out_of_track_cap"]
        if "max_steps" in kwargs:
            self._max_steps = kwargs["max_steps"]
        if "logger" in kwargs:
            self._logger = kwargs["logger"]
        return self

    def set_video_dir(self, video_dir):
        from gym.wrappers import Monitor

        self._env = Monitor(
            env=self._env, directory=video_dir, video_callable=lambda x: True
        )


def evaluate(request, solution, task):
    params = np.asarray(request["parameters"])
    solution.set_params(params)
    task.seed(request["env_seed"])
    score = task.roll_out(solution, request["evaluate"])
    penalty = 0 if request["evaluate"] else solution.get_l2_penalty()
    return score - penalty


class CMA:
    """CMA algorithm."""

    def __init__(self, seed, population_size, init_sigma, init_params):
        """Create a wrapper of cmapy."""

        self._algorithm = cma.CMAEvolutionStrategy(
            x0=init_params,
            sigma0=init_sigma,
            inopts={
                "popsize": population_size,
                "seed": seed if seed > 0 else 42,  # ignored if seed is 0
                "randn": np.random.randn,
            },
        )
        self._population = None

    def get_population(self):
        self._population = self._algorithm.ask()
        return self._population

    def evolve(self, fitness):
        self._algorithm.tell(self._population, -fitness)

    def get_current_parameters(self):
        return self._algorithm.result.xfavorite


class ESMaster:
    """Base ES master."""

    def __init__(
        self,
        task,
        solution,
        population_size,
        init_sigma,
        seed,
        n_repeat,
        max_iter,
        eval_every_n_iter,
        n_eval_roll_outs,
    ):
        """Initialization."""

        self._n_repeat = n_repeat
        self._max_iter = max_iter
        self._eval_every_n_iter = eval_every_n_iter
        self._n_eval_roll_outs = n_eval_roll_outs
        self._task = task
        self._solution = solution
        self._rnd = np.random.RandomState(seed=seed)
        self._algorithm = CMA(
            population_size=population_size,
            init_sigma=init_sigma,
            seed=seed,
            init_params=self._solution.get_params(),
        )

    def train(self):
        """Train for max_iter iterations."""

        # Evaluate before train.
        eval_scores = self._evaluate()

        print(
            (
                "TEST Iter 0: size(scores)={0}, "
                "max(scores)={1:.2f}, "
                "mean(scores)={2:.2f}, "
                "min(scores)={3:.2f}, "
                "sd(scores)={4:.2f}".format(
                    eval_scores.size,
                    np.max(eval_scores),
                    np.mean(eval_scores),
                    np.min(eval_scores),
                    np.std(eval_scores),
                )
            )
        )

        best_eval_score = -float("Inf")

        print("Start training for {} iterations.".format(self._max_iter))

        for iter_cnt in range(self._max_iter):
            # Training.
            start_time = time.time()
            scores = self._train_once()
            time_cost = time.time() - start_time
            print("1-step training time: {}s".format(time_cost))

            print(
                "Iter {0}: size(scores)={1}, "
                "max(scores)={2:.2f}, "
                "mean(scores)={3:.2f}, "
                "min(scores)={4:.2f}, "
                "sd(scores)={5:.2f}".format(
                    iter_cnt + 1,
                    scores.size,
                    np.max(scores),
                    np.mean(scores),
                    np.min(scores),
                    np.std(scores),
                )
            )

            # Evaluate periodically.
            if (iter_cnt + 1) % self._eval_every_n_iter == 0:
                # Evaluate.
                start_time = time.time()
                eval_scores = self._evaluate()
                time_cost = time.time() - start_time
                print("Evaluation time: {}s".format(time_cost))

                # Record results and save the model.
                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False

                print(
                    "TEST Iter {0}: size(scores)={1}, "
                    "max(scores)={2:.2f}, "
                    "mean(scores)={3:.2f}, "
                    "min(scores)={4:.2f}, "
                    "sd(scores)={5:.2f}".format(
                        iter_cnt + 1,
                        scores.size,
                        np.max(scores),
                        np.mean(scores),
                        np.min(scores),
                        np.std(scores),
                    )
                )

                self._save_solution(iter_count=iter_cnt + 1, best_so_far=best_so_far)

    def _evaluate(self):
        if self._algorithm is None:
            raise NotImplementedError()

        requests = self._create_requests(evaluate=True)

        fitness = []

        for r in requests:
            f = evaluate(r, self._solution, self._task)
            fitness.append(f)

        # fitness = np.array([evaluate(r, self._solution, self._task) for r in requests])
        return np.array(fitness)

    def _train_once(self):
        if self._algorithm is None:
            raise NotImplementedError()

        requests = self._create_requests(evaluate=False)

        fitness = np.array([evaluate(r, self._solution, self._task) for r in requests])
        fitness = fitness.reshape([-1, self._n_repeat]).mean(axis=1)
        self._algorithm.evolve(fitness)

        return fitness

    def _save_solution(self, iter_count, best_so_far):
        if self._algorithm is None:
            raise NotImplementedError()
        self._update_solution()
        self._solution.save(self._log_dir, iter_count, best_so_far)

    def _create_requests(self, evaluate):
        """Create requests."""

        if evaluate:
            n_repeat = 1
            num_roll_outs = self._n_eval_roll_outs
            params_list = [self._algorithm.get_current_parameters()]
        else:
            n_repeat = self._n_repeat
            params_list = self._algorithm.get_population()
            num_roll_outs = len(params_list) * n_repeat

        env_seed_list = self._rnd.randint(low=0, high=MAX_INT, size=num_roll_outs)

        requests = []
        for i, env_seed in enumerate(env_seed_list):
            ix = 0 if evaluate else i // n_repeat
            requests.append(
                dict(
                    roll_out_index=i,
                    env_seed=env_seed,
                    parameters=params_list[ix],
                    evaluate=evaluate,
                )
            )
        return requests

    def _update_solution(self):
        if self._algorithm is None:
            raise NotImplementedError()
        self._solution.set_params(self._algorithm.get_current_parameters())


if __name__ == "__main__":
    task = CarRacingTask()
    task.create_task(**dict(out_of_track_cap=20, max_steps=1000, render=False))
    solution = Solution(
        image_size=96,
        query_dim=4,
        output_dim=3,
        output_activation="tanh",
        num_hiddens=[
            16,
        ],
        l2_coefficient=0,
        patch_size=7,
        patch_stride=4,
        top_k=10,
        data_dim=3,
        activation="tanh",
        normalize_positions=True,
        use_lstm_controller=True,
    )
    master = ESMaster(
        task=task,
        solution=solution,
        population_size=256,
        init_sigma=0.1,
        seed=0,
        n_repeat=16,
        max_iter=2000,
        eval_every_n_iter=10,
        n_eval_roll_outs=2,
    )

    print("Start to train.")
    master.train()
    print("Done")
