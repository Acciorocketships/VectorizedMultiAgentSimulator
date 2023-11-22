#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Callable, Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 6)
        n_obstacles = kwargs.get("n_obstacles", 5)

        self.target_max_speed = kwargs.get("target_max_speed", 0.1)
        self.target_resample_p = kwargs.get("target_rewample_p", 0.02)

        self.collision_penalty = kwargs.get("collision_penalty", -1.)
        self.scalarisation_min = kwargs.get("scalarisation_min", False)
        self.scalarisation_weights = kwargs.get("scalarisation_weights", [1.] * 5)

        self.plot_grid = True
        self.min_collision_distance = 0.01
        self.agent_radius = 0.05
        self.obstacle_radius = 0.1
        self.min_spawn_distance = 2 * max(self.agent_radius, self.obstacle_radius)
        self.x_lim = 1.25
        self.y_lim = 1.25

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5, x_semidim=self.x_lim, y_semidim=self.y_lim)

        # Add agents
        self.target = Agent(
            name="target",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            max_speed=self.target_max_speed,
            action_script=self.target_agent_script,
        )
        self.target.target_pos = torch.zeros(batch_dim, 2, device=device)
        world.add_agent(self.target)

        goal_entity_filter: Callable[[Entity], bool] = lambda e: not isinstance(e, Agent)
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.4,
                        entity_filter=goal_entity_filter,
                    )
                ],
                render_action=True,
            )
            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.obstacle_radius),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.agent_rewards = {}

        return world

    def target_agent_script(self, agent: Agent, world: World):
        new_target_pos = torch.empty(
            (agent.target_pos.shape[0], self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(-1., 1.)

        target_resampling_mask = (
            torch.empty(
                agent.target_pos.shape[0],
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(0.0, 1.0)
            < self.target_resample_p
        )
        agent.target_pos[target_resampling_mask] = new_target_pos[target_resampling_mask]
        agent.action.u = torch.clamp((agent.target_pos - agent.state.pos) * 1., -1., 1.)

    def reset_world_at(self, env_index: int = None):
        self.target.target_pos = torch.empty(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(-1., 1.) * torch.tensor([[self.x_lim, self.y_lim]])

        ScenarioUtils.spawn_entities_randomly(
            self.obstacles + self.world.policy_agents + self.world.agents,
            self.world,
            env_index,
            self.min_spawn_distance,
            x_bounds=(-self.x_lim, self.x_lim),
            y_bounds=(-self.y_lim, self.y_lim),
        )

    def reward(self, agent: Agent):
        # Initial Calculation
        rel_pos = torch.zeros(self.world.batch_dim, len(self.world.agents) - 2, 2, device=self.world.device)
        rel_vel = torch.zeros(self.world.batch_dim, len(self.world.agents) - 2, 2, device=self.world.device)
        n = 0
        for other_agent in self.world.agents:
            if other_agent != agent and other_agent != self.target:
                rel_pos[:, n, :] = other_agent.state.pos - agent.state.pos
                rel_vel[:, n, :] = other_agent.state.vel - agent.state.vel
                n += 1
        rel_pos_mag = rel_pos.norm(dim=-1)
        rel_vel_mag = rel_vel.norm(dim=-1)
        target_pos = self.target.state.pos - agent.state.pos
        target_pos_mag = target_pos.norm(dim=-1)

        separation = -1 / rel_pos_mag / n
        separation_reward = separation.sum(dim=-1)

        cohesion = -rel_pos_mag / n
        cohesion_reward = cohesion.sum(dim=-1)

        alignment = -rel_vel_mag / n
        alignment_reward = alignment.sum(dim=-1)

        target_reward = -target_pos_mag

        # Collisions
        obstacle_reward = torch.zeros(self.world.batch_dim, device=self.world.device)
        for other in self.world.agents:
            if other == agent:
                continue
            collision = (self.world.get_distance(agent, other) <= self.min_collision_distance)
            if agent.action_script is None:
                obstacle_reward[collision] += self.collision_penalty
        for other in self.obstacles:
            collision = (self.world.get_distance(agent, other) <= self.min_collision_distance)
            if agent.action_script is None:
                obstacle_reward[collision] += self.collision_penalty

        agent_rew = torch.stack([
                                target_reward,
                                cohesion_reward,
                                separation_reward,
                                alignment_reward,
                                obstacle_reward
                              ], dim=-1)

        self.agent_rewards[agent] = agent_rew

        return self.scalarisation(agent_rew)

    def scalarisation(self, agent_rew):
        if not isinstance(self.scalarisation_weights, torch.Tensor):
            self.scalarisation_weights = torch.Tensor(self.scalarisation_weights)
        weighted_obj = agent_rew * self.scalarisation_weights[None,:] / torch.norm(self.scalarisation_weights)
        if self.scalarisation_min:
            return torch.min(weighted_obj, dim=-1).values
        else:
            return torch.sum(weighted_obj, dim=-1)

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos - self.target.state.pos,
                agent.state.vel,
                agent.sensors[0]._max_range - agent.sensors[0].measure(),
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        if agent not in self.agent_rewards:
            self.reward(agent)
        return {
            "reward": self.agent_rewards[agent]
        }


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.centralised = True

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:

        # Weights
        c_s = 1. # separation
        c_c = 1. # cohesion
        c_a = 1. # alignment
        c_t = 3. # target
        c_o = 1. # obstacle

        # Initial calculations
        n_agents = observation.shape[1]
        pos = observation[..., :2]
        vel = observation[..., 2:4]
        target_pos = torch.zeros_like(pos)
        disps = pos.unsqueeze(1) - pos.unsqueeze(2)
        dists = disps.norm(dim=-1)
        vel_disps = vel.unsqueeze(1) - vel.unsqueeze(2)
        vel_dists = vel_disps.norm(dim=-1)
        normalise_max = lambda x, p: x * p / torch.maximum(x.norm(dim=-1)[..., None], torch.tensor(p))
        normalise = lambda x, p: x * p / x.norm(dim=-1)[..., None]
        scaling = lambda x, x_int, y_int: (x_int - x) / ((x_int * x) + (x_int / y_int))

        # Separation
        separation_weight = 1 / dists[...,None] / n_agents
        separation_weight[torch.isinf(separation_weight)] = 0
        # separation_weight = normalise_max(separation_weight, weight_cutoff)
        separation_dir = -normalise(disps, 1.)
        separation_dir[torch.isnan(separation_dir)] = 0
        separation_action = torch.sum(separation_dir * separation_weight, dim=2)

        # Cohesion
        cohesion_weight = dists[...,None] / n_agents
        # cohesion_weight = normalise_max(cohesion_weight, weight_cutoff)
        cohesion_dir = normalise(disps, 1.)
        cohesion_dir[torch.isnan(cohesion_dir)] = 0
        cohesion_action = torch.sum(cohesion_dir * cohesion_weight, dim=2)

        # Alignment
        alignment_weight = vel_dists[...,None] / n_agents
        # alignment_weight = normalise_max(alignment_weight, weight_cutoff)
        alignment_dir = normalise(vel_disps, 1.)
        alignment_dir[torch.isnan(alignment_dir)] = 0
        alignment_action = torch.sum(alignment_dir * alignment_weight, dim=2)

        # Target
        target_weight = (target_pos - pos).norm(dim=-1)[...,None]
        # target_weight = normalise_max(target_weight, weight_cutoff)
        target_dir = normalise(target_pos-pos, 1.)
        target_dir[torch.isnan(target_dir)] = 0
        target_action = target_dir * target_weight

        # Move away from other agents and obstacles within visibility range
        lidar_range = 0.4
        lidar = lidar_range - observation[..., 4:]
        # object_visible = torch.any(lidar < lidar_range, dim=-1)
        object_dist, object_dir_index = torch.min(lidar, dim=-1)
        object_dir = object_dir_index / lidar.shape[-1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=-1)
        object_scaling = scaling(object_dist, x_int=lidar_range, y_int=1.)
        object_action = -object_vec * object_scaling[..., None]

        action = c_s * separation_action + \
                 c_c * cohesion_action + \
                 c_a * alignment_action + \
                 c_t * target_action

        action = normalise_max(action, 0.8)

        action += c_o * object_action
        action = normalise_max(action, 1.)

        return torch.unbind(action, dim=1)


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
