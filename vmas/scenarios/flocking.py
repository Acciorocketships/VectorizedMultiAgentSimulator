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
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        desired_distance = kwargs.get("desired_distance", 0.1)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.collision_reward = kwargs.get("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 1)

        self.plot_grid = True
        self.desired_distance = desired_distance
        self.min_collision_distance = 0.005
        self.x_dim = 1
        self.y_dim = 1

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5)
        # Add agents
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            action_script=self.action_script_creator(),
        )

        world.add_agent(self._target)
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[],
                render_action=True,
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.dist_rew = agent.collision_rew.clone()

            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.15),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def action_script_creator(self):
        def action_script(agent, world):
            vel = agent.state.last_action
            noise = torch.randn(vel.shape) / 50

            dist_func = lambda dist: torch.minimum(0.0001 / dist, torch.tensor(1.0))
            repulsor = torch.zeros(vel.shape, device=vel.device)
            left_wall_dist = torch.maximum(agent.state.pos[:,0] + self.x_dim, torch.tensor(0.01))
            right_wall_dist = torch.maximum(self.x_dim - agent.state.pos[:,0], torch.tensor(0.01))
            top_wall_dist = torch.maximum(self.y_dim - agent.state.pos[:,1], torch.tensor(0.01))
            bottom_wall_dist = torch.maximum(agent.state.pos[:, 1] + self.y_dim, torch.tensor(0.01))
            repulsor += dist_func(left_wall_dist)[:,None] * torch.tensor([[1, 0]], device=vel.device).expand(self.world.batch_dim, -1) + \
                        dist_func(right_wall_dist)[:,None] * torch.tensor([[-1, 0]], device=vel.device).expand(self.world.batch_dim, -1) + \
                        dist_func(top_wall_dist)[:,None] * torch.tensor([[0, -1]], device=vel.device).expand(self.world.batch_dim, -1) + \
                        dist_func(bottom_wall_dist)[:,None] * torch.tensor([[0, 1]], device=vel.device).expand(self.world.batch_dim, -1)
            for obstacle in self.obstacles:
                o_dist = torch.norm(obstacle.state.pos - agent.state.pos, dim=-1)
                o_vec = (agent.state.pos - obstacle.state.pos) / o_dist[:,None]
                repulsor += dist_func(o_dist - obstacle.shape.radius - agent.shape.radius)[:,None] * o_vec

            update = noise + repulsor
            new_vel = torch.clamp(vel + update, -0.3, 0.3)
            agent.action.u = new_vel
            agent.state.last_action = new_vel

        return action_script

    def reset_world_at(self, env_index: int = None):
        self._target.state.last_action = torch.zeros(self._target.state.vel.shape)
        target_pos = torch.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        )

        target_pos[:, Y] = -self.y_dim
        self._target.set_pos(target_pos, batch_index=env_index)
        ScenarioUtils.spawn_entities_randomly(
            self.obstacles + self.world.policy_agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
            occupied_positions=target_pos.unsqueeze(1),
        )

        for agent in self.world.policy_agents:
            if env_index is None:
                agent.distance_shaping = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                agent.state.pos - a.state.pos, dim=-1
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        dim=1,
                    )
                    - self.desired_distance
                ).pow(2).mean(-1) * self.dist_shaping_factor

            else:
                agent.distance_shaping[env_index] = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                agent.state.pos[env_index] - a.state.pos[env_index]
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        dim=0,
                    )
                    - self.desired_distance
                ).pow(2).mean(-1) * self.dist_shaping_factor

        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        else:
            self.t[env_index] = 0

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0

        if is_first:
            self.t += 1

            # Avoid collisions with each other
            if self.collision_reward != 0:
                for a in self.world.policy_agents:
                    a.collision_rew[:] = 0

                for i, a in enumerate(self.world.agents):
                    for j, b in enumerate(self.world.agents):
                        if j <= i:
                            continue
                        collision = (
                            self.world.get_distance(a, b) <= self.min_collision_distance
                        )
                        if a.action_script is None:
                            a.collision_rew[collision] += self.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += self.collision_reward

        # stay close together (separation)
        agents_dist_shaping = (
            torch.stack(
                [
                    torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1)
                    for a in self.world.agents
                    if a != agent
                ],
                dim=1,
            )
            - self.desired_distance
        ).pow(2).mean(-1) * self.dist_shaping_factor
        agent.dist_rew = agent.distance_shaping - agents_dist_shaping
        agent.distance_shaping = agents_dist_shaping

        return agent.collision_rew + agent.dist_rew

    def observation(self, agent: Agent):
        rel_agent_pos = torch.cat([
            torch.cat([a.state.pos - agent.state.pos, a.state.vel - agent.state.vel], dim=-1)
            for a in self.world.agents if (a != agent) and (a != self._target)
        ], dim=-1)
        rel_obs_pos = torch.cat([
            torch.cat([a.state.pos - agent.state.pos, a.state.vel - agent.state.vel], dim=-1)
            for a in self.obstacles
        ], dim=-1)
        rel_target_pos = self._target.state.pos - agent.state.pos
        rel_target_vel = self._target.state.vel - agent.state.vel
        return torch.cat([rel_target_pos, rel_target_vel, rel_agent_pos, rel_obs_pos], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "agent_collision_rew": agent.collision_rew,
            "agent_distance_rew": agent.dist_rew,
        }

        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        # First calculate the closest point to a circle of radius circle_radius given the current position
        circle_origin = torch.zeros(1, 2)
        circle_radius = 0.3
        current_pos = observation[:, :2]
        v = current_pos - circle_origin
        closest_point_on_circ = (
            circle_origin + v / torch.linalg.norm(v, dim=1).unsqueeze(1) * circle_radius
        )

        # calculate the normal vector of the vector from the origin of the circle to that closest point
        # on the circle. Adding this scaled normal vector to the other vector gives us a target point we
        # try to reach, thus resulting in a circular motion.
        closest_point_on_circ_normal = torch.stack(
            [closest_point_on_circ[:, Y], -closest_point_on_circ[:, X]], dim=1
        )
        closest_point_on_circ_normal /= torch.linalg.norm(
            closest_point_on_circ_normal, dim=1
        ).unsqueeze(1)
        closest_point_on_circ_normal *= 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents and obstcles within visibility range
        lidar = observation[:, 6:18]
        object_visible = torch.any(lidar < 0.1, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.1
        des_pos[object_visible] = des_pos_object[object_visible]

        action = torch.clamp(
            (des_pos - current_pos) * 10,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
