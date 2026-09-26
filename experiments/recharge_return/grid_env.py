"""Independent multi-task MiniGrid pilot; coordinates index 10x10 interior."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor

from scripts.audit_grid_routes import TASKS, distance
from scripts.sweep_grid_routes import generated_tasks

DELTAS = ((0,-1),(0,1),(-1,0),(1,0))


class GridRechargeEnv(MiniGridEnv):
    def __init__(self, config, condition="BAL", render_mode=None):
        super().__init__(mission_space=MissionSpace(mission_func=lambda:"Complete work and reach the dock"),
                         width=12,height=12,max_steps=config["environment"]["max_steps"],
                         see_through_walls=True,render_mode=render_mode)
        self.config=config
        if condition not in config["reward"]["conditions"]:
            raise ValueError(condition)
        self.condition=condition
        self.action_space=spaces.Discrete(5)
        self.observation_space=spaces.Box(0.,1.,shape=(109,),dtype=np.float32)
        self.tasks={}
        self.done=True

    def _gen_grid(self,width,height):
        self.grid=Grid(width,height)
        self.grid.wall_rect(0,0,width,height)
        for x,y in self.tasks.values():
            self.grid.set(x+1,y+1,Floor("green"))
        self.grid.set(10,10,Floor("yellow"))
        self.agent_pos=(1,1)
        self.agent_dir=0

    def mask(self):
        x,y=self.pos
        return np.array([y>0,y<9,x>0,x<9,self.pos in self.remaining],dtype=bool)

    def observation(self):
        cells=np.zeros(100,dtype=np.float32)
        for x,y in self.remaining:
            cells[y*10+x]=1.
        return np.concatenate(([self.pos[0]/9,self.pos[1]/9,self.battery/60,
                                1-self.steps/self.max_steps],cells,self.mask())).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        options=options or {}
        self.tasks={k:tuple(v) for k,v in options.get("tasks",TASKS).items()}
        points=list(self.tasks.values())
        if len(set(points))!=len(points) or not points or any(
                not(0<=x<10 and 0<=y<10) or (x>=5 and y>=5) or (x,y)==(0,0) for x,y in points):
            raise ValueError("Invalid task layout")
        battery=float(options.get("battery",60))
        if not 0<battery<=60:
            raise ValueError("Invalid battery")
        super().reset(seed=seed)
        self.pos=(0,0)
        self.battery=battery
        self.remaining=set(points)
        self.steps=self.work=0
        self.done=False
        return self.observation(),{}

    def step(self,action):
        action=int(action)
        if self.done:
            raise RuntimeError("Reset ended episode")
        if not 0<=action<5 or not self.mask()[action]:
            raise ValueError("Masked action")
        before=self.pos
        mask=self.mask().tolist()
        cfg=self.config
        cost=cfg["environment"]["costs"]["work" if action==4 else "move_right"]
        unit=0
        if self.battery>=cost:
            if action==4:
                self.remaining.remove(self.pos)
                x,y=self.pos
                self.grid.set(x+1,y+1,None)
                self.work+=1
                unit=1
            else:
                dx,dy=DELTAS[action]
                self.pos=(self.pos[0]+dx,self.pos[1]+dy)
        self.battery=max(0.,self.battery-cost)
        self.steps+=1
        self.step_count=self.steps
        self.agent_pos=(self.pos[0]+1,self.pos[1]+1)
        outcome=("exhausted" if self.battery<=0 else
                 ("returned" if self.work else "returned_without_work") if self.pos==(9,9) else
                 "time_limit" if self.steps>=self.max_steps else None)
        rp=cfg["reward"]
        deficit=max(0.,min(1.,(rp["safe_margin"]-(self.battery-distance(self.pos)))/60))
        common=-rp["time_cost"]
        if outcome=="exhausted": common-=rp["exhaustion_penalty"]
        if outcome=="returned": common+=rp["return_bonus"]
        if outcome=="returned_without_work": common-=rp["empty_return_penalty"]
        weights=rp["conditions"][self.condition]
        reward=weights["production"]*unit-weights["reserve"]*deficit+common
        self.done=outcome is not None
        record=dict(step=self.steps,before=before,position=self.pos,action="WORK" if action==4 else "MOVE",
                    action_id=action,mask_before=mask,battery=self.battery,work=unit,deficit=deficit,
                    common=common,reward=reward,outcome=outcome)
        return self.observation(),reward,self.done and outcome!="time_limit",outcome=="time_limit",record


class GridTrainingEnv(gym.Wrapper):
    def __init__(self,env,seed):
        super().__init__(env)
        self.training_seed=seed
        self.episode=0

    def reset(self,*,seed=None,options=None):
        rng=np.random.default_rng(np.random.SeedSequence([self.training_seed,self.episode,700]))
        kind=str(rng.choice(["aligned","detour"]))
        count=int(rng.choice([4,6]))
        layout_seed=int(rng.integers(100000,2**31))
        self.current=dict(tasks=generated_tasks(kind,count,layout_seed),battery=int(rng.integers(24,61)),
                          kind=kind,count=count,layout_seed=layout_seed)
        self.episode+=1
        self.episode_return=0.
        return self.env.reset(seed=seed,options=self.current)

    def step(self,action):
        obs,reward,term,trunc,info=self.env.step(action)
        self.episode_return+=reward
        if term or trunc:
            info["training_episode"]=dict(episode=self.episode-1,return_=self.episode_return,
                length=self.env.steps,outcome=info["outcome"],work=self.env.work,scenario=self.current)
        return obs,reward,term,trunc,info
