import os, subprocess, time, signal, math, copy
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

class HyrjaEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.timespan = 0.02
        self.buff = 4.2
        self.boss_speed = 12
        self.boss_hp = 82234248
        self.boss_dps = 500000
        self.player_hp = 4500000
        self.tank_hp = 7000000
        self.dd_dps = 1000000
        self.dd_move_discount = 0.4
        self.melee_attack_range = 8
        self.ranged_attack_range = 40
        self.healer_hps = 1000000
        self.healer_move_discount = 0.4
        self.tank_spell_coeff = 0.7
        self.tank_melee_coeff = 0.2
        self.tank_dps = 500000
        self.tank_threat_coeff = 5.0
        self.player_speed = 8
        self.sanctify_damage = 170625
        self.sanctify_splash_damage = 121875
        self.sanctify_orb_speed = 5
        self.sanctify_orb_range = 5
        self.eye_of_the_storm_damage = 132600
        self.eye_of_the_storm_reduction = 144000
        self.eye_of_the_storm_shelter_range = 10
        self.shield_of_light_damage = 774881
        self.shield_of_light_knock_distance = 60
        self.shield_of_light_range = 5
        self.expel_light_damage = 210600
        self.expel_light_range = 8
        self.arcing_bolt_damage = 195000
        self.arcing_bolt_range = 5
        self.observation_space = spaces.Tuple((spaces.Box(low=-50, high=50, shape=12), # Position
                                               spaces.Box(low=-100, high=100, shape=70), # Sanctify Orb Position
                                               spaces.Box(low=-1, high=1, shape=70), # Sanctify Orb Direction
                                               spaces.Box(low=-50, high=50, shape=2), # Shelter Position
                                               spaces.Box(low=0, high=34, shape=1), # CD: Shield of Light 
                                               spaces.Box(low=0, high=30, shape=1), # CD: Sanctify / Eye of the Storm
                                               spaces.Box(low=0, high=30, shape=1), # CD: Expel Light / Arcing Bolt
                                               spaces.Box(low=0, high=2, shape=1), # CH: Shield of Light 
                                               spaces.Box(low=-1, high=1, shape=2), # Shield of Light Direction
                                               spaces.Box(low=0, high=9.01, shape=2), # CH: Sanctify / Eye of the Storm
                                               spaces.Box(low=0, high=4.5, shape=2), # CH: Expel Light / Arcing Bolt
                                               spaces.Box(low=0, high=3, shape=5),  # EX: Expel Light
                                               spaces.Box(low=0, high=1.3, shape=5),  # EX: Arcing Bolt Target
                                               spaces.Box(low=0, high=100, shape=2), # Mystic Empowerment
                                               spaces.Box(low=0, high=1, shape=5), # Damage Statistics
                                               spaces.Box(low=0, high=1, shape=6))) # HP
        self.action_space = spaces.Box(low=-1, high=1, shape=15) # (is_move, direction_x, direction_y)x5
        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def point_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def move_towards_direction(self, x, y, dirx, diry, speed, restricted):
        normalizer = math.sqrt(dirx * dirx + diry * diry)
        norm_x = dirx * speed * self.timespan / normalizer
        norm_y = diry * speed * self.timespan / normalizer
        x = x + norm_x
        y = y + norm_y
        if restricted:
          if x > 50:
            x = 50
          elif x < -50:
            x = -50
          if y > 50:
            y = 50
          elif y < -50:
            y = -50
        else:
          if x > 100:
            x = 100
          elif x < -100:
            x = -100
          if y > 100:
            y = 100
          elif y < -100:
            y = -100
        return x, y

    def move_towards_point(self, x, y, tarx, tary, speed, restricted):
        return self.move_towards_direction(x, y, tarx - x, tary - y, speed, restricted)

    def move_backwards_point(self, x, y, tarx, tary, speed, restricted):
        return self.move_towards_direction(x, y, x - tarx, y - tary, speed, restricted)

    def shield_hit(self, x, y, boss_x, boss_y, dir_x, dir_y):
        same_direction = (x - boss_x) * dir_x + (y - boss_y) * dir_y
        l_x, l_y = self.move_towards_direction(boss_x, boss_y, -dir_y, dir_x, self.shield_of_light_range / self.timespan, False)
        r_x, r_y = self.move_towards_direction(boss_x, boss_y, dir_y, -dir_x, self.shield_of_light_range / self.timespan, False)
        not_out_of_lb = (x - l_x) * dir_y + (y - l_y) * -dir_x
        not_out_of_rb = (x - r_x) * -dir_y + (y - r_y) * dir_x
        return bool(same_direction > 0 and not_out_of_lb > 0 and not_out_of_rb > 0)

    def elapse_across(self, now, boundary):
        return bool(now - self.timespan <= boundary and now > boundary)

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        pos, orb_pos, orb_dir, eye_pos, knock_cd, ult_cd, minor_cd, knock_ch, knock_dir, ult_ch, minor_ch, expel_ex, arc_ex, empowerment, stat, hp = state
        holy_stack, thunder_stack = empowerment
        pos = list(pos)
        orb_pos = list(orb_pos)
        orb_dir = list(orb_dir)
        stat = list(stat)
        hp = list(hp)
        expel_ex = list(expel_ex)
        arc_ex = list(arc_ex)
        reward = 0

        sanctify_ch, eye_ch = ult_ch
        expel_ch, arc_ch = minor_ch
        is_casting = bool(knock_ch[0] > 0 or sanctify_ch > 0 or eye_ch > 0 or arc_ch > 0)

        # boss spell effects
        spell_coeff = (self.tank_spell_coeff * self.player_hp / self.tank_hp, 1, 1, 1, 1)
        # expel debuff
        for x in range(0,5):
          if self.elapse_across(expel_ex[x], 0):
            dmg = self.expel_light_damage * self.buff * (1 + 0.1 * holy_stack)
            for y in range(0,5):
              if self.point_distance(pos[2*x], pos[2*x+1], pos[2*y], pos[2*y+1]) <= self.expel_light_range:
                this_dmg = dmg * spell_coeff[y] / self.player_hp
                this_dmg = min(this_dmg, hp[y])
                hp[y] = hp[y] - this_dmg
            if expel_ch > 0:
              dist = self.np_random.uniform(low=0, high=1, size=(5,))
              dist[x] = -1
              for y in range(0,5):
                if hp[y] <= 0:
                  dist[y] = -2
              bounce_target = np.argmax(dist)
              expel_ex[bounce_target] = 3
        # arcing bolt channeling
        if self.elapse_across(arc_ch, 0):
          arc_target = np.argmax(arc_ex)
          if hp[arc_target] > 0:
            dmg = self.arcing_bolt_damage * self.buff * (1 + 0.1 * thunder_stack)
            for x in range(0,5):
              if self.point_distance(pos[2*x], pos[2*x+1], pos[2*arc_target], pos[2*arc_target+1]) <= self.arcing_bolt_range:
                this_dmg = dmg * spell_coeff[x] / self.player_hp
                this_dmg = min(this_dmg, hp[x])
                hp[x] = hp[x] - this_dmg
        # shield of light channeling
        if self.elapse_across(knock_ch[0], 0):
          dmg = self.shield_of_light_damage * self.buff
          for x in range(0,5):
            if hp[x] > 0 and self.shield_hit(pos[2*x], pos[2*x+1], pos[10], pos[11], knock_dir[0], knock_dir[1]):
              this_dmg = dmg * spell_coeff[x] / self.player_hp
              this_dmg = min(this_dmg, hp[x])
              hp[x] = hp[x] - this_dmg
              pos[2*x], pos[2*x+1] = self.move_towards_direction(pos[2*x], pos[2*x+1], knock_dir[0], knock_dir[1], self.shield_of_light_knock_distance / self.timespan, True)
        # eye of the storm channeling
        eye_tick = False
        for x in range(0,7):
          if self.elapse_across(eye_ch, x * 1.5):
            eye_tick = True
        if eye_tick:
          for x in range(0,5):
            if hp[x] > 0:
              dmg = self.eye_of_the_storm_damage * (1 + 0.1 * thunder_stack)
              if self.point_distance(pos[2*x], pos[2*x+1], eye_pos[0], eye_pos[1]) <= self.eye_of_the_storm_shelter_range:
                dmg = dmg - self.eye_of_the_storm_reduction
              dmg = dmg * self.buff
              this_dmg = dmg * spell_coeff[x] / self.player_hp
              this_dmg = min(this_dmg, hp[x])
              hp[x] = hp[x] - this_dmg
        # sanctify orb touching
        for x in range(0,5):
          if hp[x] > 0:
            for y in range(0, 35):
              if self.point_distance(pos[2*x], pos[2*x+1], orb_pos[2*y], orb_pos[2*y+1]) <= self.sanctify_orb_range:
                dmg = self.sanctify_damage * self.buff * (1 + 0.1 * holy_stack)
                this_dmg = dmg * spell_coeff[x] / self.player_hp
                this_dmg = min(this_dmg, hp[x])
                hp[x] = hp[x] - this_dmg
                for z in range(0, 5):
                  if hp[z] > 0 and not z == x:
                    dmg = self.sanctify_splash_damage * self.buff * (1 + 0.1 * holy_stack)
                    this_dmg = dmg * spell_coeff[z] / self.player_hp
                    this_dmg = min(this_dmg, hp[z])
                    hp[z] = hp[z] - this_dmg
                orb_pos[2*y], orb_pos[2*y+1] = self.move_towards_direction(orb_pos[2*y], orb_pos[2*y+1], orb_dir[2*y], orb_dir[2*y+1], 100 / self.timespan, False)
        # sanctify orb radiate
        sanctify_tick = False
        num_tick = 0
        for x in range(0,7):
          if self.elapse_across(sanctify_ch, x * 1.5):
            sanctify_tick = True
            num_tick = x
        if sanctify_tick:
          for x in range(0,5):
            y = num_tick * 5 + x
            orb_pos[2*y] = pos[10]
            orb_pos[2*y+1] = pos[11]
            dir = self.np_random.uniform(low=0, high=2*math.pi, size=(1,))[0]
            orb_dir[2*y] = math.cos(dir)
            orb_dir[2*y+1] = math.sin(dir)

        # time elapse
        knock_ch = (max(knock_ch[0] - self.timespan, 0),)
        sanctify_ch = max(sanctify_ch - self.timespan, 0)
        eye_ch = max(eye_ch - self.timespan, 0)
        arc_ch = max(arc_ch - self.timespan, 0)
        expel_ch = max(expel_ch - self.timespan, 0)
        ult_cd = (max(ult_cd[0] - self.timespan, 0),)
        minor_cd = (max(minor_cd[0] - self.timespan, 0),)
        knock_cd = (max(knock_cd[0] - self.timespan, 0),)
        for x in range(0, 35):
          orb_pos[2*x], orb_pos[2*x+1] = self.move_towards_direction(orb_pos[2*x], orb_pos[2*x+1], orb_dir[2*x], orb_dir[2*x+1], self.sanctify_orb_speed, False)
        for x in range(0, 5):
          expel_ex[x] = max(expel_ex[x] - self.timespan, 0)
          arc_ex[x] = max(arc_ex[x] - self.timespan, 0)

        # stack empowerment
        if pos[11] > 10:
          thunder_stack = max(thunder_stack - 1.0 * self.timespan, 0)
        else:
          thunder_stack = min(thunder_stack + 0.25 * self.timespan, 100)
        if pos[11] < -10:
          holy_stack = max(holy_stack - 1.0 * self.timespan, 0)
        else:
          holy_stack = min(holy_stack + 0.25 * self.timespan, 100)
        empowerment = (holy_stack, thunder_stack)

        # player move
        move = [0,0,0,0,0]
        move_dir = [0,0,0,0,0,0,0,0,0,0]
        for x in range(0,5):
          move[x] = action[3*x]
          move_dir[2*x] = action[3*x+1]
          move_dir[2*x+1] = action[3*x+2]
        for x in range(0,5):
          if hp[x] > 0 and move[x] > 0:
            pos[2*x], pos[2*x+1] = self.move_towards_direction(pos[2*x], pos[2*x+1], move_dir[2*x], move_dir[2*x+1], self.player_speed, True)

        tank_hp, warrior_hp, mage_hp, hunter_hp, priest_hp, boss_hp = tuple(hp)

        # player dealing dmg
        if tank_hp > 0 and self.point_distance(pos[0], pos[1], pos[10], pos[11]) <= self.melee_attack_range:
          dmg = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.tank_dps * self.timespan / (self.boss_hp * self.buff)
          if dmg > boss_hp:
            dmg = boss_hp
          stat[0] = stat[0] + dmg
          boss_hp = boss_hp - dmg
          reward = reward + dmg
        if warrior_hp > 0 and self.point_distance(pos[2], pos[3], pos[10], pos[11]) <= self.melee_attack_range:
          dmg = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.dd_dps * self.timespan / (self.boss_hp * self.buff)
          if dmg > boss_hp:
            dmg = boss_hp
          stat[1] = stat[1] + dmg
          boss_hp = boss_hp - dmg
          reward = reward + dmg
        if mage_hp > 0 and self.point_distance(pos[4], pos[5], pos[10], pos[11]) <= self.ranged_attack_range:
          dmg = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.dd_dps * self.timespan / (self.boss_hp * self.buff)
          if move[2] > 0:
            dmg = dmg * self.dd_move_discount
          if dmg > boss_hp:
            dmg = boss_hp
          stat[2] = stat[2] + dmg
          boss_hp = boss_hp - dmg
          reward = reward + dmg
        if hunter_hp > 0 and self.point_distance(pos[6], pos[7], pos[10], pos[11]) <= self.ranged_attack_range:
          dmg = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.dd_dps * self.timespan / (self.boss_hp * self.buff)
          if dmg > boss_hp:
            dmg = boss_hp
          stat[3] = stat[3] + dmg
          boss_hp = boss_hp - dmg
          reward = reward + dmg
        hp = [tank_hp, warrior_hp, mage_hp, hunter_hp, priest_hp, boss_hp]

        # player dealing heal
        heal_dist = self.np_random.uniform(low=0, high=1, size=(5,))
        for x in range(0,5):
          heal_dist[x] = heal_dist[x] * (1 - hp[x])
          if hp[x] <= 0 or self.point_distance(pos[2*x], pos[2*x+1], pos[8], pos[9]) > self.ranged_attack_range:
            heal_dist[x] = -1
        heal_target = np.argmax(heal_dist)
        if priest_hp > 0:
          heal = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.healer_hps * self.timespan
          if heal_target == 0:
            heal = heal / self.tank_hp
          else:
            heal = heal / self.player_hp
          if move[4] > 0:
            heal = heal * self.healer_move_discount
          hp[heal_target] = hp[heal_target] + heal
          if hp[heal_target] > 1:
            hp[heal_target] = 1

        # boss threat target
        threat = copy.copy(stat)
        threat[0] = threat[0] * self.tank_threat_coeff
        for x in range(0,5):
          if hp[x] <= 0:
            threat[x] = -1
        boss_target = np.argmax(threat)

        # boss melee
        if not is_casting:
          if self.point_distance(pos[2*boss_target], pos[2*boss_target+1], pos[10], pos[11]) > self.melee_attack_range:
            pos[10], pos[11] = self.move_towards_point(pos[10], pos[11], pos[2*boss_target], pos[2*boss_target+1], self.boss_speed, True)
          else:
            dmg = self.np_random.uniform(low=0.5, high=1.5, size=(1,))[0] * self.boss_dps * self.timespan * self.buff
            if boss_target == 0:
              dmg = dmg * self.tank_melee_coeff / self.tank_hp
            else:
              dmg = dmg / self.player_hp
            if dmg > hp[boss_target]:
              dmg = hp[boss_target]
            hp[boss_target] = hp[boss_target] - dmg

        tank_hp, warrior_hp, mage_hp, hunter_hp, priest_hp, boss_hp = tuple(hp)

        # boss spells
        if not is_casting:

          if knock_cd[0] <= 0:
            # cast shield of light
            knock_dir = (pos[2*boss_target] - pos[10], pos[2*boss_target+1] - pos[11])
            knock_ch = (2,)
            knock_cd = self.np_random.uniform(low=28, high=34, size=(1,))
          elif ult_cd[0] <= 0:
            if holy_stack > 1 and holy_stack >= thunder_stack:
              # cast sanctify
              sanctify_ch = 9.01
              for x in range(0,5):
                if hp[x] > 0 and self.point_distance(pos[2*x], pos[2*x+1], pos[10], pos[11]) <= 8:
                  pos[2*x], pos[2*x+1] = self.move_backwards_point(pos[2*x], pos[2*x+1], pos[10], pos[11], 5 / self.timespan, True)
              ult_cd = self.np_random.uniform(low=30, high=30, size=(1,))
            elif thunder_stack > 1 and thunder_stack > holy_stack:
              # cast eye of the storm
              eye_ch = 9.01
              eye_pos = self.np_random.uniform(low=-10, high=10, size=(2,))
              eye_pos[0] = min(max(eye_pos[0] + pos[10], -40), 40)
              eye_pos[1] = min(max(eye_pos[1] + pos[11], -40), 40)
              ult_cd = self.np_random.uniform(low=30, high=30, size=(1,))
          elif minor_cd[0] <= 0:
            if holy_stack > 1:
              # cast expel light
              dist = self.np_random.uniform(low=0, high=1, size=(5,))
              for y in range(0,5):
                if hp[y] <= 0:
                  dist[y] = -2
              expel_target = np.argmax(dist)
              expel_ex[expel_target] = 3
              expel_ch = 4.5
              minor_cd = self.np_random.uniform(low=24, high=30, size=(1,))
            if thunder_stack > 1:
              # cast arcing bolt
              dist = self.np_random.uniform(low=0, high=1, size=(5,))
              for y in range(0,5):
                if hp[y] <= 0:
                  dist[y] = -2
              arc_target = np.argmax(dist)
              arc_ex[arc_target] = 1.3
              arc_ch = 1.3
              minor_cd = self.np_random.uniform(low=24, high=30, size=(1,))

        ult_ch = (sanctify_ch, eye_ch) 
        minor_ch = (expel_ch, arc_ch)
        pos = tuple(pos)
        orb_pos = tuple(orb_pos)
        orb_dir = tuple(orb_dir)
        stat = tuple(stat)
        hp = tuple(hp)
        expel_ex = tuple(expel_ex)
        arc_ex = tuple(arc_ex)
        self.state = (pos, orb_pos, orb_dir, eye_pos, knock_cd, ult_cd, minor_cd, knock_ch, knock_dir, ult_ch, minor_ch, expel_ex, arc_ex, empowerment, stat, hp)
        done = (tank_hp <= 0 and warrior_hp <= 0 and mage_hp <= 0 and hunter_hp <= 0 and priest_hp <= 0) or boss_hp <= 0
        done = bool(done)

        if done:
          if self.steps_beyond_done is None:
              self.steps_beyond_done = 0
          else:
              if self.steps_beyond_done == 0:
                  logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
              self.steps_beyond_done += 1
              reward = 0.0

        return self.state, reward, done, {}

    def _reset(self):
        pos = (-35, -5, -30, -7, -17, -17, -28, -28, -22, -22, -40, 0)
        orb_pos = tuple(self.np_random.uniform(low=100, high=100, size=(70,)))
        orb_dir = tuple(self.np_random.uniform(low=1, high=1, size=(70,)))
        eye_pos = (0, 0)
        knock_cd = tuple(self.np_random.uniform(low=23.5, high=24.5, size=(1,)))
        ult_cd = tuple(self.np_random.uniform(low=8, high=9, size=(1,)))
        minor_cd = tuple(self.np_random.uniform(low=4.3, high=7.2, size=(1,)))
        knock_ch = (0,)
        knock_dir = (1, 1)
        ult_ch = (0, 0)
        minor_ch = (0, 0)
        expel_ex = (0, 0, 0, 0, 0)
        arc_ex = (0, 0, 0, 0, 0)
        empowerment = (0, 0)
        stat = (0, 0, 0, 0, 0)
        hp = (1, 1, 1, 1, 1, 1)
        self.state = (pos, orb_pos, orb_dir, eye_pos, knock_cd, ult_cd, minor_cd, knock_ch, knock_dir, ult_ch, minor_ch, expel_ex, arc_ex, empowerment, stat, hp)
        self.steps_beyond_done = None
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400
        from gym.envs.classic_control import rendering
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-50,50,-50,50)
            self.orb_trans = [rendering.Transform() for i in range(0,35)]
            for i in range(0,35):
              orb = rendering.make_circle(2)
              orb.set_color(1,1,0.2)
              orb.add_attr(self.orb_trans[i])
              self.viewer.add_geom(orb)
            boss = rendering.make_circle(5)
            boss.set_color(1,0,0)
            self.boss_trans = rendering.Transform()
            boss.add_attr(self.boss_trans)
            self.viewer.add_geom(boss)
            tank = rendering.make_circle(3)
            tank.set_color(0.7686,0.1176,0.2314)
            self.tank_trans = rendering.Transform()
            tank.add_attr(self.tank_trans)
            self.viewer.add_geom(tank)
            warrior = rendering.make_circle(3)
            warrior.set_color(0.7765,0.6078,0.4275)
            self.warrior_trans = rendering.Transform()
            warrior.add_attr(self.warrior_trans)
            self.viewer.add_geom(warrior)
            mage = rendering.make_circle(3)
            mage.set_color(0.4078,0.8,0.9373)
            self.mage_trans = rendering.Transform()
            mage.add_attr(self.mage_trans)
            self.viewer.add_geom(mage)
            hunter = rendering.make_circle(3)
            hunter.set_color(0.6667,0.8275,0.4471)
            self.hunter_trans = rendering.Transform()
            hunter.add_attr(self.hunter_trans)
            self.viewer.add_geom(hunter)
            self.priest_trans = rendering.Transform()
            priest_border = rendering.make_circle(3,30,False)
            priest_border.set_color(0,0,0)
            priest_border.add_attr(self.priest_trans)
            priest = rendering.make_circle(2.7)
            priest.set_color(0.9412,0.9216,0.8784)
            priest.add_attr(self.priest_trans)
            self.viewer.add_geom(priest)
            self.viewer.add_geom(priest_border)
            boss_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            boss_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            boss_hp.set_color(1.0,0,0)
            boss_hp_bg.set_color(0,0,0)
            self.boss_hp_trans = rendering.Transform()
            boss_hp.add_attr(self.boss_hp_trans)
            boss_hp_bg.add_attr(self.boss_trans)
            self.viewer.add_geom(boss_hp_bg)
            self.viewer.add_geom(boss_hp)
            tank_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            tank_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            tank_hp.set_color(1.0,0,0)
            tank_hp_bg.set_color(0,0,0)
            self.tank_hp_trans = rendering.Transform()
            tank_hp.add_attr(self.tank_hp_trans)
            tank_hp_bg.add_attr(self.tank_trans)
            self.viewer.add_geom(tank_hp_bg)
            self.viewer.add_geom(tank_hp)
            warrior_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            warrior_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            warrior_hp.set_color(1.0,0,0)
            warrior_hp_bg.set_color(0,0,0)
            self.warrior_hp_trans = rendering.Transform()
            warrior_hp.add_attr(self.warrior_hp_trans)
            warrior_hp_bg.add_attr(self.warrior_trans)
            self.viewer.add_geom(warrior_hp_bg)
            self.viewer.add_geom(warrior_hp)
            mage_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            mage_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            mage_hp.set_color(1.0,0,0)
            mage_hp_bg.set_color(0,0,0)
            self.mage_hp_trans = rendering.Transform()
            mage_hp.add_attr(self.mage_hp_trans)
            mage_hp_bg.add_attr(self.mage_trans)
            self.viewer.add_geom(mage_hp_bg)
            self.viewer.add_geom(mage_hp)
            hunter_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            hunter_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            hunter_hp.set_color(1.0,0,0)
            hunter_hp_bg.set_color(0,0,0)
            self.hunter_hp_trans = rendering.Transform()
            hunter_hp.add_attr(self.hunter_hp_trans)
            hunter_hp_bg.add_attr(self.hunter_trans)
            self.viewer.add_geom(hunter_hp_bg)
            self.viewer.add_geom(hunter_hp)
            priest_hp_bg = rendering.make_polygon(((-5.25,4.25),(5.25,4.25),(5.25,1.75),(-5.25,1.75)))
            priest_hp = rendering.make_polygon(((-5,1),(5,1),(5,-1),(-5,-1)))
            priest_hp.set_color(1.0,0,0)
            priest_hp_bg.set_color(0,0,0)
            self.priest_hp_trans = rendering.Transform()
            priest_hp.add_attr(self.priest_hp_trans)
            priest_hp_bg.add_attr(self.priest_trans)
            self.viewer.add_geom(priest_hp_bg)
            self.viewer.add_geom(priest_hp)

        if self.state is None: return None

        state = self.state
        pos, orb_pos, orb_dir, eye_pos, knock_cd, ult_cd, minor_cd, knock_ch, knock_dir, ult_ch, minor_ch, expel_ex, arc_ex, empowerment, stat, hp = state
        self.tank_trans.set_translation(pos[0], pos[1])
        self.warrior_trans.set_translation(pos[2], pos[3])
        self.mage_trans.set_translation(pos[4], pos[5])
        self.hunter_trans.set_translation(pos[6], pos[7])
        self.priest_trans.set_translation(pos[8], pos[9])
        self.boss_trans.set_translation(pos[10], pos[11])
        self.tank_hp_trans.set_scale(hp[0],1)
        self.tank_hp_trans.set_translation(-5*(1-hp[0])+pos[0],pos[1]+3)
        self.warrior_hp_trans.set_scale(hp[1],1)
        self.warrior_hp_trans.set_translation(-5*(1-hp[1])+pos[2],pos[3]+3)
        self.mage_hp_trans.set_scale(hp[2],1)
        self.mage_hp_trans.set_translation(-5*(1-hp[2])+pos[4],pos[5]+3)
        self.hunter_hp_trans.set_scale(hp[3],1)
        self.hunter_hp_trans.set_translation(-5*(1-hp[3])+pos[6],pos[7]+3)
        self.priest_hp_trans.set_scale(hp[4],1)
        self.priest_hp_trans.set_translation(-5*(1-hp[4])+pos[8],pos[9]+3)
        self.boss_hp_trans.set_scale(hp[5],1)
        self.boss_hp_trans.set_translation(-5*(1-hp[5])+pos[10],pos[11]+3)
        for i in range(0,35):
          self.orb_trans[i].set_translation(orb_pos[2*i], orb_pos[2*i+1])
        if (knock_ch[0] > 0):
          norm = math.sqrt(knock_dir[0] * knock_dir[0] + knock_dir[1] * knock_dir[1])
          dir_x = 2 * knock_dir[0] / norm
          dir_y = 2 * knock_dir[1] / norm
          self.knock = rendering.make_polygon(((pos[10]+dir_y,pos[11]-dir_x),(pos[10]-dir_y,pos[11]+dir_x),(pos[10]-dir_y+dir_x*100,pos[11]+dir_x+dir_y*100),(pos[10]+dir_y+dir_x*100,pos[11]-dir_x+dir_y*100)))
          self.knock.set_color(1,1,0.2)
          self.viewer.add_onetime(self.knock)
        if (minor_ch[1] > 0):
          tar = np.argmax(arc_ex)
          dir_x = pos[tar*2] - pos[10]
          dir_y = pos[tar*2+1] - pos[11]
          norm = math.sqrt(dir_x * dir_x + dir_y * dir_y)
          dir_x = dir_x / norm
          dir_y = dir_y / norm
          self.arc = rendering.make_polygon(((pos[10]+dir_y,pos[11]-dir_x),(pos[10]-dir_y,pos[11]+dir_x),(pos[2*tar]-dir_y,pos[2*tar+1]+dir_x),(pos[2*tar]+dir_y,pos[2*tar+1]-dir_x)))
          self.arc.set_color(0.5,0.5,0.7)
          self.viewer.add_onetime(self.arc)
        if (np.max(expel_ex) > 0):
          tar = np.argmax(expel_ex)
          self.expel = rendering.make_circle(5, 30, False)
          self.expel.set_color(1,1,0.2)
          trans = (self.tank_trans, self.warrior_trans, self.mage_trans, self.hunter_trans, self.priest_trans)
          self.expel.add_attr(trans[tar])
          self.viewer.add_onetime(self.expel)
        if (ult_ch[1] > 0):
          self.eye = rendering.make_circle(10, 30, False)
          self.eye.set_color(0.5,0.5,0.7)
          self.eye_trans = rendering.Transform()
          self.eye_trans.set_translation(eye_pos[0], eye_pos[1])
          self.eye.add_attr(self.eye_trans)
          self.viewer.add_onetime(self.eye)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
