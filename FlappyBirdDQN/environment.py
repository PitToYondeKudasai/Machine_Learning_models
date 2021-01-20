#################
## ENVIRONMENT ##
#################

import pygame
import random

pygame.init()

class tube:
    def __init__(self, render = False):
        self.x = 300
        self.y = random.randint(-75, 150)
        self.render = render
        
    def move_and_draw(self, env):
        self.x -= env.mov_vel
        if self.render:
            env.screen.blit(env.tube_down, (self.x, self.y + 210))
            env.screen.blit(env.tube_up, (self.x, self.y - 210))

    def between_tubes(self, env):
        tolerance = 5
        bird_rightside = env.bird_x + env.bird.get_width() - tolerance
        bird_leftside = env.bird_x + tolerance
        tube_rightside = self.x + env.tube_down.get_width()
        tube_leftside = self.x
        if bird_rightside > tube_leftside and bird_leftside < tube_rightside:
            return True

    def collision(self, env):
        tolerance = 5
        done = False

        bird_rightside = env.bird_x + env.bird.get_width() - tolerance
        bird_leftside = env.bird_x + tolerance
        tube_rightside = self.x + env.tube_down.get_width()
        tube_leftside = self.x
        
        bird_upside = env.bird_y + tolerance
        bird_downside = env.bird_y + env.bird.get_height() - tolerance
        tube_upside = self.y + 110
        tube_downside = self.y + 210

        # IL primo if controlla se l'uccello si trova tra due tubi
        if bird_rightside > tube_leftside and bird_leftside < tube_rightside:
            if bird_upside < tube_upside or bird_downside > tube_downside:
                done = env.you_lose()
        return done 

class Env:
    def __init__(self, render = False):
        self.background = pygame.image.load('images/background.png')
        self.bird = pygame.image.load('images/bird.png')
        self.base = pygame.image.load('images/base.png')
        self.tube_down = pygame.image.load('images/tube.png')
        self.gameover = pygame.image.load('images/gameover.png')
        self.tube_up = pygame.transform.flip(self.tube_down, False, True)
        self.screen_size = (288, 512)
        self.mov_vel = 5
        self.gamma = 0.99
        self.bird_x = 60
        self.bird_y = 150
        self.bird_vel_y = 0
        self.base_x = 0
        self.tubes = []
        self.between_tubes = False
        self.render = render
        self.rew_base = 0.5
        self.rew_pass_pipe = 10
        self.rew_game_over = -5
        if self.render:
            self.screen = pygame.display.set_mode((288, 512))
            self.fps = 50
            self.font = pygame.font.SysFont('Comic Sans MS', 50, bold = True)

    def reset(self):
        self.bird_x, self.bird_y = 60, 150
        self.bird_vel_y = 0
        self.base_x = 0
        self.tubes = []
        self.scores = 0
        self.between_tubes = False
        self.tubes.append(tube(self.render))
        return self.bird_y, self.bird_vel_y, self.tubes[-1].x, self.tubes[-1].y
    
    def update(self):
        pygame.display.flip()
        pygame.display.update()
        pygame.time.Clock().tick(self.fps)

    def get_next_tube(self):
        if len(self.tubes) > 1:
            return self.tubes[-2]
        return self.tubes[-1]

    def you_lose(self):
        if self.render:
            self.screen.blit(self.gameover, (50,180))
            self.update()
        return True

    def draw_objects(self):
        if self.render:
            self.screen.blit(self.background, (0,0))
        for t in self.tubes:
            t.move_and_draw(self)
        if self.render:    
            self.screen.blit(self.bird, (self.bird_x, self.bird_y))
            self.screen.blit(self.base, (self.base_x, 400))
            score_render = self.font.render(str(self.scores), 1, (255, 255, 255))
            self.screen.blit(score_render, (144, 0))

    def step(self,action):
        rew = self.rew_base
        if action == 1:
            self.bird_vel_y = -5
        self.base_x -= self.mov_vel
        if self.base_x < -45:
            self.base_x = 0
            
        # Gravity       
        self.bird_vel_y += 2
        self.bird_y += self.bird_vel_y
        self.draw_objects()
        
        if self.tubes[-1].x < 150:
            self.tubes.append(tube(self.render))

        obs = (self.bird_y, self.bird_vel_y, self.get_next_tube().x, self.get_next_tube().y)
    

        for t in self.tubes:
            done = t.collision(self)
            if done:
                rew = self.rew_game_over
                return obs, rew, done, 'info'
                      
        if not self.between_tubes:
            for t in self.tubes:
                if t.between_tubes(self):
                    self.between_tubes = True
                    break
        else:
            self.between_tubes = False
            for t in self.tubes:
                if t.between_tubes(self):
                    self.between_tubes = True
                    break
            if not self.between_tubes:
                self.scores += 1
                rew = self.rew_pass_pipe
                    
        if self.bird_y > 380:
            done = self.you_lose()
            rew = self.rew_game_over
        return obs, rew, done, 'info'
    
    def has_quit(self):
        if pygame.get_init():
            pygame.quit()
