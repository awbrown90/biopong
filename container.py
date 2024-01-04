import pygame
import sys
import random
import numpy as np
import cv2

class BioPong:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # Game window settings
        self.WIDTH, self.HEIGHT = 800, 600

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        # Paddle and ball
        self.paddle_height = 100
        self.paddle_width = 10
        self.ball_size = 10

        self.reset_paddle_y = self.HEIGHT // 2 - self.paddle_height // 2

        self.paddle_y = self.reset_paddle_y

        self.reset_ball_x = self.WIDTH // 2 - self.ball_size // 2
        self.reset_ball_y = self.HEIGHT // 2 - self.ball_size // 2
        self.ball_x, self.ball_y = self.reset_ball_x, self.reset_ball_y

        self.reset_ball_speed_x = 5
        self.reset_ball_speed_y = 5
        self.ball_speed_x, self.ball_speed_y = self.reset_ball_speed_x, self.reset_ball_speed_y
        
        self.reset_counter = 0
        self.reset_duration = 20

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.rally = 0
        self.in_a_row = -1
        self.trajectory_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
    
    def reset(self):
        
        if self.reset_counter > 1:
            # Place paddle and ball in random positions for a short duration
            self.paddle_y = random.randint(0, self.HEIGHT - self.paddle_height)
            self.ball_x, self.ball_y = random.randint(0, self.WIDTH - self.ball_size), random.randint(0, self.HEIGHT - self.ball_size)
            self.reset_counter -= 1
        else:
            self.paddle_y = self.reset_paddle_y
            self.ball_x, self.ball_y = self.reset_ball_x, self.reset_ball_y
            self.ball_speed_x, self.ball_speed_y = self.reset_ball_speed_x, self.reset_ball_speed_y
            self.reset_counter -= 1


        return self._get_state()
    
    def step(self, action):
        self._update_paddle(action)
        self._update_ball()

        if self.reset_counter > 0:
            self.reset()

        state = self._get_state()
        reward = 0  # No reward in this simple game
        done = False

        return state

    def checkpoint(self):
        self.reset_paddle_y = self.paddle_y
        self.reset_ball_x = self.ball_x-10
        self.reset_ball_y = self.ball_y
        #self.reset_ball_speed_x = self.ball_speed_x
        #self.reset_ball_speed_y = self.ball_speed_y


    def render(self, value1, value2, value3):
        self.screen.fill(self.BLACK)

        render_paddle_y = (value1+0.5) * self.HEIGHT
        render_ball_x = (value2+0.5) * self.WIDTH
        render_ball_y = (value3+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()
        self.clock.tick(60)

    def render2(self, value11, value12, value13, value21, value22, value23):
        self.screen.fill(self.BLACK)

        render_paddle_y = (value11+0.5) * self.HEIGHT
        render_ball_x = (value12+0.5) * self.WIDTH
        render_ball_y = (value13+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        render_paddle_y = (value21+0.5) * self.HEIGHT
        render_ball_x = (value22+0.5) * self.WIDTH
        render_ball_y = (value23+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.BLUE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.BLUE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()

        # Convert the Surface to a format compatible with OpenCV
        image_data = pygame.surfarray.array3d(self.screen)
        image_data = np.transpose(image_data, (1, 0, 2))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # Tick the clock
        self.clock.tick(60)

        return image_data

    def render2_traj(self, value11, value12, value13, value21, value22, value23, trajs=None):
        self.screen.fill(self.BLACK)

        if trajs is None:
             self.screen.blit(self.trajectory_surface, (0, 0))
        else:
            for traj in trajs:

                value1 = traj[0]
                value2 = traj[1]
                value3 = traj[2]

                render_paddle_y = (value1+0.5) * self.HEIGHT
                render_ball_x = (value2+0.5) * self.WIDTH
                render_ball_y = (value3+0.5) * self.HEIGHT

                # Draw the paddle
                pygame.draw.rect(self.screen, self.RED, (0, render_paddle_y, self.paddle_width, self.paddle_height))

                # Draw the ball
                pygame.draw.rect(self.screen, self.RED, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        render_paddle_y = (value11+0.5) * self.HEIGHT
        render_ball_x = (value12+0.5) * self.WIDTH
        render_ball_y = (value13+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        render_paddle_y = (value21+0.5) * self.HEIGHT
        render_ball_x = (value22+0.5) * self.WIDTH
        render_ball_y = (value23+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.BLUE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.BLUE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()

        # Convert the Surface to a format compatible with OpenCV
        image_data = pygame.surfarray.array3d(self.screen)
        image_data = np.transpose(image_data, (1, 0, 2))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # Tick the clock
        self.clock.tick(60)

        return image_data

    def render_traj(self, value1, value2, value3, trajs):
        self.screen.fill(self.BLACK)

        render_paddle_y = (value1+0.5) * self.HEIGHT
        render_ball_x = (value2+0.5) * self.WIDTH
        render_ball_y = (value3+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        self.trajectory_surface.fill(self.BLACK)
        for traj in trajs:

            value1 = traj[0]
            value2 = traj[1]
            value3 = traj[2]

            render_paddle_y = (value1+0.5) * self.HEIGHT
            render_ball_x = (value2+0.5) * self.WIDTH
            render_ball_y = (value3+0.5) * self.HEIGHT

            # Draw the paddle
            pygame.draw.rect(self.trajectory_surface, self.RED, (0, render_paddle_y, self.paddle_width, self.paddle_height))

            # Draw the ball
            pygame.draw.rect(self.trajectory_surface, self.RED, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))
        self.screen.blit(self.trajectory_surface, (0, 0))

        pygame.display.flip()

        # Convert the Surface to a format compatible with OpenCV
        image_data = pygame.surfarray.array3d(self.screen)
        image_data = np.transpose(image_data, (1, 0, 2))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # Tick the clock
        self.clock.tick(60)

        return image_data

    def render_traj2(self, value1, value2, value3, trajs1, trajs2):
        self.screen.fill(self.BLACK)

        render_paddle_y = (value1+0.5) * self.HEIGHT
        render_ball_x = (value2+0.5) * self.WIDTH
        render_ball_y = (value3+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        for traj in trajs1:

            value1 = traj[0]
            value2 = traj[1]
            value3 = traj[2]

            render_paddle_y = (value1+0.5) * self.HEIGHT
            render_ball_x = (value2+0.5) * self.WIDTH
            render_ball_y = (value3+0.5) * self.HEIGHT

            # Draw the paddle
            pygame.draw.rect(self.screen, self.RED, (0, render_paddle_y, self.paddle_width, self.paddle_height))

            # Draw the ball
            pygame.draw.rect(self.screen, self.RED, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        for traj in trajs2:

            value1 = traj[0]
            value2 = traj[1]
            value3 = traj[2]

            render_paddle_y = (value1+0.5) * self.HEIGHT
            render_ball_x = (value2+0.5) * self.WIDTH
            render_ball_y = (value3+0.5) * self.HEIGHT

            # Draw the paddle
            pygame.draw.rect(self.screen, self.GREEN, (0, render_paddle_y, self.paddle_width, self.paddle_height))

            # Draw the ball
            pygame.draw.rect(self.screen, self.GREEN, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        pygame.display.flip()

        # Convert the Surface to a format compatible with OpenCV
        image_data = pygame.surfarray.array3d(self.screen)
        image_data = np.transpose(image_data, (1, 0, 2))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # Tick the clock
        self.clock.tick(60)

        return image_data

    def render3(self, value11, value12, value13, value21, value22, value23, value31, value32, value33, energies):
        self.screen.fill(self.BLACK)

        render_paddle_y = (value11+0.5) * self.HEIGHT
        render_ball_x = (value12+0.5) * self.WIDTH
        render_ball_y = (value13+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.WHITE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.WHITE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        render_paddle_y = (value21+0.5) * self.HEIGHT
        render_ball_x = (value22+0.5) * self.WIDTH
        render_ball_y = (value23+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.BLUE, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.BLUE, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        render_paddle_y = (value31+0.5) * self.HEIGHT
        render_ball_x = (value32+0.5) * self.WIDTH
        render_ball_y = (value33+0.5) * self.HEIGHT

        # Draw the paddle
        pygame.draw.rect(self.screen, self.RED, (0, render_paddle_y, self.paddle_width, self.paddle_height))

        # Draw the ball
        pygame.draw.rect(self.screen, self.RED, (render_ball_x, render_ball_y, self.ball_size, self.ball_size))

        # Draw energy
        max_energy = 2.0
        border_color = (50, 50, 50)  # Red color for the border
        border_width = 1
        rect_width = 5
        offset = rect_width + 2 * border_width
        for idx, energy in enumerate(energies):
            if energy < 1:
                color_intensity = (0, 255 * min(energy/max_energy,1), 0)
            else:
                color_intensity = (255 * min(energy/max_energy,1), 0, 0)
            pygame.draw.rect(self.screen, border_color, (0.5 * self.WIDTH + idx * offset, 0.5 * self.HEIGHT, rect_width + border_width, rect_width + border_width))
            pygame.draw.rect(self.screen, color_intensity, (0.5 * self.WIDTH + idx * offset, 0.5 * self.HEIGHT, rect_width, rect_width))

        pygame.display.flip()
        self.clock.tick(60)
    
    def _update_paddle(self, action):
        #keys = pygame.key.get_pressed()
        
        if action == 0:
            self.paddle_y -= 5
        elif action == 1:
            self.paddle_y += 0
        elif action == 2:
            self.paddle_y += 5

        # Clamp paddle position
        self.paddle_y = max(0, min(self.HEIGHT - self.paddle_height, self.paddle_y))
    
    def _update_ball(self):
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # Check for collisions
        if self.ball_y <= 0 or self.ball_y + self.ball_size >= self.HEIGHT:
            self.ball_speed_y = -self.ball_speed_y

        if self.ball_x <= self.paddle_width and (self.paddle_y <= self.ball_y <= self.paddle_y + self.paddle_height):
            self.ball_speed_x = -self.ball_speed_x

        # Bounce off the right wall
        if self.ball_x + self.ball_size >= self.WIDTH:
            self.ball_speed_x = -self.ball_speed_x
            self.rally += 1
            self.in_a_row += 1
            #if self.in_a_row > 0:
            #    print('set checkpoint')
            #    self.checkpoint()

        # Reset the ball position if it goes past the paddle
        if self.ball_x < 0:
            self.reset_counter = self.reset_duration
            self.reset()
            self.in_a_row = -1

    def _get_score(self):
        return self.rally, self.in_a_row
    
    def _get_state(self):
        self.screen.fill(self.BLACK)

        # Get the encoded signal for paddle and ball positions
        paddle_pos = self.paddle_y / self.HEIGHT -0.5
        ball_pos_x = self.ball_x / self.WIDTH -0.5
        ball_pos_y = self.ball_y / self.HEIGHT -0.5

        return [paddle_pos, ball_pos_x, ball_pos_y]
