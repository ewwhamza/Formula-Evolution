import pygame
import math
import sys

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 10
        self.width = 40
        self.length = 20
        
        try:
            original_image = pygame.image.load("F1car.png").convert_alpha()
            self.sprite = pygame.transform.scale(original_image, (self.width, self.length))
        except FileNotFoundError:
            print("[Error] f1car.jpg not found! Make sure it is in the same folder.")
            sys.exit()
            
        self.radars = [] 
        self.is_alive = True
        self.finished = False
        self.distance = 0 

    def update(self, track_surface):
        if not self.is_alive:
            return

        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.speed
        self.y -= math.sin(radians) * self.speed
        self.distance += self.speed

        self.radars.clear()
        
        for d in range(-90, 120, 45):
            self.check_radar(d, track_surface)
            
        self.check_collision(track_surface)

    def check_radar(self, degree, track_surface):
        length = 0
        radian = math.radians(self.angle + degree)
        x = int(self.x)
        y = int(self.y)

        while length < 400: 
            x = int(self.x + math.cos(radian) * length)
            y = int(self.y - math.sin(radian) * length)
            
            if x <= 0 or x >= track_surface.get_width() or y <= 0 or y >= track_surface.get_height():
                break
                
            if track_surface.get_at((x, y))[0:3] == (0, 0, 0):
                break
            length += 1

        dist = int(math.sqrt(math.pow(x - self.x, 2) + math.pow(y - self.y, 2)))
        self.radars.append([(x, y), dist])

    def check_collision(self, track_surface):
        if self.x <= 0 or self.x >= track_surface.get_width() or self.y <= 0 or self.y >= track_surface.get_height():
            self.is_alive = False
            return
            
        # We now check 3 points (Front, Center, Rear) to prevent teleporting over thin lines
        radians = math.radians(self.angle)
        points_to_check = [
            (int(self.x), int(self.y)), # Center
            (int(self.x + math.cos(radians) * 10), int(self.y - math.sin(radians) * 10)), # Front
            (int(self.x - math.cos(radians) * 10), int(self.y + math.sin(radians) * 10))  # Rear
        ]

        for px, py in points_to_check:
            # Screen bounds check for each individual point
            if px <= 0 or px >= track_surface.get_width() or py <= 0 or py >= track_surface.get_height():
                continue
                
            pixel_color = track_surface.get_at((px, py))[0:3]

            if pixel_color == (0, 0, 0): 
                self.is_alive = False
                break 
            elif pixel_color == (0, 255, 0) and self.distance > 500: 
                self.finished = True
                self.is_alive = False
                break
            elif pixel_color == (0, 0, 255) and self.distance > 50:
                self.is_alive = False
                break

    def draw(self, screen):
        if self.is_alive or self.finished:
            rotated_image = pygame.transform.rotate(self.sprite, self.angle)
            rect = rotated_image.get_rect(center=(self.x, self.y))
            screen.blit(rotated_image, rect.topleft)

            if self.is_alive:
                for radar in self.radars:
                    pos = radar[0]
                    pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), pos, 1)
                    pygame.draw.circle(screen, (0, 255, 0), pos, 3)