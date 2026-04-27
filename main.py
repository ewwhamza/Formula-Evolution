import pygame
import sys
import os
import neat
from car import Car

# --- Constants ---
WIDTH, HEIGHT = 1200, 800
FPS = 60
BG_COLOR = (0, 0, 0)
TRACK_COLOR = (255, 255, 255)
START_COLOR = (0, 0, 255) 
FINISH_COLOR = (0, 255, 0) 
BRUSH_SIZE = 50

# Increased height to guarantee it spans diagonal or vertical track sections
GATE_WIDTH = 15
GATE_HEIGHT = 200

track_surface = None
start_pos = (200, 200)
screen = None

# Telemetry Globals
generation_count = 0
fastest_lap_frames = float('inf')
stagnation_counter = 0  
autopilot_net = None
STAT_FONT = None

class AutopilotTriggered(Exception):
    pass

def draw_telemetry(screen, gen, alive, max_spd, best_lap_frames, stagnation):
    panel = pygame.Surface((230, 170)) 
    panel.set_alpha(200)
    panel.fill((0, 0, 0))
    screen.blit(panel, (WIDTH - 240, 10))

    lap_text = f"{best_lap_frames / FPS:.2f}s" if best_lap_frames != float('inf') else "--"
    
    if isinstance(stagnation, int):
        if best_lap_frames == float('inf'):
            stag_text = "Awaiting Lap 1"
        else:
            stag_text = f"{stagnation}/10"
    else:
        stag_text = stagnation 

    text_color = (255, 255, 255)
    t_gen = STAT_FONT.render(f"Generation: {gen}", True, text_color)
    t_alive = STAT_FONT.render(f"Cars Alive: {alive}", True, text_color)
    t_spd = STAT_FONT.render(f"Max Speed: {max_spd:.1f}", True, text_color)
    t_lap = STAT_FONT.render(f"Fastest Lap: {lap_text}", True, text_color)
    t_stag = STAT_FONT.render(f"Stagnation: {stag_text}", True, (255, 100, 100)) 

    screen.blit(t_gen, (WIDTH - 230, 20))
    screen.blit(t_alive, (WIDTH - 230, 50))
    screen.blit(t_spd, (WIDTH - 230, 80))
    screen.blit(t_lap, (WIDTH - 230, 110))
    screen.blit(t_stag, (WIDTH - 230, 140))

def eval_genomes(genomes, config):
    global track_surface, start_pos, screen, autopilot_net
    global generation_count, fastest_lap_frames, stagnation_counter
    
    generation_count += 1
    
    if fastest_lap_frames != float('inf'):
        stagnation_counter += 1 
    
    nets = []
    cars = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(start_pos[0], start_pos[1]))
        genome.fitness = 0.0
        ge.append(genome)

    clock = pygame.time.Clock()
    run = True
    frames = 0 
    
    while run:
        frames += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    print("\n[Hamza-Sys] Manual Override. Locking in best network...")
                    best_idx = max(range(len(ge)), key=lambda i: ge[i].fitness)
                    autopilot_net = nets[best_idx]
                    raise AutopilotTriggered()
                elif event.key == pygame.K_k:
                    print("\n[Hamza-Sys] Kill sequence activated! Terminating generation early...")
                    run = False

        screen.blit(track_surface, (0, 0))
        alive_cars = 0
        current_max_speed = 0
        
        for i, car in enumerate(cars):
            if car.is_alive:
                alive_cars += 1
                car.update(track_surface)
                
                if car.speed > current_max_speed:
                    current_max_speed = car.speed
                
                if car.finished:
                    ge[i].fitness += 100000 / frames
                    
                    if frames < fastest_lap_frames:
                        fastest_lap_frames = frames
                        stagnation_counter = 0 
                        autopilot_net = nets[i] 
                        print(f"[Hamza-Sys] Track record broken: {frames/FPS:.2f}s!")
                        
                    run = False 
                    break 
                
                ge[i].fitness += car.speed / 10.0

                inputs = [radar[1] for radar in car.radars] if len(car.radars) == 5 else [400]*5
                output = nets[i].activate(inputs)

                if output[0] > 0.5: car.speed = min(car.speed + 0.2, car.max_speed)
                elif output[1] > 0.5: car.speed = max(car.speed - 0.2, 3)
                else: car.speed = max(car.speed - 0.1, 3)
                    
                if output[2] > 0.5: car.angle += 4
                if output[3] > 0.5: car.angle -= 4

        if alive_cars == 0:
            run = False 

        for car in cars:
            if car.is_alive or car.finished:
                car.draw(screen)

        draw_telemetry(screen, generation_count, alive_cars, current_max_speed, fastest_lap_frames, stagnation_counter)
        pygame.display.flip()
        clock.tick(FPS)

    if fastest_lap_frames != float('inf') and stagnation_counter >= 10:
        print("\n[Hamza-Sys] 10 generations without improvement. Initiating Auto-Pilot!")
        if autopilot_net is None: 
            best_idx = max(range(len(ge)), key=lambda i: ge[i].fitness)
            autopilot_net = nets[best_idx]
        raise AutopilotTriggered()

def run_neat(config_file):
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    
    try:
        p.run(eval_genomes, 100)
    except AutopilotTriggered:
        return "AUTOPILOT"
    return "DONE"

def main():
    global track_surface, start_pos, screen, autopilot_net, STAT_FONT
    global generation_count, fastest_lap_frames, stagnation_counter
    
    pygame.init()
    pygame.font.init()
    STAT_FONT = pygame.font.SysFont("arial", 20, bold=True)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT F1 Simulator - Hamza")
    
    screen.fill(BG_COLOR)
    state = "DRAWING"
    auto_car = None
    clock = pygame.time.Clock()

    print("--- 1. DRAWING PHASE ---")
    print("Left-Click & Drag   : Draw Track")
    print("Right-Click         : Place START Gate (Blue)")
    print("Middle-Click        : Place FINISH Gate (Green)")
    print("SPACEBAR            : Start Training AI")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                        
            if state == "DRAWING" and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                print("\n--- 2. TRAINING PHASE ---")
                print("Press 'A' at any time to enter Autopilot mode.")
                print("Press 'K' at any time to Kill the current generation.")
                
                generation_count = 0
                fastest_lap_frames = float('inf')
                stagnation_counter = 0
                
                track_surface = screen.copy()
                config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
                
                result = run_neat(config_path)
                
                if result == "AUTOPILOT":
                    print("\n--- 3. AUTOPILOT PHASE ---")
                    state = "AUTOPILOT"
                    auto_car = Car(start_pos[0], start_pos[1])

        if state == "DRAWING":
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            if mouse_buttons[0]: 
                pygame.draw.circle(screen, TRACK_COLOR, mouse_pos, BRUSH_SIZE)
            elif mouse_buttons[2]: 
                rect = (mouse_pos[0] - GATE_WIDTH//2, mouse_pos[1] - GATE_HEIGHT//2, GATE_WIDTH, GATE_HEIGHT)
                pygame.draw.rect(screen, START_COLOR, rect)
                start_pos = mouse_pos
            elif mouse_buttons[1]: 
                rect = (mouse_pos[0] - GATE_WIDTH//2, mouse_pos[1] - GATE_HEIGHT//2, GATE_WIDTH, GATE_HEIGHT)
                pygame.draw.rect(screen, FINISH_COLOR, rect)

            pygame.display.flip()
            
        elif state == "AUTOPILOT":
            screen.blit(track_surface, (0, 0))
            
            if not auto_car.is_alive or auto_car.finished:
                auto_car = Car(start_pos[0], start_pos[1])
                
            auto_car.update(track_surface)
            
            inputs = [r[1] for r in auto_car.radars] if len(auto_car.radars) == 5 else [400]*5
            output = autopilot_net.activate(inputs)

            if output[0] > 0.5: auto_car.speed = min(auto_car.speed + 0.2, auto_car.max_speed)
            elif output[1] > 0.5: auto_car.speed = max(auto_car.speed - 0.2, 3)
            else: auto_car.speed = max(auto_car.speed - 0.1, 3)
                
            if output[2] > 0.5: auto_car.angle += 4
            if output[3] > 0.5: auto_car.angle -= 4
                
            auto_car.draw(screen)
            draw_telemetry(screen, "Autopilot", "1 (Elite)", auto_car.speed, fastest_lap_frames, "Locked")
            pygame.display.flip()
            
        clock.tick(FPS)

if __name__ == "__main__":
    main()