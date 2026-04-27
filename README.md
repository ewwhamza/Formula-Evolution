# NeuroApex F1 🏎️ 🧠

An interactive, 2D autonomous racing simulator where neural networks learn to drive from scratch using Reinforcement Learning concepts and the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. 

Instead of pre-training a model on a dataset, this project spawns generations of AI drivers on a blank canvas. By drawing custom tracks and placing start/finish gates, you can watch the AI evolve generation by generation as it figures out how to navigate complex corners and optimize lap times.

## ✨ Features

* **Dynamic Environment Generation:** Draw custom tracks in real-time using mouse inputs. No two training environments have to be the same.
* **Raycast Vision:** Cars "see" the track using 5 directional raycast sensors that measure the distance to the track walls.
* **Neuroevolution (NEAT):** The algorithm breeds the highest-performing neural networks from each generation, passing their "genes" (weights and biases) to the next batch of cars.
* **Anti-Cheat Logic:** Includes distance-tracking physics to prevent the AI from gaming the system by driving backward over the finish line.
* **Live Telemetry Dashboard:** Tracks generation count, surviving cars, top speeds, and tracks the fastest lap down to the frame.
* **Early-Stopping & Autopilot:** The system automatically locks in the optimal brain if lap times stagnate for 10 generations, transitioning into an endless Autopilot victory lap. Also includes manual overrides.

## 🛠️ Tech Stack
* **Python 3.x**
* **Pygame:** Engine for 2D physics, rendering, and dynamic surface masking.
* **neat-python:** Core library handling the genetic algorithms and neural network topology creation.

