import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy  as np
import os 
import torch.nn as nn
import matplotlib.pyplot as plt
import time  
import numpy as np 
import pandas as pd 
from MultiAgent import Agent 
import math




# Initialize Pygame module
pygame.init()

# Create a font object with the specified font type ("comicsans") and size (50)
font = pygame.font.SysFont("comicsans", 50)

# Alternatively, you can use the following line to choose a different font ('arial') and size (25)
# font = pygame.font.SysFont('arial', 25)

# Enum to represent directions: RIGHT, LEFT, UP, DOWN
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple representing a point with 'x' and 'y' coordinates
Point = namedtuple('Point', 'x, y')

# Define color constants
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Size of each block in the game grid
BLOCK_SIZE = 30

# Initial speed of the snake
SPEED = 50

# Frames per second for the game
FPS = 50

# Number of apples to be placed on the game grid
NUM_APPLES = 5

# Number of teams in the game
NumberOFTeams = 2




class SnakeGame:

  # Class definition for the game environment
    def __init__(self, w=600, h=600, num_snakes=1):
        # Initialize the game window dimensions
        self.w = w
        self.h = h

        # Load and scale the background image
        self.background_image = pygame.transform.scale(pygame.image.load("gray.jpg"), (self.w, self.h))

        # Set up the display window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

        # Set up the game clock
        self.clock = pygame.time.Clock()

        # Define rewards for different events in the game
        self.Belohnung = {
            "Colliding with itself": [-10, -1, -10, -1],
            "Collidind with wall": [-10, -1, -10, -1],
            "Eat gifted apple": [-20, -20, -20, -20],
            "Eat apple with same Color": [50, 10, 20, 30],
            "eat opoent Apple": [50, 5, 15, 18],
            "Collided with Team mate": [-80, -100, -90, -110],
            "eat opoent": [100, 70, 80, 85],
            "opoent eated me": [-40, -50, -20, -60]
        }

        # Load and scale images for the blue snake facing different directions
        self.rightblue = pygame.image.load("BlueSnake/right.png").convert_alpha()
        self.rightblue = pygame.transform.scale(self.rightblue, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.leftblue = pygame.image.load("BlueSnake/left.png").convert_alpha()
        self.leftblue = pygame.transform.scale(self.leftblue, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.downblue = pygame.image.load("BlueSnake/down.png").convert_alpha()
        self.downblue = pygame.transform.scale(self.downblue, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.upblue = pygame.image.load("BlueSnake/up.png").convert_alpha()
        self.upblue = pygame.transform.scale(self.upblue, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        # Load and scale images for the red snake facing different directions
        self.rightred = pygame.image.load("RedSnake/right.png").convert_alpha()
        self.rightred = pygame.transform.scale(self.rightred, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.leftred = pygame.image.load("RedSnake/left.png").convert_alpha()
        self.leftred = pygame.transform.scale(self.leftred, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.downred = pygame.image.load("RedSnake/down.png").convert_alpha()
        self.downred = pygame.transform.scale(self.downred, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.upred = pygame.image.load("RedSnake/up.png").convert_alpha()
        self.upred = pygame.transform.scale(self.upred, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        # Load and scale images for different types of apples
        self.blackapple = pygame.image.load("Apples/black.png").convert_alpha()
        self.blackapple = pygame.transform.scale(self.blackapple, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.redapple = pygame.image.load("Apples/red.png").convert_alpha()
        self.redapple = pygame.transform.scale(self.redapple, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.blueapple = pygame.image.load("Apples/blue.png").convert_alpha()
        self.blueapple = pygame.transform.scale(self.blueapple, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        self.whiteapple = pygame.image.load("Apples/white.png").convert_alpha()
        self.whiteapple = pygame.transform.scale(self.whiteapple, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        # Load and scale image for the snake tail
        self.tail = pygame.image.load("tail.png").convert_alpha()
        self.tail = pygame.transform.scale(self.whiteapple, (BLOCK_SIZE-2, BLOCK_SIZE-2))

        # Initialize variables for the number of snakes, frames, etc.
        self.num_snakes = num_snakes
        self.snakes = []
        self.frames_since_last_action = [0] * self.num_snakes
        self.MAX_FRAMES_INACTIVITY = 100000
        self.start_time = []

        # Initialize variables for tracking game statistics
        self.EatenTeamMate = [0] * self.num_snakes
        self.EatenOponent = [0] * self.num_snakes
        self.AppleSameColor = [0] * self.num_snakes
        self.GiftedApples = [0] * self.num_snakes
        self.OponentApples = [0] * self.num_snakes
        self.TeamSCore = [0] * NumberOFTeams

        # Define a list of colors for the snakes
        self.colors = [
            "blue",
            "red",
            "white",
            "black",
            "cyan",
            "magenta",
            "yellow",
            "orange",
        ]

        # Define a list of restricted colors
        self.RestrictedColors = ["white", "black"]

        # Assign colors to each snake
        current_index = 0
        self.SnakeColors = []
        for _ in range(self.num_snakes):
            current_index = current_index % 2
            self.SnakeColors.append(self.colors[current_index])
            current_index += 1

        # Initialize variables for the colors of different types of food
        self.FoodColors = []
        current_index = 0
        for index in range(NUM_APPLES):
            current_index = current_index % 4
            self.FoodColors.append(self.colors[current_index])
            current_index += 1

        # Reset the game state
        self.reset()



    def reset_snake(self, snake_index):

        snake_positions = []  # List to store the positions of existing snakes

        for i, snake in enumerate(self.snakes):
            if i != snake_index:
                snake_positions.extend(snake)

        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            head = Point(x, y)

            # Check if the generated position conflicts with existing snakes
            if head not in snake_positions:
                self.heads[snake_index] = head
                self.snakes[snake_index] = [head,
                                            Point(head.x - BLOCK_SIZE, head.y),
                                            Point(head.x - (2 * BLOCK_SIZE), head.y)]
                self.directions[snake_index] = Direction.RIGHT
                self.game_over[snake_index] = False
                self.score[snake_index] = 0
                self.start_time[snake_index] = time.time()
                self.Apple_EatenSnakes[snake_index] = [0, 0]
                self.snakesMovment[snake_index] = 0
                break


    def reset(self):
        """
        Reset the game state for a new round.

        This function initializes or resets various attributes such as frame iteration,
        start time, scores, snake positions, game over status, and apple positions.

        Attributes:
        - frame_iteration: Counter for the number of frames in the current iteration.
        - start_time: List storing the start time for each snake.
        - score: List keeping track of the score for each snake.
        - Apple_EatenSnakes: List storing information about apples eaten by each snake.
        - directions: List representing the current direction of each snake.
        - heads: List to store snake head positions.
        - snakes: List to store snake body segments.
        - game_over: List indicating whether each snake has reached a game-over state.
        - snakesMovment: List storing the movement status of each snake.
        - snake_positions: List storing the positions of existing snakes.
        - food: List to store apple positions.

        Procedure:
        - Initialize frame_iteration, start_time, score, Apple_EatenSnakes, directions, heads, snakes,
        game_over, and snakesMovment.
        - Generate random starting positions for each snake, ensuring no position conflicts with existing snakes.
        - Populate heads and snakes lists with the generated positions.
        - Initialize an empty list for food (apple positions).
        - Call _place_food() to generate initial apples on the game board.
        """
        # Initialize frame-related attributes
        self.frame_iteration = 0
        self.start_time = [time.time() for _ in range(self.num_snakes)]
        self.score = [0] * self.num_snakes
        self.Apple_EatenSnakes = [[0, 0]] * self.num_snakes

        # Initialize direction, head, and body-related attributes
        self.directions = [Direction.RIGHT for _ in range(self.num_snakes)]
        self.heads = []  # List to store snake head positions
        self.snakes = []  # List to store snake body segments
        self.game_over = [False] * self.num_snakes
        self.snakesMovment = [0] * self.num_snakes

        # Generate random starting positions for each snake
        snake_positions = []  # List to store the positions of existing snakes

        for _ in range(self.num_snakes):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                head = Point(x, y)

                # Check if the generated position conflicts with existing snakes
                if head not in snake_positions:
                    snake_positions.append(head)
                    break

            self.heads.append(head)
            snake = [head, Point(head.x - BLOCK_SIZE, head.y), Point(head.x - (2 * BLOCK_SIZE), head.y)]
            self.snakes.append(snake)

        # Initialize food (apple positions)
        self.food = []
        self._place_food()  # Generate initial apples



    
    def convert_colors_to_binary(self, color_index):
        """
        Convert a color index to a binary representation.

        This function takes an integer color index and converts it into a binary representation
        in the form of a list of three integers (0 or 1), representing the binary values of Red, Green, and Blue channels.

        Parameters:
        - color_index (int): An integer representing the color index.

        Returns:
        - binary_colors (list): A list containing three integers (0 or 1), representing the binary values of Red, Green, and Blue channels.

        Procedure:
        - Initialize an empty list `binary_colors` to store the binary representation.
        - Based on the provided `color_index`, append the corresponding binary values for Red, Green, and Blue channels to `binary_colors`.
        - If the `color_index` is out of the expected range, print an error message.

        Example:
        - If `color_index` is 2, the function returns [[0, 1, 0]], representing the binary values for RGB.

        Note: This function assumes a color_index in the range [0, 7] as it corresponds to 8 possible colors.
        """
        binary_colors = []

        # Convert the color index to binary representation
        if color_index == 0:
            binary_colors.append([0, 0, 0])
        elif color_index == 1:
            binary_colors.append([0, 0, 1])
        elif color_index == 2:
            binary_colors.append([0, 1, 0])
        elif color_index == 3:
            binary_colors.append([0, 1, 1])
        elif color_index == 4:
            binary_colors.append([1, 0, 0])
        elif color_index == 5:
            binary_colors.append([1, 0, 1])
        elif color_index == 6:
            binary_colors.append([1, 1, 0])
        elif color_index == 7:
            binary_colors.append([1, 1, 1])
        else:
            # Handle the case of an invalid color index
            print("Invalid color index.")

        return binary_colors

                
    def get_state(self):
        """
        Generate the state representation for each snake in the game environment.

        This function compiles a state representation for each snake, including information about the snake's position,
        surroundings, actions of other snakes, lengths of other snakes, and the location and colors of food items.

        Returns:
        - states (list): A list containing the state representation for each snake in the game environment.

        Procedure:
        - Iterate through each snake in the game.
            - Collect information about the snake's head position and surrounding pixels.
            - Determine the direction in which the snake is moving.
            - Collect information about other snakes, such as their positions, lengths, and actions.
            - Compile a state representation for the current snake based on the collected information.
            - Add the state representation to the list of states.

        Example:
        - A state representation for a snake might include information about its surroundings, the presence of other snakes,
        their lengths, and the location and colors of food items.

        Note: This function relies on several helper functions and assumes their definitions are available in the code.
        """
        states = []

        # Iterate through each snake in the game
        for snake_index in range(self.num_snakes):
            # Access the head of the snake
            head = self.heads[snake_index]

            # Define points in the vicinity of the snake's head
            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)

            # Check the direction in which the snake is moving
            dir_l = self.directions[snake_index] == Direction.LEFT
            dir_r = self.directions[snake_index] == Direction.RIGHT
            dir_u = self.directions[snake_index] == Direction.UP
            dir_d = self.directions[snake_index] == Direction.DOWN

            # Collect positions, lengths, and actions of other snakes, as well as colors
            opponent_positions = []
            opponent_lengths = []
            opponent_actions = []

            for snake_idx in range(self.num_snakes):
                if snake_idx != snake_index:
                    opponent_positions.append(self.heads[snake_idx])
                    opponent_lengths.append(len(self.snakes[snake_idx]))

                    opponent_action = [0, 0, 0, 0]
                    if self.directions[snake_idx] == Direction.LEFT:
                        opponent_action = [1, 0, 0, 0]
                    elif self.directions[snake_idx] == Direction.RIGHT:
                        opponent_action = [0, 1, 0, 0]
                    elif self.directions[snake_idx] == Direction.UP:
                        opponent_action = [0, 0, 1, 0]
                    elif self.directions[snake_idx] == Direction.DOWN:
                        opponent_action = [0, 0, 0, 1]
                    opponent_actions.append(opponent_action)

            # Compile the state representation for the current snake
            state = [
                int((dir_r and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_d)))),
                # Danger right
                int((dir_u and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_d)))),
                # Danger left
                int((dir_d and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_d)))),

                # Move direction
                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d)
            ]

            # Add the opponent's move
            for action in opponent_actions:
                state += action

            # Add lengths
            my_length = len(self.snakes[snake_index])

            for opponent_length in opponent_lengths:
                if my_length > opponent_length:
                    state += [1, 0]  # Snake length is greater than opponent
                elif my_length < opponent_length:
                    state += [0, 1]  # Snake length is smaller than opponent
                else:
                    state += [0, 0]  # Snake length is equal to opponent

            # Add food location for each snake
            for food_item in self.food:
                state += [
                    int(food_item.x < head.x),   # food left
                    int(food_item.x > head.x),   # food right
                    int(food_item.y < head.y),   # food up
                    int(food_item.y > head.y)    # food down
                ]

            # Add agent colors
            for color_index in range(len(self.snakes)):
                item = self.SnakeColors[color_index]
                value = self.convert_colors_to_binary(self.colors.index(item))

                for item in value:
                    state.extend(item)

            # Add food colors
            for food in range(len(self.food)):
                item = self.FoodColors[food]
                value = self.convert_colors_to_binary(self.colors.index(item))

                for item in value:
                    state.extend(item)

            states.append(np.array(state, dtype=int))

        return states




    
    def CheckForGetState(self, snake_index, pt=None):

        """
        Check for collision or boundary violation at a given point for a specific snake.

        Parameters:
        - snake_index (int): Index of the snake in the game.
        - pt (Point, optional): Point to check for collision. If not provided, defaults to the head of the snake.

        Returns:
        - collision (bool): True if collision or boundary violation occurs, False otherwise.

        Procedure:
        - If a specific point (`pt`) is not provided, use the head of the snake.
        - Check if the point hits the boundaries of the game window.
        - Check if the point collides with the snake's own body.
        - Check if the point collides with the body of other snakes.
            - If the colors of the colliding snakes are the same, or the colliding snake is longer, consider it a collision.
        - Check if the point collides with a food item of a restricted color.

        Example:
        - If `CheckForGetState(0)` returns True, it indicates a collision or boundary violation for the first snake.

        Note: This function assumes the existence of the `Point`, `self.snakes`, `self.SnakeColors`, `self.food`,
        `self.FoodColors`, `self.RestrictedColors`, and `BLOCK_SIZE` variables or classes in the code.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Hits itself
        for i, body in enumerate(self.snakes[snake_index][1:]):
            if pt == body:
                return True

        # Hits other snakes
        for i, snake in enumerate(self.snakes):
            if i != snake_index:
                if pt in snake:
                    if self.SnakeColors[i] == self.SnakeColors[snake_index] or len(snake) >= len(self.snakes[snake_index]):
                        return True

        # Hits food of restricted color
        if pt in self.food:
            for i, apple in enumerate(self.food):
                if pt == apple and self.FoodColors[i] in self.RestrictedColors:
                    return True

        return False

    
    

    def _place_food(self):
        """
        Place food items on the game board.

        This function generates and places food items on the game board. It ensures that food items are placed
        at unoccupied positions to avoid collisions with snakes.

        Procedure:
        - Initialize an empty list `self.food` to store food positions.
        - Initialize a set `occupied_positions` to keep track of positions where food is already placed.
        - Generate and place the desired number of food items (`NUM_APPLES`) on the game board.
            - Ensure each food item is placed at an unoccupied position.
            - Randomly select positions within the game boundaries for each food item.

        Example:
        - If `_place_food()` is called, it populates the `self.food` list with positions for food items.

        Note: This function assumes the existence of `Point`, `NUM_APPLES`, `self.w`, `self.h`, `BLOCK_SIZE`, and `random`
        variables or classes in the code.
        """
        self.food = []
        occupied_positions = set()

        # Generate and place the desired number of food items
        for _ in range(NUM_APPLES):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                position = (x, y)

                # Check if the position is unoccupied
                if position not in occupied_positions:
                    occupied_positions.add(position)
                    self.food.append(Point(x, y))
                    break


    

                            
    def calculate_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points in a 2D space.

        Parameters:
        - point1 (tuple): Tuple representing the (x, y) coordinates of the first point.
        - point2 (tuple): Tuple representing the (x, y) coordinates of the second point.

        Returns:
        - distance (float): Euclidean distance between the two points.

        Procedure:
        - Extract x and y coordinates from both points.
        - Use the Euclidean distance formula to calculate the distance between the two points.

        Example:
        - If `calculate_distance((0, 0), (3, 4))` is called, it returns 5.0, as it calculates the distance
        between points (0, 0) and (3, 4) in a 2D space.

        Note: This function assumes the use of the `math` library for the square root operation.
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

 
    def play_step(self, actions):
        """
        Execute a single step in the game based on the provided actions.

        Parameters:
        - actions (list): List of actions, one for each snake, indicating their movement direction.

        Returns:
        - rewards (list): List of rewards for each snake after the step.
        - game_over (list): List indicating whether each snake has reached a game-over state.
        - score (list): List representing the score of each snake.
        - snake_info (list): List containing information about each snake's actions and interactions.
        - total_time_played (list): List of total time played by each snake.
        - Apple_EatenSnakes (list): List representing the number of apples eaten by each snake.
        - DistanceToSnakesList (list): List containing distances to other snakes for each snake.
        - AppleDistanceList (list): List containing distances to apples of restricted colors for each snake.

        Procedure:
        - Increment the frame iteration count.
        - Check for pygame events, such as quitting the game.
        - Initialize various lists to store information about each snake.
        - Execute the movement based on the provided actions.
        - Calculate distances to apples and other snakes for each snake.
        - Check for collisions, apple eating, and other game events for each snake.
        - Update the game state, including snake positions, food positions, and scores.
        - Return information about rewards, game-over status, scores, and various gameplay details.

        Example:
        - If `play_step([Direction.UP, Direction.LEFT])` is called, it executes a game step with the specified
        movements for each snake and returns relevant information.

        Note: This function assumes the existence of various variables and functions such as `pygame`, `Point`,
        `self.num_snakes`, `self.food`, `self.snakes`, `self.SnakeColors`, `self.RestrictedColors`, `self.MAX_FRAMES_INACTIVITY`,
        `self.Belohnung`, `self.TeamSCore`, and `self.calculate_distance` in the code.
        """
        self.frame_iteration += 1

        # Check for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialize lists to store information about each snake
        snake_info = ["none"] * self.num_snakes
        total_time_played = [[] for _ in range(self.num_snakes)]
        rewards = [0] * self.num_snakes

        # Execute the movement based on provided actions
        self._move(actions)

        # Calculate distances to apples and other snakes for each snake
        AppleDistanceList = []
        DistanceToSnakesList = []

        for snake_index in range(self.num_snakes):
            AppleDistance = []
            DistanceToSnakes = []

            for i, food in enumerate(self.food):
                # Add the distance to the restricted apple
                if self.FoodColors[i] in self.RestrictedColors:
                    apple_distance = self.calculate_distance(self.snakes[snake_index][0], self.food[i])
                    AppleDistance.append(apple_distance)

            AppleDistanceList.append(AppleDistance)

            for i, snake in enumerate(self.snakes):
                if i != snake_index:
                    snake_distance = self.calculate_distance(self.snakes[snake_index][0], self.snakes[i][0])
                    DistanceToSnakes.append(snake_distance)

            DistanceToSnakesList.append(DistanceToSnakes)

            # Elapsed time
            elapsed_time = time.time() - self.start_time[snake_index]
            snake_head = self.heads[snake_index]

            # Check collision with the wall
            if self.collsion_wall(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = self.Belohnung["Collidind with wall"][snake_index]
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index] = "I collided with the wall"

            # Check collision with itself
            elif self.collison_with_itself(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = self.Belohnung["Colliding with itself"][snake_index]
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index] = "I collided with myself"

            # Otherwise
            else:
                # Update snake position and check for game events (collisions, apple eating, etc.)
                self.snakes[snake_index].insert(0, snake_head)
                eaten_apple = None

                if snake_head in self.food:
                    # Handle apple eating event
                    self.Apple_EatenSnakes[snake_index][0] += 1

                    # Find out which apple was eaten
                    for i, apple in enumerate(self.food):
                        if snake_head == apple:
                            eaten_apple = i

                            # If we ate a gifted apple
                            if self.FoodColors[i] in self.RestrictedColors:
                                total_time_played[snake_index] = elapsed_time
                                self.game_over[snake_index] = True
                                rewards[snake_index] = self.Belohnung["Eat gifted apple"][snake_index]
                                snake_info[snake_index] = "I ate a gifted apple!"
                                self.GiftedApples[snake_index] += 1
                            else:
                                # If both snake and apple have the same color, else check the other condition
                                if self.SnakeColors[snake_index] == self.FoodColors[i]:
                                    self.score[snake_index] += 1
                                    rewards[snake_index] = self.Belohnung["Eat apple with same Color"][snake_index]
                                    self.AppleSameColor[snake_index] += 1
                                    snake_info[snake_index] = "I ate an apple with the same color as me!"
                                else:
                                    self.score[snake_index] += 1
                                    rewards[snake_index] = self.Belohnung["eat opoent Apple"][snake_index]
                                    self.OponentApples[snake_index] += 1
                                    snake_info[snake_index] = "I ate a normal apple"

                    # Remove the eaten apple and place a new one at a random position
                    self.food.pop(eaten_apple)

                    while True:
                        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

                        new_apple = Point(x, y)
                        if new_apple not in self.food:
                            self.food.insert(eaten_apple, new_apple)
                            break

                    self.frames_since_last_action[snake_index] = 0

                else:
                    # Handle snake movement and check for collisions with other snakes
                    self.snakes[snake_index].pop()
                    self.frames_since_last_action[snake_index] += 1
                    snake_info[snake_index] = "Exploring the environment!"

                    for other_snake_index in range(self.num_snakes):
                        if snake_index != other_snake_index:
                            # Check for collision with other snakes
                            if snake_head in self.snakes[other_snake_index]:
                                # Check if collided with teammate
                                if self.SnakeColors[snake_index] == self.SnakeColors[other_snake_index]:
                                    rewards[snake_index] = self.Belohnung["Collided with Team mate"][snake_index]
                                    snake_info[snake_index] = "Collided with my teammate!"
                                    if self.SnakeColors[snake_index] == "white":
                                        self.TeamSCore[0] -= 1
                                    else:
                                        self.TeamSCore[1] -= 1
                                    self.EatenTeamMate[snake_index] += 1
                                    self.game_over[snake_index] = True
                                else:
                                    # Check if eaten by opponent or eating opponent
                                    if len(self.snakes[snake_index]) <= len(self.snakes[other_snake_index]):
                                        rewards[snake_index] = self.Belohnung["opoent eated me"][snake_index]
                                        snake_info[snake_index] = "Oponent snake ate me!"
                                        self.game_over[snake_index] = True
                                    elif len(self.snakes[snake_index]) > len(self.snakes[other_snake_index]):
                                        rewards[snake_index] = self.Belohnung["eat opoent"][snake_index]
                                        snake_info[snake_index] = "I ate the oponent snake!"
                                        if self.SnakeColors[snake_index] == "white":
                                            self.TeamSCore[0] += 1
                                        else:
                                            self.TeamSCore[1] += 1
                                        self.EatenOponent[snake_index] += 1
                                        self.score[snake_index] += 1

                    total_time_played[snake_index] = elapsed_time

                    # Check if snake has been inactive for too long
                    if self.frames_since_last_action[snake_index] >= self.MAX_FRAMES_INACTIVITY:
                        self.game_over[snake_index] = True
                        rewards[snake_index] = -10
                        self.frames_since_last_action[snake_index] = 0
                        snake_info[snake_index] = "I didn't do anything for n iterations"

        # Return information about the game state after the step
        return rewards, self.game_over, self.score, snake_info, total_time_played, self.Apple_EatenSnakes, DistanceToSnakesList, AppleDistanceList


    def is_collision(self, snake_index, pt=None):
        """
        Check if a collision has occurred for the specified snake.

        Parameters:
        - snake_index (int): Index of the snake to check for collision.
        - pt (Point, optional): Point to check for collision. If not provided, the head of the snake is used.

        Returns:
        - collision (bool): True if a collision has occurred, False otherwise.

        Procedure:
        - If no point is provided, use the head of the specified snake.
        - Check if the point hits the boundary of the game window.
        - Check if the point hits the body of the snake itself.
        - Return True if any collision condition is met, otherwise return False.

        Example:
        - If `is_collision(0)` is called, it checks for collision for the first snake, using its head.

        Note: This function assumes the existence of various variables such as `pt`, `self.heads`, `self.w`, `self.h`,
        `BLOCK_SIZE`, `self.snakes`, and `Point` in the code.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Check if the point hits the body of the snake itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # No collision
        return False

    
    def collsion_wall(self, snake_index, pt=None):
        """
        Check if the specified snake has collided with the boundaries of the game window.

        Parameters:
        - snake_index (int): Index of the snake to check for collision.
        - pt (Point, optional): Point to check for collision. If not provided, the head of the snake is used.

        Returns:
        - collision (bool): True if a collision with the wall has occurred, False otherwise.

        Procedure:
        - If no point is provided, use the head of the specified snake.
        - Check if the point hits the boundary of the game window.
        - Return True if a collision with the wall occurs, otherwise return False.

        Example:
        - If `collsion_wall(0)` is called, it checks for collision with the wall for the first snake, using its head.

        Note: This function assumes the existence of various variables such as `pt`, `self.heads`, `self.w`, `self.h`,
        `BLOCK_SIZE`, and `Point` in the code.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # No collision with the wall
        return False
    
    def collison_with_itself(self, snake_index, pt=None):
        """
        Check if the specified snake has collided with its own body.

        Parameters:
        - snake_index (int): Index of the snake to check for self-collision.
        - pt (Point, optional): Point to check for collision. If not provided, the head of the snake is used.

        Returns:
        - collision (bool): True if self-collision has occurred, False otherwise.

        Procedure:
        - If no point is provided, use the head of the specified snake.
        - Check if the point hits any segment of the snake's body except the head.
        - Return True if self-collision occurs, otherwise return False.

        Example:
        - If `collison_with_itself(0)` is called, it checks for self-collision for the first snake, using its head.

        Note: This function assumes the existence of various variables such as `pt`, `self.heads`, `self.snakes`,
        and `Point` in the code.
        """
        if pt is None:
            pt = self.heads[snake_index]

        # Check if the point hits any segment of the snake's body except the head
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # No self-collision
        return False




    def eate_other_snake(self, snake_index):
        """
        Check if the specified snake has eaten any other snake.

        Parameters:
        - snake_index (int): Index of the snake to check for eating other snakes.

        Returns:
        - Tuple (bool, list): First element is True if the snake has eaten another snake, False otherwise.
                            Second element is a list containing the indices of the collided snakes.

        Procedure:
        - Retrieve the head of the specified snake.
        - Initialize variables: 'truth' to track if eating has occurred, and 'collided' to store indices of collided snakes.
        - Iterate through all snakes, excluding the specified snake.
        - If the head of the specified snake is in another snake's body, check and record if the specified snake has a greater length.
        - Return a tuple containing 'truth' and 'collided' list.

        Example:
        - If `eate_other_snake(0)` is called, it checks if the first snake has eaten any other snake.

        Note: This function assumes the existence of various variables such as `self.heads`, `self.snakes`, and `len` in the code.
        """
        head = self.heads[snake_index]
        truth = False
        collided = []

        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                # Get the index of the collided snake
                collided.append(i)
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])

                # Check if the specified snake has a greater length than the collided snake
                if current_snake_length > collided_snake_length:
                    truth = True

        return truth, collided

    
    def eaten_by_other_snake(self, snake_index):
        """
        Check if the specified snake has been eaten by any other snake.

        Parameters:
        - snake_index (int): Index of the snake to check for being eaten.

        Returns:
        - bool: True if the snake has been eaten by another snake, False otherwise.

        Procedure:
        - Retrieve the head of the specified snake.
        - Iterate through all snakes, excluding the specified snake.
        - If the head of the specified snake is in another snake's body, check if the specified snake has a smaller or equal length.
        - Return True if the specified snake has been eaten by another snake, otherwise return False.

        Example:
        - If `eaten_by_other_snake(0)` is called, it checks if the first snake has been eaten by any other snake.

        Note: This function assumes the existence of various variables such as `self.heads`, `self.snakes`, and `len` in the code.
        """
        head = self.heads[snake_index]

        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])

                # Check if the specified snake has a smaller or equal length than the collided snake
                if current_snake_length <= collided_snake_length:
                    return True

        # No collision or being eaten by other snakes
        return False

    

    def grid(self):
        """
        Draw a grid on the game window for visual reference.

        Procedure:
        - Iterate through rows and columns in the game window with a step size of BLOCK_SIZE.
        - For each row and column, draw a rectangle (grid cell) with a border using Pygame.
        - Update the game display to render the grid.

        Example:
        - If `grid()` is called, it will draw a grid on the game window.

        Note: This function assumes the existence of various variables such as `self.h`, `BLOCK_SIZE`, `self.display`, and `pygame` in the code.
        """
        for row in range(0, self.h, BLOCK_SIZE):
            for col in range(0, self.h, BLOCK_SIZE):
                # Draw a rectangle (grid cell) with a border
                rect = pygame.Rect(row, col, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, "green", rect, 3)

        # Update the game display to render the grid
        pygame.display.update()



    def _update_ui(self):
        """
        Update the game user interface (UI) for visual representation.

        Procedure:
        - Fill the display with a black background and draw the scaled background image.
        - Draw grid lines on the game window for visual reference.
        - Iterate through each snake and its body segments, drawing them on the display based on color and movement.
        - Iterate through each food item, drawing them on the display based on color.
        - Display the score of each snake on the top-left corner of the window.
        - Update the game display and control the frame rate with Pygame's clock.

        Note: This function assumes the existence of various variables such as `self.w`, `self.h`, `BLOCK_SIZE`, `self.display`, `self.clock`, `FPS`, `pygame`, and others in the code.
        """
        # Fill the display with a black background and draw the scaled background image
        self.display.fill((0, 0, 0))
        self.display.blit(self.background_image, (0, 0))

        # Draw grid lines on the game window for visual reference
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, "green", (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, "green", (0, y), (self.w, y), 1)

        # Iterate through each snake and its body segments, drawing them on the display based on color and movement
        for snake_index in range(self.num_snakes):
            for point_index, point in enumerate(self.snakes[snake_index]):

                # Check the current color used
                if self.SnakeColors[snake_index] == "red":
                    if point_index == 0:  # Snake head
                        # Draw the snake head based on movement direction
                        if self.snakesMovment[snake_index] == 1:
                            self.display.blit(self.rightred, (point.x, point.y))
                        elif self.snakesMovment[snake_index] == 2:
                            self.display.blit(self.leftred, (point.x, point.y))
                        elif self.snakesMovment[snake_index] == 3:
                            self.display.blit(self.upred, (point.x, point.y))
                        else:
                            self.display.blit(self.downred, (point.x, point.y))
                    else:  # Snake body segment
                        pygame.draw.circle(
                            self.display,
                            self.SnakeColors[snake_index],
                            (point.x + BLOCK_SIZE // 2, point.y + BLOCK_SIZE // 2),
                            BLOCK_SIZE // 2
                        )
                else:
                    if point_index == 0:  # Snake head
                        # Draw the snake head based on movement direction
                        if self.snakesMovment[snake_index] == 1:
                            self.display.blit(self.rightblue, (point.x, point.y))
                        elif self.snakesMovment[snake_index] == 2:
                            self.display.blit(self.leftblue, (point.x, point.y))
                        elif self.snakesMovment[snake_index] == 3:
                            self.display.blit(self.upblue, (point.x, point.y))
                        else:
                            self.display.blit(self.downblue, (point.x, point.y))
                    else:  # Snake body segment
                        pygame.draw.circle(
                            self.display,
                            self.SnakeColors[snake_index],
                            (point.x + BLOCK_SIZE // 2, point.y + BLOCK_SIZE // 2),
                            BLOCK_SIZE // 2
                        )

            # Iterate through each food item, drawing them on the display based on color
            for i, food in enumerate(self.food):
                # Choose food colors
                if self.FoodColors[i] == "red":
                    self.display.blit(self.redapple, (food.x, food.y))
                elif self.FoodColors[i] == "white":
                    self.display.blit(self.whiteapple, (food.x, food.y))
                elif self.FoodColors[i] == "blue":
                    self.display.blit(self.blueapple, (food.x, food.y))
                else:
                    self.display.blit(self.blackapple, (food.x, food.y))

            # Display the score of each snake on the top-left corner of the window
            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render("Score: " + str(self.score[snake_index]), True, (255, 255, 255))
            self.display.blit(score_text, (10, 10 + 40 * snake_index))

        # Update the game display and control the frame rate with Pygame's clock
        pygame.display.flip()
        self.clock.tick(FPS)






    def handle_user_input(self):
        """
        Handle user input for controlling the snake's direction.

        Procedure:
        - Get the current state of keys pressed using Pygame's key.get_pressed().
        - If the UP arrow key is pressed, return a corresponding action for a left turn.
        - If the DOWN arrow key is pressed, return a corresponding action for a right turn.
        - If no arrow keys are pressed, return a default action for no change in direction.

        Returns:
            List[int]: A list representing the action to be taken by the human agent.
                    [0, 0, 1] for a left turn, [0, 1, 0] for a right turn, and [1, 0, 0] for no change.
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return [0, 0, 1]  # Left turn action
        elif keys[pygame.K_DOWN]:
            return [0, 1, 0]  # Right turn action
        else:
            return [1, 0, 0]  # No change action



    def _move(self, actions):
        """
        Move each snake based on the specified actions.

        Args:
            actions (List[List[int]]): List of actions for each snake.
                                    Each action is represented as a list of three integers.
                                    [1, 0, 0] for no change, [0, 1, 0] for a right turn,
                                    and [0, 0, 1] for a left turn.

        Procedure:
        - Define the clockwise order of directions (right, down, left, up).
        - Iterate over each snake and update its direction based on the specified action.
        - Calculate the new head position based on the updated direction and move the snake.

        Returns:
            None
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        for snake_index in range(self.num_snakes):

            # Get the current direction index
            idx = clock_wise.index(self.directions[snake_index])

            if np.array_equal(actions[snake_index], [1, 0, 0]):
                # No change in direction
                new_dir = clock_wise[idx]
            elif np.array_equal(actions[snake_index], [0, 1, 0]):
                # Right turn
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            else:
                # Left turn
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]

            # Update the snake's direction
            self.directions[snake_index] = new_dir

            # Get current head position
            x = self.heads[snake_index].x
            y = self.heads[snake_index].y

            # Update head position based on the current direction
            if self.directions[snake_index] == Direction.RIGHT:
                self.snakesMovment[snake_index] = Direction.RIGHT.value
                x += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.LEFT:
                self.snakesMovment[snake_index] = Direction.LEFT.value
                x -= BLOCK_SIZE
            elif self.directions[snake_index] == Direction.DOWN:
                self.snakesMovment[snake_index] = Direction.DOWN.value
                y += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.UP:
                self.snakesMovment[snake_index] = Direction.UP.value
                y -= BLOCK_SIZE

            # Update the head position
            self.heads[snake_index] = Point(x, y)



def Create_agent(input_dim, dim1, dim2, n_actions, lr, batch_size, mem_size, gamma):
    """
    Create and initialize an instance of the Agent class with the specified parameters.

    Args:
        input_dim (int): Number of input dimensions for the agent's neural network.
        dim1 (int): Number of neurons in the first hidden layer.
        dim2 (int): Number of neurons in the second hidden layer.
        n_actions (int): Number of possible actions the agent can take.
        lr (float): Learning rate for the agent's neural network.
        batch_size (int): Batch size for experience replay.
        mem_size (int): Size of the memory buffer for experience replay.
        gamma (float): Discount factor for future rewards.

    Returns:
        Agent: An instance of the Agent class with the specified configuration.
    """
    # Instantiate and return an Agent object with the specified parameters
    return Agent(input_dim, dim1, dim2, n_actions, lr, batch_size, mem_size, gamma)


def plot(scores, mean_scores):
    """
    Plot the scores and mean scores during training.

    Args:
        scores (list): List of scores achieved in each training episode.
        mean_scores (list): List of mean scores over a moving window during training.
    """
    # Clear the current figure
    plt.clf()

    # Set plot title and labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot the scores and mean_scores
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')

    # Set y-axis lower limit to 0
    plt.ylim(ymin=0)

    # Display the last score values as annotations on the plot
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # Display the legend
    plt.legend()

    # Show the plot without blocking and pause for a short time
    plt.show(block=False)
    plt.pause(0.1)
if __name__ == '__main__':

        # Set the number of snakes in the game
    number_of_snakes = 4

    # Initialize variables to track the maximum score achieved
    current_max = 0

    # Create an instance of the SnakeGame with the specified number of snakes
    game = SnakeGame(num_snakes=number_of_snakes)

    # Initialize an agent using the Agent class with specific configurations for each snake
    agent = Agent(
        input_dimlsit=[72, 72, 72, 72],
        fc1_dimlsit=[400, 300, 200, 250],
        fc2_dimlist=[512, 512, 512, 512],
        fc3_dimlist=[256, 256, 256, 256],
        fc4_dimlist=[256, 256, 256, 256],
        n_actions=3,
        losslist=[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
        lrlist=[0.004, 0.002, 0.001, 0.003],
        batch_size=[10, 10, 10, 10, 10],
        mem_size=[10000, 10000, 10000, 10000],
        gamma_list=[0.30, 0.40, 0.30, 0.40],
        num_agents=4
    )

    # Initialize variables to track game state and performance metrics for each snake
    running = True
    step = [0] * game.num_snakes
    TotalPlayerScorePro = [0] * game.num_snakes
    Total_PlayedTime = [0] * game.num_snakes
    TotalTimeBeforeDeath = [0] * game.num_snakes
    TotalSnakeEaten = [0] * game.num_snakes
    TotalAppleEaten = [0] * game.num_snakes
    TotalEatenSnakes = [[] for _ in range(game.num_snakes)]
    TotalEatenApples = [[] for _ in range(game.num_snakes)]
    Total_score_list = [[] for _ in range(game.num_snakes)]
    Total_Time_List = [[] for _ in range(game.num_snakes)]
    DataFrames = []
    BestPerformance = [0] * game.num_snakes

    # Loop through each agent to initialize data structures for logging performance metrics
    for agent_idx in range(number_of_snakes):
        data = {
            f'n_games{agent_idx}': [],
            f'playerScoreProRound{agent_idx}': [],
            f'playerTotalScore{agent_idx}': [],
            f'TimePlayedPRoRound{agent_idx}': [],
            f'TotalTimePlayed{agent_idx}': [],
            f'MeanScore{agent_idx}': [],
            f'TimeOverScore{agent_idx}': [],
            f'TotalNumberofDeath{agent_idx}': [],
            f'TotalTimeSpendOverTotalTimeOfDeath{agent_idx}': [],
            f'Epsilon{agent_idx}': [],
            f'GifftedApples{agent_idx}': [],
            f'StealedOponentAppels{agent_idx}': [],
            f'ApplesWithSameColorASmine{agent_idx}': [],
            f'EatenOponent{agent_idx}': [],
            f'EatenTeamMates{agent_idx}': [],
            f'TeamScore{agent_idx}': [],
            f'CurrentState{agent_idx}': [],
            f'Distancetoobstacle{agent_idx}': [],
        }
        DataFrames.append(data)


    i = 0 
 
                
    while i < 20000 :

        
      # Get the current state of the game
        old_states = game.get_state()

        # Choose actions for each snake using the agent
        actions = agent.choose_action(old_states)

        # Play a step in the game with the chosen actions
        rewards, game_over, scores, info, time_played, apple_snake, DistanceToSnakesList, DistanceToAppleList = game.play_step(actions)

        # Update the game UI
        game._update_ui()

        # Control the game speed
        game.clock.tick(SPEED)

        # Get the new state of the game after playing a step
        states_new = game.get_state()

        # Store the experience in the agent's short-term memory
        agent.short_mem(old_states, states_new, actions, rewards, game_over)

        # Save a screenshot of the game display
        screenshot_filename = f"/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/Coding/Env2/5Snakes/screenshot{i}.png"
        if i >= 0:
            pygame.image.save(game.display, screenshot_filename)

        # Check if any snake  is Dead 
        if any(game_over) and not all(game_over):
            # Get indices of snakes that have collided
            indices = [index for index, value in enumerate(game_over) if value == True]

            # Process data and update performance metrics for each collided snake
            for index in indices:
                step[index] += 1  # Number of rounds will be incremented
                TotalPlayerScorePro[index] = TotalPlayerScorePro[index] + scores[index]
                Total_PlayedTime[index] = time_played[index] + Total_PlayedTime[index]
                TotalSnakeEaten[index] = TotalSnakeEaten[index] + apple_snake[index][1]
                TotalAppleEaten[index] = TotalAppleEaten[index] + apple_snake[index][0]

                # Append data to the respective metric lists in DataFrames
                DataFrames[index][f'TeamScore{index}'].append(game.TeamSCore[game.SnakeColors.index(game.SnakeColors[index])])
                DataFrames[index][f'GifftedApples{index}'].append(game.GiftedApples[index])
                DataFrames[index][f'StealedOponentAppels{index}'].append(game.OponentApples[index])
                DataFrames[index][f'ApplesWithSameColorASmine{index}'].append(game.AppleSameColor[index])
                DataFrames[index][f'EatenOponent{index}'].append(game.EatenOponent[index])
                DataFrames[index][f'EatenTeamMates{index}'].append(game.EatenTeamMate[index])

                DataFrames[index][f'n_games{index}'].append(step[index])
                DataFrames[index][f'CurrentState{index}'].append(info[index])
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index] / step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])

                # Avoid division by zero if TotalPlayerScorePro[index] is zero
                if TotalPlayerScorePro[index] > 0:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index] / TotalPlayerScorePro[index])
                else:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])

                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(Total_PlayedTime[index] / step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'Distancetoobstacle{index}'].append(DistanceToAppleList[index])

                # Reset the snake for the next round, update the UI, and control the game speed
                game.reset_snake(index)
                game._update_ui()
                game.clock.tick(SPEED)

                # Store the experience in the agent's long-term memory
                agent.long_memory(index)



            '''   
            for  agent_index  in range(game.num_snakes) :
               if   BestPerformance[agent_index] < scores[agent_index] : 
                   BestPerformance[agent_index] = scores[agent_index] 
                   agent.save(agent_index)
            '''
       # Check if all snakes are Dead
        elif all(game_over):
            # Get indices of snakes that have collided
            indices = [index for index, value in enumerate(game_over) if value == True]
            
            # Process data and update performance metrics for each collided snake
            for index in indices:
                step[index] += 1  # Number of rounds will be incremented
                TotalPlayerScorePro[index] += scores[index]
                Total_PlayedTime[index] += time_played[index]
                TotalSnakeEaten[index] += apple_snake[index][1]
                TotalAppleEaten[index] += apple_snake[index][0]

                # Append data to the respective metric lists in DataFrames
                DataFrames[index][f'TeamScore{index}'].append(game.TeamSCore[game.SnakeColors.index(game.SnakeColors[index])])
                DataFrames[index][f'GifftedApples{index}'].append(game.GiftedApples[index])
                DataFrames[index][f'StealedOponentAppels{index}'].append(game.OponentApples[index])
                DataFrames[index][f'ApplesWithSameColorASmine{index}'].append(game.AppleSameColor[index])
                DataFrames[index][f'EatenOponent{index}'].append(game.EatenOponent[index])
                DataFrames[index][f'EatenTeamMates{index}'].append(game.EatenTeamMate[index])

                DataFrames[index][f'n_games{index}'].append(step[index])
                DataFrames[index][f'CurrentState{index}'].append(info[index])
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index] / step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'TimePlayedPRoRound{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalTimePlayed{index}'].append(Total_PlayedTime[index])

                # Avoid division by zero if TotalPlayerScorePro[index] is zero
                if TotalPlayerScorePro[index] > 0:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index] / TotalPlayerScorePro[index])
                else:
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])

                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TotalTimeSpendOverTotalTimeOfDeath{index}'].append(Total_PlayedTime[index] / step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'Distancetoobstacle{index}'].append(DistanceToAppleList[index])

                
             
            '''
            for  agent_index  in range(game.num_snakes) :
               if  BestPerformance[agent_index] < scores[agent_index] : 
                   BestPerformance[agent_index] = scores[agent_index] 
                   agent.save(agent_index)

                if  i %  500  : 
                    agent.save(agent_index ,game.snake , i )
            '''

           # Reset the game state, update the user interface, and tick the clock
            game.reset()
            game._update_ui()
            game.clock.tick(SPEED)

            # Update the long-term memory of the agent
            agent.long_mem()


         
        #save the mdoell each 500 iterations  
        for  agent_index  in range(game.num_snakes) :
                if i ==  9500  : 
                    if  i %  100== 0   : 
                        agent.save(agent_index , game.SnakeColors[agent_index]  , i )
        print(f"step : {i}")
        i+=1 



        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
       
    pygame.quit()

# Specify the directory to save CSV files
save_path = '/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/Coding/Env2/ExcelFiles'

# Iterate over each snake's DataFrame and save it as a CSV file
for dataFrame_index in range(game.num_snakes):
    # Create a DataFrame from the collected data
    df = pd.DataFrame(DataFrames[dataFrame_index])

    # Define the full file path
    file_path = os.path.join(save_path, f'Snake{dataFrame_index}.csv')

    # Save the DataFrame to a CSV file, append if the file exists
    df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))



