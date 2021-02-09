# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Start-> python pacman.py -p ExpectimaxAgent -a depth=2 -l smallClassic
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total_score = successorGameState.getScore()
        px, py = newPos

        num_of_remained_foods = 0
        distance_with_food = float('inf')
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    num_of_remained_foods += 1
                    temp = util.manhattanDistance((i, j), (px, py))
                    if temp < distance_with_food:
                        distance_with_food = temp
        if distance_with_food == float('inf'):
            distance_with_food = -10
        total_score += 1 / (num_of_remained_foods + 1e-5)
        total_score -= distance_with_food / 2

        for i in range(len(newScaredTimes)):
            gx, gy = newGhostStates[i].configuration.pos
            distance_with_ghost = util.manhattanDistance((gx, gy), (px, py))
            if newScaredTimes[i] > 0 and distance_with_ghost < 1:
                total_score += 100

            if 40 > newScaredTimes[i] > 0:
                total_score += 100

            if distance_with_ghost <= 2 and newScaredTimes[i] <= 0:
                total_score -= 100

        return total_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    # evalFn換成這個會更強-> betterEvaluationFunction
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function, expectimax=False):
    if agentIndex >= num_agent:
        agentIndex -= num_agent
        depth += 1
    if depth == max_depth or gameState.isWin() or gameState.isLose():
        return evaluation_function(gameState)
    if agentIndex == 0:
        return max_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function)
    if agentIndex > 0:
        if expectimax:
            return exp_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function)
        else:
            return min_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function)


def max_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    v = float('-inf')
    for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        v = max(v, value(successor, agentIndex + 1, num_agent, depth, max_depth, evaluation_function))

    return v


def min_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    v = float('inf')
    for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        v = min(v, value(successor, agentIndex + 1, num_agent, depth, max_depth, evaluation_function))

    return v


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        v = float('-inf')
        final_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp = value(successor, 1, successor.getNumAgents(), 0, self.depth, self.evaluationFunction)
            if temp > v:
                v = temp
                final_action = action
        return final_action


def value_prune(alpha, beta, gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    if agentIndex >= num_agent:
        agentIndex -= num_agent
        depth += 1
    if depth == max_depth or gameState.isWin() or gameState.isLose():
        return evaluation_function(gameState)
    if agentIndex == 0:
        return max_value_prune(alpha, beta, gameState, agentIndex, num_agent, depth, max_depth, evaluation_function)
    if agentIndex > 0:
        return min_value_prune(alpha, beta, gameState, agentIndex, num_agent, depth, max_depth, evaluation_function)


def max_value_prune(alpha, beta, gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    v = float('-inf')
    for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        v = max(v, value_prune(alpha, beta, successor, agentIndex + 1, num_agent, depth, max_depth, evaluation_function))
        if v > beta:
            return v
        alpha = max(v, alpha)

    return v


def min_value_prune(alpha, beta, gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    v = float('inf')
    for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        v = min(v, value_prune(alpha, beta, successor, agentIndex + 1, num_agent, depth, max_depth, evaluation_function))
        if v < alpha:
            return v
        beta = min(v, beta)

    return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        v = float('-inf')
        final_action = None
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp = value_prune(alpha, beta, successor, 1, successor.getNumAgents(), 0, self.depth, self.evaluationFunction)
            if temp > v:
                v = temp
                final_action = action
            if v > beta:
                break
            alpha = max(alpha, v)
        return final_action


def exp_value(gameState, agentIndex, num_agent, depth, max_depth, evaluation_function):
    v = 0
    legal_actions = gameState.getLegalActions(agentIndex)
    for action in legal_actions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v += value(successor, agentIndex + 1, num_agent, depth, max_depth, evaluation_function, True) / len(legal_actions)

    return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        v = float('-inf')
        final_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp = value(successor, 1, successor.getNumAgents(), 0, self.depth, self.evaluationFunction, True)
            if temp > v:
                v = temp
                final_action = action
        return final_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    total_score = currentGameState.getScore()
    pacman_position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    foods_num = currentGameState.getNumFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    remained_foods = foods.asList()
    if remained_foods:
        distance_with_food = float('inf')
        for food in remained_foods:
            temp = util.manhattanDistance(food, pacman_position)
            if temp < distance_with_food:
                distance_with_food = temp
    else:
        distance_with_food = 0

    total_score += 100 / (foods_num + 1e-5)
    total_score += 1 / (distance_with_food + 1e-5)

    for i in range(len(scared_times)):
        distance_with_ghost = util.manhattanDistance(ghost_states[i].configuration.pos, pacman_position)
        if scared_times[i] > 0:
            total_score += 10 / (distance_with_ghost + 1e-5)
        else:
            total_score -= 1 / (distance_with_ghost + 1e-5)
    return total_score


# Abbreviation
better = betterEvaluationFunction
