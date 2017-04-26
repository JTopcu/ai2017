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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        # The score we're going to return
        score = 0.0

        # If the move results in being adjacent to a ghost, don't move there
        for agent in range(successorGameState.getNumAgents()):
            if agent > 0 and util.manhattanDistance(newPos, successorGameState.getGhostPosition(agent)) <= 1:
                return 0

        # Find the reciprical of all food, add the largest number (closest food) to the score
        for food in newFood.asList():
            foodDistance = 1.0 / util.manhattanDistance(newPos, food)
            if foodDistance > score:
                score = foodDistance

        # If the new state is on food, add 1 to the score
        if currentGameState.getFood()[newPos[0]][newPos[1]]:
            score += 1.0

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

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
        """
        # Constant to identify the pacman agent
        PACMAN = 0

        # Recursive function used to navigate tree
        def minimax(gameState, depth, agent):
            # Return the value of the evaluation function if the state is a terminal state
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # If the agent is pacman, return the max value of all possible moves
            if agent == PACMAN:
                bestVal = -(float("inf"))
                legalMoves = gameState.getLegalActions(agent)

                for action in legalMoves:
                    val = minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                    bestVal = max(bestVal, val)

                return bestVal
            # If the agent is a ghost, return the min value of all possible moves
            else:
                bestVal = float("inf")
                legalMoves = gameState.getLegalActions(agent)

                # If the agent isn't the last ghost, find the min and call the function on the next agent
                if agent < gameState.getNumAgents() - 1:
                    for action in legalMoves:
                        val = minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                        bestVal = min(bestVal, val)
                # Once we reach the last agent (the last ghost), call the function on pacman and decrement
                # the depth as we have cycled through every agent
                else:
                    for action in legalMoves:
                        val = minimax(gameState.generateSuccessor(agent, action), depth - 1, PACMAN)
                        bestVal = min(bestVal, val)

                return bestVal

        # Set required variables to use in recursive minimax function
        legalMoves = gameState.getLegalActions(PACMAN)
        bestVal = -(float("inf"))
        actionToReturn = ""

        # Iterate through all of pacman's initial legal moves, set actionToReturn to the action corresponding
        # with the highest value of the minimax function and return it
        for action in legalMoves:
            lastVal = bestVal
            bestVal = max(bestVal, minimax(gameState.generateSuccessor(0, action), self.depth, 1))
            if bestVal > lastVal:
                actionToReturn = action

        return actionToReturn

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Constant to identify the pacman agent
        PACMAN = 0

        # Recursive function used to navigate tree
        def alphabeta(gameState, depth, alpha, beta, agent):
            # Return the value of the evaluation function if the state is a terminal state
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # If the agent is pacman, look for the max of all legal moves
            if agent == PACMAN:
                val = -(float("inf"))
                legalMoves = gameState.getLegalActions(agent)

                for action in legalMoves:
                    val = max(val, alphabeta(gameState.generateSuccessor(agent, action), depth, alpha, beta, agent + 1))
                    # If the max value so far is greater than beta, there is no need to check any more states as the
                    # min agent (ghosts) will pick the beta value regardless
                    if val > beta:
                        break
                    # Set alpha to the new highest value if it's higher than the current value
                    alpha = max(alpha, val)

                return val
            # If the agent is a gohst, look for the min of all legal moves
            else:
                val = float("inf")
                legalMoves = gameState.getLegalActions(agent)

                # Cycle through the ghosts and decrement the depth once we reach the final ghost
                if agent < gameState.getNumAgents() - 1:
                    for action in legalMoves:
                        val = min(val, alphabeta(gameState.generateSuccessor(agent, action), depth, alpha, beta, agent + 1))
                        # If the min value so far is less than alpha, there is no need to check any more states as the
                        # max agent (pacman) will pick the alpha value regardless
                        if val < alpha:
                            break
                        beta = min(beta, val)
                else:
                    for action in legalMoves:
                        val = min(val, alphabeta(gameState.generateSuccessor(agent, action), depth - 1, alpha, beta, PACMAN))
                        # If the min value so far is less than alpha, there is no need to check any more states as the
                        # max agent (pacman) will pick the alpha value regardless
                        if val < alpha:
                            break
                        beta = min(beta, val)

                return val

        # Set required variables to use in recursive alphabeta function
        legalMoves = gameState.getLegalActions(PACMAN)
        alpha = -(float("inf"))
        beta = float("inf")
        actionToReturn = ""
        val = alpha

        # Iterate through all of pacman's initial legal moves, set actionToReturn to the action corresponding
        # with the highest value of the alphabeta function and return it
        for action in legalMoves:
            lastVal = val
            val = max(val, alphabeta(gameState.generateSuccessor(0, action), self.depth, alpha, beta, 1))
            if val > lastVal:
                actionToReturn = action

            alpha = max(alpha, val)

        return actionToReturn

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
        # Constant to identify the pacman agent
        PACMAN = 0

        # Recursive function used to navigate tree
        def expectimax(gameState, depth, agent):
            # Return the value of the evaluation function if the state is a terminal state
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # If the agent is pacman, return the highest value of the expectimax function in the same
            # manner as the minimax function
            if agent == PACMAN:
                bestVal = -(float("inf"))
                legalMoves = gameState.getLegalActions(agent)

                for action in legalMoves:
                    val = expectimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                    bestVal = max(bestVal, val)

                return bestVal
            # If the agent is a ghost, return the average of all the values of the expectimax function
            # calls, which is one for every legal move
            else:
                val = 0
                legalMoves = gameState.getLegalActions(agent)

                if agent < gameState.getNumAgents() - 1:
                    for action in legalMoves:
                        val += expectimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                else:
                    for action in legalMoves:
                        val += expectimax(gameState.generateSuccessor(agent, action), depth - 1, PACMAN)

                return val / len(legalMoves)

        # Set required variables to use in recursive alphabeta function
        legalMoves = gameState.getLegalActions(PACMAN)
        bestVal = -(float("inf"))
        actionToReturn = ""

        # Iterate through all of pacman's initial legal moves, set actionToReturn to the action corresponding
        # with the highest value of the expectimax function and return it
        for action in legalMoves:
            lastVal = bestVal
            bestVal = max(bestVal, expectimax(gameState.generateSuccessor(0, action), self.depth, 1))
            if bestVal > lastVal:
                actionToReturn = action

        return actionToReturn

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This function will take the current score of the game to use
      as the base score for pacman. There are two features that will affect the
      final score of the evaluation function:

      1. It will calculate the sum of the reciprical of the manhattan distance
      from pacman to each ghost and subtract them from the score. The closer the ghosts
      are to pacman, the bigger the subtraction will be.

      2. It will find the closest food, take the reciprical and add it to the current
      score. The closer the food the bigger the score will be.
    """
    # Pacman's current position
    pos = currentGameState.getPacmanPosition()

    # The score we're going to return
    score = currentGameState.getScore()

    # Find the sum of the reciprical of all ghost distances to pacman
    ghostDistances = 0.0
    for agent in range(currentGameState.getNumAgents() - 1):
        if agent > 0:
            ghostDistance = 1.0 / util.manhattanDistance(pos, currentGameState.getGhostPosition(agent))
            ghostDistances += ghostDistance

    # Subtract the distances from the current score
    score -= ghostDistances


    # Find the reciprical of all food, add the largest number (closest food) to the score
    closestFood = 0.0
    for food in currentGameState.getFood().asList():
        foodDistance = 1.0 / util.manhattanDistance(pos, food)
        if foodDistance > closestFood:
            closestFood = foodDistance

    score += closestFood

    return score

# Abbreviation
better = betterEvaluationFunction

