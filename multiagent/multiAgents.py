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
        for ghost in newGhostStates:
            if util.manhattanDistance(newPos, successorGameState.getGhostPosition(1)) <= 1:
                return 0

        # Find the reciprical of all food, add the largest number (closest food) to the score
        for food in newFood.asList():
            foodDistance = float(1.0 / util.manhattanDistance(newPos, food))
            if foodDistance > score:
                score = foodDistance

        # If the new state is on food, add 1 to the score
        if currentGameState.getFood()[newPos[0]][newPos[1]]:
            score += 1

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
        PACMAN = 0

        def minimax(gameState, depth, agent):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            if agent == PACMAN:
                bestVal = -(float("inf"))
                legalMoves = gameState.getLegalActions(agent)

                for action in legalMoves:
                    val = minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                    bestVal = max(bestVal, val)

                return bestVal
            else:
                bestVal = float("inf")
                legalMoves = gameState.getLegalActions(agent)

                if agent < gameState.getNumAgents() - 1:
                    for action in legalMoves:
                        val = minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                        bestVal = min(bestVal, val)
                else:
                    for action in legalMoves:
                        val = minimax(gameState.generateSuccessor(agent, action), depth - 1, PACMAN)
                        bestVal = min(bestVal, val)

                return bestVal

        legalMoves = gameState.getLegalActions(PACMAN)
        bestVal = -(float("inf"))
        bestAction = ""

        for action in legalMoves:
            lastVal = bestVal
            bestVal = max(bestVal, minimax(gameState.generateSuccessor(0, action), self.depth, 1))
            if bestVal > lastVal:
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        PACMAN = 0

        def alphabeta(gameState, depth, alpha, beta, agent):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            if agent == PACMAN:
                val = -(float("inf"))
                legalMoves = gameState.getLegalActions(agent)

                for action in legalMoves:
                    val = max(val, alphabeta(gameState.generateSuccessor(agent, action), depth, alpha, beta, agent + 1))
                    if val > beta:
                        break
                    alpha = max(alpha, val)

                return val
            else:
                val = float("inf")
                legalMoves = gameState.getLegalActions(agent)

                if agent < gameState.getNumAgents() - 1:
                    for action in legalMoves:
                        val = min(val, alphabeta(gameState.generateSuccessor(agent, action), depth, alpha, beta, agent + 1))
                        if val < alpha:
                            break
                        beta = min(beta, val)
                else:
                    for action in legalMoves:
                        val = min(val, alphabeta(gameState.generateSuccessor(agent, action), depth - 1, alpha, beta, PACMAN))
                        if val < alpha:
                            break
                        beta = min(beta, val)

                return val

        legalMoves = gameState.getLegalActions(PACMAN)
        val = -(float("inf"))
        bestAction = ""

        for action in legalMoves:
            lastVal = val
            val = max(val, alphabeta(gameState.generateSuccessor(0, action), self.depth, val, abs(val), 1))
            if val > lastVal:
                bestAction = action

        return bestAction

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

