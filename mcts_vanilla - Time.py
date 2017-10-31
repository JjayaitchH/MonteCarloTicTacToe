
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log
from timeit import default_timer as time

num_nodes = 1000
explore_faction = 2.

def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.

    Returns:        A node from which the next stage of the search can proceed.

    """
    explore = lambda w, n, t, c: (w / n) + (c * sqrt((log(t) / n)))
    bestest = 0
    values = {}
    new_node = node

    if not node.untried_actions:
        for child in node.child_nodes:
            values[child] = explore(node.child_nodes[child].wins, node.child_nodes[child].visits, node.visits, explore_faction)

            if values[child] > bestest:
                bestest = values[child]
                new_node = node.child_nodes[child]

        if bestest != 0:
            new_node = traverse_nodes(new_node,board,state,identity)
    return new_node

def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    if not board.is_ended(state):
        action = choice(node.untried_actions)
        state = board.next_state(state, action)
        actions = board.legal_actions(state)
        node.child_nodes[action] = MCTSNode(node, action, actions)
        node.untried_actions.remove(action)
        return node.child_nodes[action], state
    else:
        return node, state

def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """
    while not board.is_ended(state):
        action = choice(board.legal_actions(state))
        state = board.next_state(state, action)
    return state


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if won:
        node.wins += 1
    node.visits += 1

    if node.parent is not None:
        backpropagate(node.parent, won)


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """

    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))
    sampled_game = None
    start = time()
    time_elapsed = time() - start
    while time_elapsed <= 1:
        # Copy the game for sampling a playthrough
        sampled_game = state

        # Start at root
        node = root_node

        # Do MCTS - This is all you!
        leaf = traverse_nodes(node, board, sampled_game, identity_of_bot)

        if leaf.untried_actions:
            node, sampled_game = expand_leaf(leaf, board, sampled_game)

        sampled_game = rollout(board, sampled_game)

        player = board.current_player(sampled_game)
        won = False
        if player != identity_of_bot:
            won = True
        backpropagate(node, won)
        time_elapsed = time() - start

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    action = None
    best = 0.0
    for child in root_node.child_nodes:
        value = float(root_node.child_nodes[child].wins) / float(root_node.child_nodes[child].visits)

        if value >= best:
            best = value
            action = root_node.child_nodes[child].parent_action

    return action
