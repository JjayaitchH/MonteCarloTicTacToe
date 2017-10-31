
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

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

        #for child in node.child_nodes:
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
    action = choice(node.untried_actions)
    state = board.next_state(state, action)
    actions = board.legal_actions(state)
    node.child_nodes[action] = MCTSNode(node, action, actions)
    node.untried_actions.remove(action)
    return node.child_nodes[action], state


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """
    ROLLOUTS = 10
    MAX_DEPTH = 5

    me = board.current_player(state)

    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    def outcome(owned_boxes, game_points):
        if game_points is not None:
            # Try to normalize it up?  Not so sure about this code anyhow.
            red_score = game_points[1]*9
            blue_score = game_points[2]*9
        else:
            red_score = len([v for v in owned_boxes.values() if v == 1])
            blue_score = len([v for v in owned_boxes.values() if v == 2])
        return red_score - blue_score if me == 1 else blue_score - red_score

    while not board.is_ended:
        if board.current_player(state) == me:
            moves = board.legal_actions(state)
            best_move = moves[0]
            best_expectation = float('-inf')

            for move in moves:
                total_score = 0.0

                # Sample a set number of games where the target move is immediately applied.
                for r in range(ROLLOUTS):
                    rollout_state = board.next_state(state, move)

                    # Only play to the specified depth.
                    for i in range(MAX_DEPTH):
                        if board.is_ended(rollout_state):
                            break
                        rollout_move = choice(board.legal_actions(rollout_state))
                        rollout_state = board.next_state(rollout_state, rollout_move)

                    total_score += outcome(board.owned_boxes(rollout_state),
                                           board.points_values(rollout_state))

                expectation = float(total_score) / ROLLOUTS

                # If the current move has a better average score, replace best_move and best_expectation
                if expectation > best_expectation:
                    best_expectation = expectation
                    best_move = move

            print("Rollout bot picking %s with expected score %f" % (str(best_move), best_expectation))
            state = board.next_state(state, best_move)
        else:
            state = board.next_state(state, choice(board.legal_actions(state)))
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

    for step in range(num_nodes):
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
