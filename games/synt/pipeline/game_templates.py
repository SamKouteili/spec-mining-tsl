"""
Game Templates module for the Spec Validator Pipeline.

Generates C game harness files from configuration parameters.
Each game type has its own template function that takes configuration
parameters and produces a complete C game file (without the controller).
"""

from typing import Any


def generate_ice_lake_game(params: dict[str, Any]) -> str:
    """
    Generate an Ice Lake game harness.

    Args:
        params: Configuration with grid_size, goal, holes, start_pos

    Returns:
        Complete C game harness string (controller appended separately)
    """
    grid_size = params.get("grid_size", 4)
    goal = params.get("goal", {"x": 3, "y": 3})
    holes = params.get("holes", [{"x": 1, "y": 1}])
    start_pos = params.get("start_pos", {"x": 0, "y": 0})
    max_steps = params.get("max_steps", 1000)

    goal_x = goal.get("x", grid_size - 1)
    goal_y = goal.get("y", grid_size - 1)
    start_x = start_pos.get("x", 0)
    start_y = start_pos.get("y", 0)

    # Generate hole arrays
    hole_count = len(holes)
    if hole_count > 0:
        x_holes = "{" + ", ".join(str(h["x"]) for h in holes) + "};"
        y_holes = "{" + ", ".join(str(h["y"]) for h in holes) + "};"
    else:
        x_holes = "{-1};"
        y_holes = "{-1};"

    game = f'''/*
 * Ice Lake Game - Automated Validation Harness
 * Generated from configuration parameters
 *
 * Grid: {grid_size}x{grid_size}, Goal: ({goal_x},{goal_y}), Start: ({start_x},{start_y}), Holes: {hole_count}
 *
 * Exit codes:
 *   0 = Success (goal reached)
 *   1 = Fell into hole
 *   2 = Out of bounds
 *   3 = Step timeout
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* Configuration from pipeline */
#define GRID_SIZE {grid_size}
#define GOAL_X {goal_x}
#define GOAL_Y {goal_y}
#define START_X {start_x}
#define START_Y {start_y}
#define MAX_STEPS {max_steps}
#define NUM_HOLES {hole_count}

static int hole_x[] = {x_holes}
static int hole_y[] = {y_holes}
static int num_holes = NUM_HOLES;

/* Step counter */
static int step_count = 0;

/* Forward declarations for controller state variables */
extern int x;
extern int y;

/* Check if position is a hole */
static bool is_hole(int px, int py) {{
    for (int i = 0; i < num_holes; i++) {{
        if (hole_x[i] >= 0 && px == hole_x[i] && py == hole_y[i]) {{
            return true;
        }}
    }}
    return false;
}}

/* Check if position is in bounds */
static bool in_bounds(int px, int py) {{
    return px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE;
}}

/* Check if at goal */
static bool at_goal(int px, int py) {{
    return px == GOAL_X && py == GOAL_Y;
}}

/*
 * read_inputs() - Called by controller at each step
 * Performs validation and handles termination
 */
void read_inputs(void) {{
    step_count++;

    // Print current position for debugging
    printf("Step %d: Position (%d,%d)\\n", step_count, x, y);

    /* Check for hole */
    if (is_hole(x, y)) {{
        printf("FAIL: Fell into hole at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(1);
    }}

    /* Check bounds */
    if (!in_bounds(x, y)) {{
        printf("FAIL: Out of bounds at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(2);
    }}

    /* Check for goal */
    if (at_goal(x, y)) {{
        printf("SUCCESS: Goal reached at (%d,%d) in %d steps\\n", x, y, step_count);
        exit(0);
    }}

    /* Check step limit */
    if (step_count >= MAX_STEPS) {{
        printf("FAIL: Step timeout (%d steps) at (%d,%d)\\n", MAX_STEPS, x, y);
        exit(3);
    }}
}}

'''
    return game


def generate_taxi_game(params: dict[str, Any]) -> str:
    """
    Generate a Taxi game harness.

    Args:
        params: Configuration with grid_size, barriers, start_pos, colored_cells, pickup_color, dropoff_color

    Returns:
        Complete C game harness string (controller appended separately)
    """
    grid_size = params.get("grid_size", 5)
    barriers = params.get("barriers", params.get("Barriers", []))
    start_pos = params.get("start_pos", {"x": 2, "y": 2})
    max_steps = params.get("max_steps", 1000)

    # Get colored cell positions (defaults match traditional taxi layout)
    colored_cells = params.get("colored_cells", {
        "red": {"x": 0, "y": 0},
        "green": {"x": 4, "y": 4},
        "blue": {"x": 4, "y": 0},
        "yellow": {"x": 0, "y": 4}
    })

    # Get pickup and dropoff colors
    pickup_color = params.get("pickup_color", "red")
    dropoff_color = params.get("dropoff_color", "green")

    bound_max = grid_size - 1
    start_x = start_pos.get("x", 2)
    start_y = start_pos.get("y", 2)

    # Get coordinates from colored cells
    red_pos = colored_cells.get("red", {"x": 0, "y": 0})
    green_pos = colored_cells.get("green", {"x": bound_max, "y": bound_max})
    blue_pos = colored_cells.get("blue", {"x": bound_max, "y": 0})
    yellow_pos = colored_cells.get("yellow", {"x": 0, "y": bound_max})

    # Derive pickup/dropoff from colors
    color_positions = {
        "red": red_pos,
        "green": green_pos,
        "blue": blue_pos,
        "yellow": yellow_pos
    }
    pickup_pos = color_positions[pickup_color]
    dropoff_pos = color_positions[dropoff_color]

    pickup_x = pickup_pos.get("x", 0)
    pickup_y = pickup_pos.get("y", 0)
    dropoff_x = dropoff_pos.get("x", bound_max)
    dropoff_y = dropoff_pos.get("y", bound_max)

    # Generate barrier arrays
    barrier_count = len(barriers)
    if barrier_count > 0:
        x_barriers = "{" + ", ".join(str(b["x"]) for b in barriers) + "};"
        y_barriers = "{" + ", ".join(str(b["y"]) for b in barriers) + "};"
    else:
        x_barriers = "{-1};"
        y_barriers = "{-1};"

    # Determine forbidden colors (colors that are neither pickup nor dropoff)
    all_colors = ["red", "green", "blue", "yellow"]
    forbidden_colors = [c for c in all_colors if c != pickup_color and c != dropoff_color]

    # Generate forbidden color position defines
    forbidden_defines = ""
    forbidden_checks = []
    for i, color in enumerate(forbidden_colors):
        pos = color_positions[color]
        forbidden_defines += f"#define FORBIDDEN_{i}_X {pos['x']}\n"
        forbidden_defines += f"#define FORBIDDEN_{i}_Y {pos['y']}\n"
        forbidden_checks.append(f'(px == FORBIDDEN_{i}_X && py == FORBIDDEN_{i}_Y)')

    num_forbidden = len(forbidden_colors)

    # Build forbidden check function body
    if forbidden_checks:
        forbidden_check_expr = " || ".join(forbidden_checks)
    else:
        forbidden_check_expr = "false"

    # Generate color names for comments
    forbidden_names = ", ".join(forbidden_colors) if forbidden_colors else "none"

    game = f'''/*
 * Taxi Game - Automated Validation Harness
 * Generated from configuration parameters
 *
 * Grid: {grid_size}x{grid_size}, Pickup: {pickup_color}({pickup_x},{pickup_y}), Dropoff: {dropoff_color}({dropoff_x},{dropoff_y}), Start: ({start_x},{start_y}), Barriers: {barrier_count}
 * Forbidden colors: {forbidden_names}
 *
 * Exit codes:
 *   0 = Success (passenger delivered)
 *   1 = Hit barrier
 *   2 = Out of bounds
 *   3 = Step timeout
 *   4 = Entered forbidden colored cell
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* Configuration from pipeline */
#define GRID_SIZE {grid_size}
#define PICKUP_X {pickup_x}
#define PICKUP_Y {pickup_y}
#define DEST_X {dropoff_x}
#define DEST_Y {dropoff_y}
#define START_X {start_x}
#define START_Y {start_y}
#define MAX_STEPS {max_steps}
#define NUM_BARRIERS {barrier_count}

/* Colored cell positions */
#define RED_X {red_pos['x']}
#define RED_Y {red_pos['y']}
#define GREEN_X {green_pos['x']}
#define GREEN_Y {green_pos['y']}
#define BLUE_X {blue_pos['x']}
#define BLUE_Y {blue_pos['y']}
#define YELLOW_X {yellow_pos['x']}
#define YELLOW_Y {yellow_pos['y']}

/* Forbidden colored cells (neither pickup nor dropoff) */
#define NUM_FORBIDDEN {num_forbidden}
{forbidden_defines}
static int barrier_x[] = {x_barriers}
static int barrier_y[] = {y_barriers}
static int num_barriers = NUM_BARRIERS;

/* Step counter */
static int step_count = 0;

/* Forward declarations for controller state variables */
extern bool passengerInTaxi;
extern int x;
extern int y;

/* Check if position is a barrier */
static bool is_barrier(int px, int py) {{
    for (int i = 0; i < num_barriers; i++) {{
        if (barrier_x[i] >= 0 && px == barrier_x[i] && py == barrier_y[i]) {{
            return true;
        }}
    }}
    return false;
}}

/* Check if position is in bounds */
static bool in_bounds(int px, int py) {{
    return px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE;
}}

/* Check if position is a forbidden colored cell */
static bool is_forbidden_color(int px, int py) {{
    return {forbidden_check_expr};
}}

/* Check if at destination */
static bool at_destination(int px, int py) {{
    return px == DEST_X && py == DEST_Y;
}}

/*
 * read_inputs() - Called by controller at each step
 * Performs validation and handles termination
 */
void read_inputs(void) {{
    step_count++;

    // Print current position for debugging
    printf("Step %d: Position (%d,%d), passengerInTaxi=%d\\n", step_count, x, y, passengerInTaxi);

    /* Check for barrier */
    if (is_barrier(x, y)) {{
        printf("FAIL: Hit barrier at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(1);
    }}

    /* Check bounds */
    if (!in_bounds(x, y)) {{
        printf("FAIL: Out of bounds at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(2);
    }}

    /* Check for forbidden colored cell */
    if (is_forbidden_color(x, y)) {{
        printf("FAIL: Entered forbidden colored cell at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(4);
    }}

    /* Check for delivery complete (at destination with passenger) */
    if (at_destination(x, y) && passengerInTaxi) {{
        printf("SUCCESS: Passenger delivered in %d steps\\n", step_count);
        exit(0);
    }}

    /* Check step limit */
    if (step_count >= MAX_STEPS) {{
        printf("FAIL: Step timeout (%d steps) at (%d,%d), passengerInTaxi=%d\\n",
               MAX_STEPS, x, y, passengerInTaxi);
        exit(3);
    }}
}}

'''
    return game


def generate_cliff_walking_game(params: dict[str, Any]) -> str:
    """
    Generate a Cliff Walking game harness.

    Args:
        params: Configuration with grid_size, grid_rows, cliff_min, cliff_max, goal_pos, start_pos

    Returns:
        Complete C game harness string (controller appended separately)
    """
    grid_cols = params.get("grid_size", 12)
    grid_rows = params.get("grid_rows", 4)
    cliff_min = params.get("cliff_min", 1)
    cliff_max = params.get("cliff_max", 10)
    goal_pos = params.get("goal_pos", {"x": 11, "y": 0})
    start_pos = params.get("start_pos", {"x": 0, "y": 0})
    max_steps = params.get("max_steps", 1000)

    goal_x = goal_pos.get("x", grid_cols - 1)
    goal_y = goal_pos.get("y", 0)
    start_x = start_pos.get("x", 0)
    start_y = start_pos.get("y", 0)

    game = f'''/*
 * Cliff Walking Game - Automated Validation Harness
 * Generated from configuration parameters
 *
 * Grid: {grid_cols}x{grid_rows}, Goal: ({goal_x},{goal_y}), Start: ({start_x},{start_y}), Cliff: y=0, x=[{cliff_min},{cliff_max}]
 *
 * Exit codes:
 *   0 = Success (goal reached)
 *   1 = Fell off cliff
 *   2 = Out of bounds
 *   3 = Step timeout
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* Configuration from pipeline */
#define GRID_COLS {grid_cols}
#define GRID_ROWS {grid_rows}
#define GOAL_X {goal_x}
#define GOAL_Y {goal_y}
#define START_X {start_x}
#define START_Y {start_y}
#define CLIFF_Y 0
#define CLIFF_X_MIN {cliff_min}
#define CLIFF_X_MAX {cliff_max}
#define MAX_STEPS {max_steps}

/* Step counter */
static int step_count = 0;

/* Forward declarations for controller state variables */
extern int x;
extern int y;

/* Check if position is on cliff */
static bool is_cliff(int px, int py) {{
    return py == CLIFF_Y && px >= CLIFF_X_MIN && px <= CLIFF_X_MAX;
}}

/* Check if position is in bounds */
static bool in_bounds(int px, int py) {{
    return px >= 0 && px < GRID_COLS && py >= 0 && py < GRID_ROWS;
}}

/* Check if at goal */
static bool at_goal(int px, int py) {{
    return px == GOAL_X && py == GOAL_Y;
}}

/*
 * read_inputs() - Called by controller at each step
 * Performs validation and handles termination
 */
void read_inputs(void) {{
    step_count++;

    // Print current position for debugging
    printf("Step %d: Position (%d,%d)\\n", step_count, x, y);

    /* Check for cliff */
    if (is_cliff(x, y)) {{
        printf("FAIL: Fell off cliff at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(1);
    }}

    /* Check bounds */
    if (!in_bounds(x, y)) {{
        printf("FAIL: Out of bounds at (%d,%d) after %d steps\\n", x, y, step_count);
        exit(2);
    }}

    /* Check for goal */
    if (at_goal(x, y)) {{
        printf("SUCCESS: Goal reached at (%d,%d) in %d steps\\n", x, y, step_count);
        exit(0);
    }}

    /* Check step limit */
    if (step_count >= MAX_STEPS) {{
        printf("FAIL: Step timeout (%d steps) at (%d,%d)\\n", MAX_STEPS, x, y);
        exit(3);
    }}
}}

'''
    return game


def generate_blackjack_game(params: dict[str, Any]) -> str:
    """
    Generate a Blackjack game harness.

    Args:
        params: Configuration with num_games

    Returns:
        Complete C game harness string (controller appended separately)
    """
    num_games = params.get("num_games", 100)

    game = f'''/*
 * Blackjack Game - Full Simulation Harness
 *
 * Plays {num_games} games of blackjack.
 * Controller manages: count (player hand total), stood (stand decision)
 * Harness provides: card (next card), dealer (dealer's visible card)
 *
 * Exit codes:
 *   0 = All games completed
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/* Configuration */
#define NUM_GAMES {num_games}
#define DEALER_STAND 17
#define BUST_THRESHOLD 21

/* Controller variables */
extern bool stood;
extern int count;

/* Input variables (defined by controller, set by harness) */
extern int dealer;
extern int card;

/* Game state */
static int current_game = 0;
static int total_steps = 0;
static bool game_in_progress = false;
static bool first_call = true;

/* Game outcome tracking */
static int player_wins = 0;
static int dealer_wins = 0;
static int pushes = 0;
static int player_busts = 0;
static int dealer_busts = 0;
static int player_blackjacks = 0;
static int dealer_blackjacks = 0;

/* Dealer state for current game */
static int dealer_total = 0;
static int dealer_hole_card = 0;

/* Generate random card value (2-11) */
static int random_card(void) {{
    int val = (rand() % 13) + 1;
    if (val > 10) val = 10;      /* Face cards = 10 */
    if (val == 1) val = 11;      /* Ace = 11 (simplified) */
    return val;
}}

/* Play out dealer's hand (draws to 17) */
static void play_dealer_hand(void) {{
    printf("  Dealer hole card: %d, total: %d\\n", dealer_hole_card, dealer_total);
    while (dealer_total < DEALER_STAND) {{
        int c = random_card();
        dealer_total += c;
        printf("  Dealer draws %d, total: %d\\n", c, dealer_total);
    }}
    if (dealer_total > BUST_THRESHOLD) {{
        printf("  Dealer BUSTS with %d\\n", dealer_total);
    }} else {{
        printf("  Dealer stands with %d\\n", dealer_total);
    }}
}}

/* Determine winner (called only when player stands without busting) */
static void determine_winner(void) {{
    printf("  Final: Player %d vs Dealer %d - ", count, dealer_total);
    if (dealer_total > BUST_THRESHOLD) {{
        dealer_busts++;
        player_wins++;
        printf("PLAYER WINS (dealer bust)\\n");
    }} else if (count > dealer_total) {{
        player_wins++;
        printf("PLAYER WINS\\n");
    }} else if (dealer_total > count) {{
        dealer_wins++;
        printf("DEALER WINS\\n");
    }} else {{
        pushes++;
        printf("PUSH\\n");
    }}
}}

void read_inputs(void) {{
    total_steps++;

    /* Initialize on first call */
    if (first_call) {{
        srand((unsigned int)time(NULL));
        first_call = false;
        game_in_progress = false;
    }}

    /* Process previous decision if game in progress */
    if (game_in_progress) {{
        /* First check if player busted (from a hit) */
        if (count > BUST_THRESHOLD) {{
            printf("  Player BUSTS with %d! Dealer wins.\\n", count);
            player_busts++;
            dealer_wins++;
            game_in_progress = false;
        }} else if (stood) {{
            /* Player stands - dealer plays */
            printf("  Player STANDS with %d\\n", count);
            play_dealer_hand();
            determine_winner();
            game_in_progress = false;
        }} else {{
            /* Player hits - show updated count */
            printf("  Player HITS, now has: %d\\n", count);
        }}
    }}

    /* Start new game if needed */
    if (!game_in_progress) {{
        current_game++;

        /* Check if all games done */
        if (current_game > NUM_GAMES) {{
            printf("\\n============ FINAL RESULTS ============\\n");
            printf("Games played:      %d\\n", NUM_GAMES);
            printf("\\n--- Outcomes ---\\n");
            printf("Player wins:       %d (%.1f%%)\\n", player_wins, 100.0 * player_wins / NUM_GAMES);
            printf("Dealer wins:       %d (%.1f%%)\\n", dealer_wins, 100.0 * dealer_wins / NUM_GAMES);
            printf("Pushes:            %d (%.1f%%)\\n", pushes, 100.0 * pushes / NUM_GAMES);
            printf("\\n--- Busts ---\\n");
            printf("Player busts:      %d\\n", player_busts);
            printf("Dealer busts:      %d\\n", dealer_busts);
            printf("\\n--- Blackjacks ---\\n");
            printf("Player blackjacks: %d\\n", player_blackjacks);
            printf("Dealer blackjacks: %d\\n", dealer_blackjacks);
            printf("Total steps:       %d\\n", total_steps);
            printf("\\nSUCCESS: All games completed\\n");
            exit(0);
        }}

        /* Deal new game */
        printf("\\n====== Game %d ======\\n", current_game);

        /* Deal dealer's cards */
        dealer = random_card();
        dealer_hole_card = random_card();
        dealer_total = dealer + dealer_hole_card;

        /* Deal player's initial 2 cards - controller will set count */
        int card1 = random_card();
        int card2 = random_card();
        count = card1 + card2;

        printf("  Player: %d + %d = %d\\n", card1, card2, count);
        printf("  Dealer shows: %d\\n", dealer);

        /* Check for blackjacks */
        if (count == 21 && dealer_total == 21) {{
            printf("  Both BLACKJACK - Loss.\\n");
            player_blackjacks++;
            dealer_blackjacks++;
            dealer_wins++;
            game_in_progress = false;
        }} else if (count == 21) {{
            printf("  Player BLACKJACK!\\n");
            player_blackjacks++;
            player_wins++;
            game_in_progress = false;
        }} else if (dealer_total == 21) {{
            printf("  Dealer BLACKJACK! (hole: %d)\\n", dealer_hole_card);
            dealer_blackjacks++;
            dealer_wins++;
            game_in_progress = false;
        }} else {{
            game_in_progress = true;
            /* First step - spec uses X G so this card value is ignored */
            card = random_card();
        }}
    }} else {{
        /* Continue game - provide next card for the hit */
        card = random_card();
    }}
}}

'''
    return game


# Mapping from game name to generator function
GAME_GENERATORS = {
    "ice_lake": generate_ice_lake_game,
    "taxi": generate_taxi_game,
    "cliff_walking": generate_cliff_walking_game,
    "blackjack": generate_blackjack_game,
}


def generate_game(game_name: str, params: dict[str, Any]) -> str:
    """
    Generate a C game harness for the given game and configuration.

    Args:
        game_name: Name of the game (ice_lake, taxi, cliff_walking, blackjack)
        params: Game-specific configuration parameters

    Returns:
        Complete C game harness string (without controller)

    Raises:
        ValueError: If game_name is not recognized
    """
    if game_name == "frozen_lake":
        game_name = "ice_lake"
    generator = GAME_GENERATORS.get(game_name.lower())
    if generator is None:
        raise ValueError(f"Unknown game type: {game_name}. Supported: {list(GAME_GENERATORS.keys())}")

    return generator(params)
