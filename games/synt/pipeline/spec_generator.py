"""
Spec Generator module for the Spec Validator Pipeline.

Generates TSLMT specifications from game configurations.
Each game type has its own generator function that takes configuration
parameters and produces a complete TSLMT spec.
"""

from typing import Any


def get_default_moves(var_name: str) -> str:
    """Return the default moves expression for a variable: identity, +1, -1."""
    return f"[{var_name} <- {var_name}] || [{var_name} <- add {var_name} i1()] || [{var_name} <- sub {var_name} i1()]"


def generate_ice_lake_spec(params: dict[str, Any], objective: str) -> str:
    """
    Generate an Ice Lake TSLMT specification.

    Args:
        params: Configuration with grid_size, goal, holes, start_pos
        objective: The TSL guarantee objective (e.g., "F atGoal")

    Returns:
        Complete TSLMT specification string
    """
    grid_size = params.get("grid_size", 4)
    goal = params.get("goal", {"x": 3, "y": 3})
    holes = params.get("holes", [{"x": 1, "y": 1}])
    start_pos = params.get("start_pos", {"x": 0, "y": 0})
    start_x = start_pos.get("x", 0)
    start_y = start_pos.get("y", 0)

    goal_x = goal.get("x", grid_size - 1)
    goal_y = goal.get("y", grid_size - 1)
    bound_max = grid_size - 1

    # Generate hole constants
    hole_constants = []
    for i, hole in enumerate(holes):
        hole_constants.append(f"hole{i}x = i{hole['x']}();")
        hole_constants.append(f"hole{i}y = i{hole['y']}();")
    hole_constants_str = "\n".join(hole_constants) if hole_constants else "/* No holes */"

    # Get custom variable updates if specified (raw TSL expressions)
    # If not specified, use default: identity, +1, -1
    variable_updates = params.get("variable_updates", {})
    x_moves = variable_updates.get("x", get_default_moves("x"))
    y_moves = variable_updates.get("y", get_default_moves("y"))

    spec = f'''var Int x
var Int y

SPECIFICATION

/* Ice Lake: Robot navigates {grid_size}x{grid_size} grid avoiding holes to reach the goal */
/* Goal: ({goal_x},{goal_y}), Start: ({start_x},{start_y}), Holes: {len(holes)} */

goalx = i{goal_x}();
goaly = i{goal_y}();
startx = i{start_x}();
starty = i{start_y}();
BOUND_MIN = i0();
BOUND_MAX = i{bound_max}();

/* Hole positions */
{hole_constants_str}

/* Position predicates */
inBounds = (gte x BOUND_MIN) && (lte x BOUND_MAX) && (gte y BOUND_MIN) && (lte y BOUND_MAX);

/* Potential Variable Updates */
xMoves = {x_moves};
yMoves = {y_moves};

assume {{
    eq x startx;
    eq y starty;
}}

guarantee {{
    /* Stay in bounds */
    G inBounds;

    /* Movement */
    G ((xMoves && [y <- y]) || ([x <- x] && yMoves));

    /* Objective */
    {objective};
}}
'''
    return spec


def generate_taxi_spec(params: dict[str, Any], objective: str) -> str:
    """
    Generate a Taxi TSLMT specification.

    Args:
        params: Configuration with grid_size, pickup, dropoff, locations, start_pos
        objective: The TSL guarantee objective
        params: Configuration with grid_size, barriers, start_pos, colored_cells, pickup_color, dropoff_color
        objective: The TSL guarantee objective (e.g., "F (eq x DEST_X && eq y DEST_Y && passengerInTaxi)")

    Returns:
        Complete TSLMT specification string
    """
    grid_size = params.get("grid_size", 5)
    barriers = params.get("barriers", params.get("Barriers", []))
    start_pos = params.get("start_pos", {"x": 2, "y": 2})

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

    # Generate barrier predicates
    if len(barriers) == 0:
        barriers_predicate = "false"
    else:
        barrier_checks = [f"(eq x i{b['x']}() && eq y i{b['y']}())" for b in barriers]
        barriers_predicate = " ||\n           ".join(barrier_checks)

    spec = f'''var Int x
var Int y
var Bool passengerInTaxi

SPECIFICATION

/* Taxi: Navigate {grid_size}x{grid_size} grid, pickup passenger, deliver to destination */
/* Pickup: {pickup_color}({pickup_x},{pickup_y}), Dropoff: {dropoff_color}({dropoff_x},{dropoff_y}), Start: ({start_x},{start_y}), Barriers: {len(barriers)} */

DEST_X = i{dropoff_x}();
DEST_Y = i{dropoff_y}();

START_X = i{start_x}();
START_Y = i{start_y}();

PASSENGER_X = i{pickup_x}();
PASSENGER_Y = i{pickup_y}();

BOUND_MIN = i0();
BOUND_MAX = i{bound_max}();

/* Possible Pick Up Areas/Destinations */
red = (eq x i{red_pos['x']}() && eq y i{red_pos['y']}());
green = (eq x i{green_pos['x']}() && eq y i{green_pos['y']}());
blue = (eq x i{blue_pos['x']}() && eq y i{blue_pos['y']}());
yellow = (eq x i{yellow_pos['x']}() && eq y i{yellow_pos['y']}());

/* Barriers */
barriers = {barriers_predicate};

/* Position predicates */
inBounds = (gte x BOUND_MIN) && (lte x BOUND_MAX) && (gte y BOUND_MIN) && (lte y BOUND_MAX) && !barriers;

/* Potential Variable Updates */
xMoves = [x <- x] || [x <- add x i1()] || [x <- sub x i1()];
yMoves = [y <- y] || [y <- add y i1()] || [y <- sub y i1()];


assume {{
    /* Taxi Start Position */
    eq x START_X;
    eq y START_Y;

    ! passengerInTaxi;
}}

guarantee {{
    /* Stay in bounds */
    G inBounds;

    /* Movement - only one direction at a time */
    G ((xMoves && [y <- y]) || ([x <- x] && yMoves));

    /* Passenger pickup: pick up when at location, keep once picked up */
    G (([passengerInTaxi <- true] && ((eq x PASSENGER_X && eq y PASSENGER_Y) || passengerInTaxi)) ||
         ([passengerInTaxi <- false] && !(eq x PASSENGER_X && eq y PASSENGER_Y) && !passengerInTaxi));

    /* Objective: deliver passenger to destination, avoid invalid destinations */
    {objective};
}}
'''
    return spec


def generate_cliff_walking_spec(params: dict[str, Any], objective: str) -> str:
    """
    Generate a Cliff Walking TSLMT specification.

    Args:
        params: Configuration with grid_size, cliff_min, cliff_max, start_pos, goal_pos
        objective: The TSL guarantee objective

    Returns:
        Complete TSLMT specification string
    """
    grid_cols = params.get("grid_size", 12)
    grid_rows = params.get("grid_rows", 4)
    cliff_min = params.get("cliff_min", 1)
    cliff_max = params.get("cliff_max", 10)
    cliffHeight = params.get("cliff_height", 1)
    start_pos = params.get("start_pos", {"x": 0, "y": 0})
    goal_pos = params.get("goal_pos", {"x": 11, "y": 0})

    max_x = grid_cols - 1
    max_y = grid_rows - 1
    goal_x = goal_pos.get("x", max_x)
    goal_y = goal_pos.get("y", 0)
    start_x = start_pos.get("x", 0)
    start_y = start_pos.get("y", 0)

    # Get custom variable updates if specified (raw TSL expressions)
    # If not specified, use default: identity, +1, -1
    variable_updates = params.get("variable_updates", {})
    x_moves = variable_updates.get("x", get_default_moves("x"))
    y_moves = variable_updates.get("y", get_default_moves("y"))

    spec = f'''var Int x
var Int y

SPECIFICATION

/* Cliff Walking: Robot navigates {grid_cols}x{grid_rows} grid avoiding cliff to reach goal */
/* Goal: ({goal_x},{goal_y}), Start: ({start_x},{start_y}), Cliff: y<{cliffHeight}, x=[{cliff_min},{cliff_max}] */

MIN_X = i0();
MAX_X = i{max_x}();
MIN_Y = i0();
MAX_Y = i{max_y}();

START_X = i{start_x}();
START_Y = i{start_y}();
goalx = i{goal_x}();
goaly = i{goal_y}();

cliffHeight = i{cliffHeight}();
cliffXMin = i{cliff_min}();
cliffXMax = i{cliff_max}();

/* Position predicates */
inBounds = (gte x MIN_X) && (lte x MAX_X) && (gte y MIN_Y) && (lte y MAX_Y);

/* Potential Variable Updates */
xMoves = {x_moves};
yMoves = {y_moves};

assume {{
    eq x START_X;
    eq y START_Y;
}}

guarantee {{
    /* Stay in bounds */
    G inBounds;

    /* Movement */
    G ((xMoves && [y <- y]) || ([x <- x] && yMoves));

    /* Objective */
    {objective};
}}
'''
    return spec


def generate_blackjack_spec(params: dict[str, Any], objective: str) -> str:
    """
    Generate a Blackjack TSLMT specification.

    Args:
        params: Configuration (minimal for blackjack)
        objective: The TSL guarantee objective

    Returns:
        Complete TSLMT specification string
    """
    spec = f'''var Bool stood
var Int count

inp Int dealer
inp Int card

SPECIFICATION

dealerStand = i17();

always assume {{
    gte card i1();
    lte card i11();
}}

/* All logic starts from second step (X G) to skip initialization */
guarantee {{
    G ([stood <- true] || [stood <- false]);
    G ([count <- count] || [count <- add count card]);

    X G ((!stood) -> [count <- add count card]);
    X G ((stood) -> [count <- count]);

    {objective};
}}
'''
    return spec


# Mapping from game name to generator function
SPEC_GENERATORS = {
    "ice_lake": generate_ice_lake_spec,
    "taxi": generate_taxi_spec,
    "cliff_walking": generate_cliff_walking_spec,
    "blackjack": generate_blackjack_spec,
}


def generate_spec(game_name: str, params: dict[str, Any], objective: str) -> str:
    """
    Generate a TSLMT specification for the given game and configuration.

    Args:
        game_name: Name of the game (ice_lake, taxi, cliff_walking, blackjack)
        params: Game-specific configuration parameters
        objective: The TSL guarantee objective

    Returns:
        Complete TSLMT specification string

    Raises:
        ValueError: If game_name is not recognized
    """
    if game_name == "frozen_lake":
        game_name = "ice_lake"
    generator = SPEC_GENERATORS.get(game_name.lower())
    if generator is None:
        raise ValueError(f"Unknown game type: {game_name}. Supported: {list(SPEC_GENERATORS.keys())}")

    return generator(params, objective)
