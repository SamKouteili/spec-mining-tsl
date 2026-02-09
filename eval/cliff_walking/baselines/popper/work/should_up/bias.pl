%% Bias file for should_up

head_pred(should_up, 1).

body_pred(at_goal, 1).
body_pred(at_goal_x, 1).
body_pred(not_at_goal_x, 1).
body_pred(left_of_goal, 1).
body_pred(above_goal_y, 1).
body_pred(above_cliff, 1).
body_pred(cliff_danger, 1).
body_pred(safe, 1).
body_pred(at_start, 1).
body_pred(not_at_start, 1).

max_body(3).
max_clauses(3).
max_vars(2).
