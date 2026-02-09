/*
 * Ice Lake Game - Automated Validation Harness
 * Generated from configuration parameters
 *
 * Grid: 4x4, Goal: (2,1), Start: (0,0), Holes: 3
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
#define GRID_SIZE 4
#define GOAL_X 2
#define GOAL_Y 1
#define START_X 0
#define START_Y 0
#define MAX_STEPS 1000
#define NUM_HOLES 3

static int hole_x[] = {1, 3, 0};
static int hole_y[] = {1, 0, 1};
static int num_holes = NUM_HOLES;

/* Step counter */
static int step_count = 0;

/* Forward declarations for controller state variables */
extern int x;
extern int y;

/* Check if position is a hole */
static bool is_hole(int px, int py) {
    for (int i = 0; i < num_holes; i++) {
        if (hole_x[i] >= 0 && px == hole_x[i] && py == hole_y[i]) {
            return true;
        }
    }
    return false;
}

/* Check if position is in bounds */
static bool in_bounds(int px, int py) {
    return px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE;
}

/* Check if at goal */
static bool at_goal(int px, int py) {
    return px == GOAL_X && py == GOAL_Y;
}

/*
 * read_inputs() - Called by controller at each step
 * Performs validation and handles termination
 */
void read_inputs(void) {
    step_count++;

    // Print current position for debugging
    printf("Step %d: Position (%d,%d)\n", step_count, x, y);

    /* Check for hole */
    if (is_hole(x, y)) {
        printf("FAIL: Fell into hole at (%d,%d) after %d steps\n", x, y, step_count);
        exit(1);
    }

    /* Check bounds */
    if (!in_bounds(x, y)) {
        printf("FAIL: Out of bounds at (%d,%d) after %d steps\n", x, y, step_count);
        exit(2);
    }

    /* Check for goal */
    if (at_goal(x, y)) {
        printf("SUCCESS: Goal reached at (%d,%d) in %d steps\n", x, y, step_count);
        exit(0);
    }

    /* Check step limit */
    if (step_count >= MAX_STEPS) {
        printf("FAIL: Step timeout (%d steps) at (%d,%d)\n", MAX_STEPS, x, y);
        exit(3);
    }
}


/* ======================================== CONTROLLER ======================================== */
#include <stdlib.h>
int x = 0;
int y = 0;
int main() {
  x = START_X; y = START_Y;
  {
    int prog_counter = 0;
    prog_counter = 37;
    for(;;)
      {
        if ((prog_counter == 1))
          {
            read_inputs();
            prog_counter = 1;
            continue;
          }
        if (((prog_counter == 2) && (y == 1)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 5;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 3) && (y == 0)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 6;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 10;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 5) && (y == 1)&& (x == 3)))
          {
            read_inputs();
            prog_counter = 2;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 6) && (y == 0)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 3;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 14;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 10;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 4;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            read_inputs();
            prog_counter = 4;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 11) && (y == 0)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 6;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = ((((0 <= y) && (y < 4)&& (x == 2)&& (!((x == -1)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 2)&& (!((y == 0)))&& (!((y == 1))))) ? 21 : 20);
            x = ((((0 <= y) && (y < 4)&& (x == 2)&& (!((x == -1)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 2)&& (!((y == 0)))&& (!((y == 1))))) ? (1 + x) : (-1 + x));
            continue;
          }
        if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 4;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
          {
            read_inputs();
            prog_counter = 7;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 4;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 9;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            read_inputs();
            prog_counter = 4;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = ((((0 <= y) && (y < 4)&& (x == 2)&& (!((x == -1)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 2)&& (!((y == 0)))&& (!((y == 1))))) ? 21 : 20);
            x = ((((0 <= y) && (y < 4)&& (x == 2)&& (!((x == -1)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 2)&& (!((y == 0)))&& (!((y == 1))))) ? (1 + x) : (-1 + x));
            continue;
          }
        if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 14;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
          {
            read_inputs();
            prog_counter = 28;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
          {
            read_inputs();
            prog_counter = 7;
            x = (-1 + x);
            continue;
          }
        if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 9;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 9;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
          {
            read_inputs();
            prog_counter = 28;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 4;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            read_inputs();
            prog_counter = 4;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
          {
            read_inputs();
            prog_counter = 24;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
          {
            read_inputs();
            prog_counter = (((2 <= y) && (y < 4)&& (x == 2)) ? 32 : 24);
            x = (((2 <= y) && (y < 4)&& (x == 2)) ? (1 + x) : (-1 + x));
            continue;
          }
        if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 9;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            read_inputs();
            prog_counter = 9;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            read_inputs();
            prog_counter = 14;
            x = (1 + x);
            continue;
          }
        if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            read_inputs();
            prog_counter = 4;
            x = (-1 + x);
            continue;
          }
        if ((prog_counter == 1))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 2) && (y == 1)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 3) && (y == 0)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 5) && (y == 1)&& (x == 3)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 6) && (y == 0)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 11) && (y == 0)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 33) && (y == 0)&& (x == 0)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 34) && (y == 0)&& (x == 1)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 35) && (y == 0)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if (((prog_counter == 36) && (y == 1)&& (x == 2)))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        if ((prog_counter == 37))
          {
            for(;;)
              {
                if ((prog_counter == 1))
                  break;
                if (((prog_counter == 2) && (y == 1)&& (x == 2)))
                  break;
                if (((prog_counter == 3) && (y == 0)&& (x == 2)))
                  break;
                if (((prog_counter == 4) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 5) && (y == 1)&& (x == 3)))
                  break;
                if (((prog_counter == 6) && (y == 0)&& (x == 1)))
                  break;
                if (((prog_counter == 7) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 8) && (2 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 9) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 10) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 11) && (y == 0)&& (x == 0)))
                  break;
                if (((prog_counter == 12) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 13) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 14) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 15) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 16) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 17) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 18) && (0 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 19) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 20) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 21) && (((0 <= y) && (y < 4)&& (x == 3)&& (!((x == 0)))&& (!((y == 0)))) || ((0 <= y) && (y < 4)&& (x == 3)&& (!((y == 0)))&& (!((y == 1)))))))
                  break;
                if (((prog_counter == 22) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 23) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 24) && (0 <= y)&& (y < 4)&& (x == 1)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 25) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 26) && (2 <= y)&& (y < 4)&& (x == 1)))
                  break;
                if (((prog_counter == 27) && (0 <= y)&& (y < 4)&& (x == 0)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 28) && (0 <= y)&& (y < 4)&& (x == 2)&& (!((y == 1)))))
                  break;
                if (((prog_counter == 29) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 30) && (2 <= y)&& (y < 4)&& (x == 0)))
                  break;
                if (((prog_counter == 31) && (1 <= y)&& (y < 4)&& (x == 2)))
                  break;
                if (((prog_counter == 32) && (2 <= y)&& (y < 4)&& (x == 3)))
                  break;
                if (((prog_counter == 36) && (y == 1)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 5;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 37) && ((4 <= y) || (!((x == 0)))|| (!((y == 0)))|| ((y == 0) && (x == 3)&& (!((x == 1))))|| ((y == 0) && (x == 3)&& (!((y == 1))))|| ((y == 1) && (x == 0)&& (!((x == 1))))|| (x < 0)|| (y < 0))))
                  {
                    read_inputs();
                    prog_counter = 1;
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = (1 + y);
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                if (((prog_counter == 35) && (y == 0)&& (x == 2)))
                  {
                    read_inputs();
                    prog_counter = 36;
                    y = (1 + y);
                    continue;
                  }
                if (((prog_counter == 34) && (y == 0)&& (x == 1)))
                  {
                    read_inputs();
                    prog_counter = 35;
                    x = (1 + x);
                    continue;
                  }
                if (((prog_counter == 33) && (y == 0)&& (x == 0)))
                  {
                    read_inputs();
                    prog_counter = 34;
                    x = (1 + x);
                    continue;
                  }
                if ((prog_counter == 37))
                  {
                    read_inputs();
                    prog_counter = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? 1 : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? 1 : 34) : 1)) : 1);
                    { /* atomic update */
                      int _new_x = (1 + x);
                      int _new_y = ((((!(((x == 1) && (y == 1)))) || (((x == 3) && (y == 0)) || ((x == 0) && (y == 1)))) && ((!((((x == 3) && (y == 0)) || ((x == 0) && (y == 1))))) || ((x == 1) && (y == 1)))) ? (((x == 2) && (y == 1)) ? (1 + y) : (((0 <= x) && (x < 4)&& (0 <= y)&& (y < 4)) ? ((!(((x == 0) && (y == 0)))) ? (1 + y) : y) : (1 + y))) : (1 + y));
                      x = _new_x;
                      y = _new_y;
                    }
                    continue;
                  }
                abort();
              }
            continue;
          }
        abort();
      }
  }
}
/* ======================================== CONTROLLER END ======================================== */
