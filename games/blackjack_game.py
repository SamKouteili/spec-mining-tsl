#!/usr/bin/env python3
"""
Interactive Blackjack Game for TSL_f Specification Mining

This version generates traces where positive/negative indicates whether a STRATEGY
was followed, not whether the player won (since even optimal play can lose).

State representation for mining:
  - count: player's current hand total (integer, 4-21+)
  - dealer: dealer's visible card value (1-10, where 1=Ace)
  - dealerStand: constant 17 (threshold when dealer must stand)
  - stood: whether player has stood (0 or 1)
  - busted: whether player busted (0 or 1)

Actions: 'h' to hit, 's' to stand

Positive traces: Player followed the specified strategy
Negative traces: Player violated the strategy at least once

Mineable TSL_f patterns:
  - G (ge count dealerStand -> X stood)     -- "stand at 17+"
  - G (lt count 12 -> !X stood)             -- "never stand below 12"
  - G ((ge count 12) & (le dealer 6) -> X stood)  -- "stand 12+ vs weak dealer"
"""

import os
import json
import sys
import tty
import termios
import random
import argparse
from datetime import datetime
from pathlib import Path
from enum import Enum


class Strategy(Enum):
    """Available strategies for trace generation."""
    THRESHOLD = "threshold"      # Stand at count >= 17
    CONSERVATIVE = "conservative"  # Stand at 12+ vs dealer 2-6
    BASIC = "basic"              # Full basic strategy


class BlackjackGame:
    """Blackjack game for TSL_f trace generation."""

    # Constants
    DEALER_STAND_THRESHOLD = 17  # Dealer must stand at 17
    DEALER_WEAK_THRESHOLD = 6    # Dealer showing 2-6 is "weak" (likely to bust)
    STAND_THRESHOLD_VS_WEAK = 12 # Stand at 12+ vs weak dealer
    BUST_THRESHOLD = 21

    def __init__(self, session_dir=None, strategy=Strategy.THRESHOLD):
        """Initialize a Blackjack game.

        Args:
            session_dir: Directory to save traces
            strategy: Which strategy to use for pos/neg classification
        """
        self.strategy = strategy
        self.session_dir = session_dir or self._create_session_dir()

        # Initialize game state
        self._init_game()

    def _create_session_dir(self):
        """Create a new directory for this game session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/blackjack/{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pos").mkdir(exist_ok=True)
        (session_dir / "neg").mkdir(exist_ok=True)
        return session_dir

    def _init_game(self):
        """Initialize deck and deal initial hands."""
        # Create and shuffle deck (single deck for simplicity)
        self.deck = []
        for _ in range(4):  # 4 suits
            # Card values: 1=Ace, 2-10=number cards, 10 for J/Q/K
            self.deck.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        random.shuffle(self.deck)

        # Deal initial hands
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]

        # Game state
        self.stood = False
        self.busted = False
        self.game_over = False
        self.strategy_violated = False
        self.trace = []

        # Log initial state
        self._log_state()

    def _draw_card(self):
        """Draw a card from the deck."""
        if not self.deck:
            # Reshuffle
            for _ in range(4):
                self.deck.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
            random.shuffle(self.deck)
        return self.deck.pop()

    def _calculate_count(self, cards):
        """Calculate hand value, treating Ace optimally."""
        total = sum(cards)
        # Use ace as 11 if it doesn't bust
        if 1 in cards and total + 10 <= self.BUST_THRESHOLD:
            total += 10
        return total

    def _get_dealer_showing(self):
        """Get dealer's visible card (first card)."""
        return self.dealer_cards[0]

    def _log_state(self):
        """Log the current state as a jsonl entry.

        Constants and derived booleans depend on strategy:
        - THRESHOLD: standThreshold=17
        - CONSERVATIVE: standThreshold=17, isWeakDealer (dealer <= 6)
        - BASIC: standThreshold=17, standVsWeakMin=13, isWeakDealer
        """
        count = self._calculate_count(self.player_cards)
        dealer = self._get_dealer_showing()

        state = {
            "count": count,
            "stood": self.stood,
        }

        # Add strategy-specific constants and derived booleans
        if self.strategy == Strategy.THRESHOLD:
            state["standThreshold"] = self.DEALER_STAND_THRESHOLD  # 17
        elif self.strategy == Strategy.CONSERVATIVE:
            state["standThreshold"] = self.DEALER_STAND_THRESHOLD  # 17
            state["standVsWeakMin"] = 12  # Stand at 12+ vs weak dealer
            # Derived boolean: is the dealer showing a weak card (2-6)?
            state["isWeakDealer"] = 2 <= dealer <= self.DEALER_WEAK_THRESHOLD
        elif self.strategy == Strategy.BASIC:
            state["standThreshold"] = self.DEALER_STAND_THRESHOLD  # 17
            state["standVsWeakMin"] = 13  # Stand at 13+ vs weak dealer
            # Derived boolean: is the dealer showing a weak card (2-6)?
            state["isWeakDealer"] = 2 <= dealer <= self.DEALER_WEAK_THRESHOLD

        self.trace.append(state)

    def _should_hit(self, count, dealer):
        """Determine if player should hit according to strategy.

        Returns True if strategy says to hit, False if should stand.
        """
        if self.strategy == Strategy.THRESHOLD:
            # Simple: hit if count < 17
            return count < self.DEALER_STAND_THRESHOLD

        elif self.strategy == Strategy.CONSERVATIVE:
            # Stand at 12+ vs dealer 2-6, otherwise like threshold
            # Note: dealer 2-6 is weak (Ace=1 is NOT weak, it's strong)
            if count >= 17:
                return False  # Always stand
            elif count >= 12 and 2 <= dealer <= self.DEALER_WEAK_THRESHOLD:
                return False  # Stand vs weak dealer
            elif count <= 11:
                return True  # Always hit
            else:
                return True  # Hit vs strong dealer (7-A)

        elif self.strategy == Strategy.BASIC:
            # Basic strategy: stand at 17+, hit at 11-, stand at 13+ vs weak dealer
            # Note: dealer 2-6 is weak (Ace=1 is NOT weak, it's strong)
            if count >= 17:
                return False
            elif count <= 11:
                return True
            elif count >= 13 and 2 <= dealer <= 6:
                return False
            else:
                return True

        return count < 17  # Default

    def hit(self):
        """Player takes another card."""
        if self.game_over:
            return

        count = self._calculate_count(self.player_cards)
        dealer = self._get_dealer_showing()

        # Check if this action violates strategy
        if not self._should_hit(count, dealer):
            self.strategy_violated = True

        # Draw card
        self.player_cards.append(self._draw_card())
        count = self._calculate_count(self.player_cards)

        # Check for bust
        if count > self.BUST_THRESHOLD:
            self.busted = True
            self.game_over = True

        self._log_state()

    def stand(self):
        """Player stands."""
        if self.game_over:
            return

        count = self._calculate_count(self.player_cards)
        dealer = self._get_dealer_showing()

        # Check if this action violates strategy
        if self._should_hit(count, dealer):
            self.strategy_violated = True

        self.stood = True
        self.game_over = True

        self._log_state()

    def reset(self):
        """Reset for a new game."""
        self._init_game()

    def is_positive_trace(self):
        """Return True if trace follows strategy (positive trace)."""
        return not self.strategy_violated

    def save_trace(self):
        """Save the trace to the appropriate directory."""
        is_positive = self.is_positive_trace()
        subdir = "pos" if is_positive else "neg"
        prefix = "pos_trace_" if is_positive else "neg_trace_"

        trace_dir = self.session_dir / subdir
        existing_files = list(trace_dir.glob(f"{prefix}*.jsonl"))
        trace_num = len(existing_files) + 1

        trace_file = trace_dir / f"{prefix}{trace_num}.jsonl"

        with open(trace_file, 'w') as f:
            for entry in self.trace:
                f.write(json.dumps(entry) + '\n')

        return trace_file

    def _trace_to_key(self, trace):
        """Convert trace to hashable key for uniqueness checking."""
        return tuple(tuple(sorted(state.items())) for state in trace)

    def get_config(self, name: str = None) -> dict:
        """
        Export the current game configuration as a dictionary.

        Args:
            name: Optional name for this configuration

        Returns:
            Configuration dictionary compatible with spec_generator
        """
        if name is None:
            name = f"blackjack_{self.strategy.value}"

        config = {
            "name": name,
            "strategy": self.strategy.value,
            "dealer_stand_threshold": self.DEALER_STAND_THRESHOLD,
            "bust_threshold": self.BUST_THRESHOLD,
        }

        # Add strategy-specific parameters
        if self.strategy in [Strategy.CONSERVATIVE, Strategy.BASIC]:
            config["dealer_weak_threshold"] = self.DEALER_WEAK_THRESHOLD
            config["stand_threshold_vs_weak"] = self.STAND_THRESHOLD_VS_WEAK

        return config

    def _bfs_optimal_path(self):
        """BFS exploration following optimal decision tree."""
        while not self.game_over:
            count = self._calculate_count(self.player_cards)
            dealer = self._get_dealer_showing()

            if self._should_hit(count, dealer):
                self.hit()
            else:
                self.stand()

    def _bfs_explore_alternate(self):
        """BFS exploration with alternate decision branches."""
        branch_taken = False
        steps = 0
        max_steps = 10

        while not self.game_over and steps < max_steps:
            steps += 1
            count = self._calculate_count(self.player_cards)
            dealer = self._get_dealer_showing()

            should_hit = self._should_hit(count, dealer)

            if not branch_taken and random.random() < 0.5:
                if should_hit:
                    self.stand()
                else:
                    self.hit()
                branch_taken = True
            else:
                if should_hit:
                    self.hit()
                else:
                    self.stand()

        if not branch_taken and not self.game_over:
            count = self._calculate_count(self.player_cards)
            dealer = self._get_dealer_showing()
            should_hit = self._should_hit(count, dealer)

            if should_hit:
                self.stand()
            else:
                self.hit()

    def generate_traces(self, num_traces):
        """Generate num_traces unique traces via BFS exploration."""
        pos_traces = set()
        neg_traces = set()

        attempts = 0
        max_attempts = num_traces * 200

        print(f"\nBFS exploration: generating {num_traces * 2} traces...")
        print(f"Strategy: {self.strategy.value}")

        while (len(pos_traces) < num_traces or len(neg_traces) < num_traces) and attempts < max_attempts:
            attempts += 1

            if len(pos_traces) < num_traces:
                self.reset()
                self._bfs_optimal_path()

                if self.is_positive_trace():
                    trace_key = self._trace_to_key(self.trace)
                    if trace_key not in pos_traces:
                        pos_traces.add(trace_key)
                        self._save_trace_to_file(self.trace, is_positive=True,
                                                  trace_num=len(pos_traces))
                        print(f"  BFS trace {len(pos_traces)}/{num_traces} (optimal)")

            if len(neg_traces) < num_traces:
                self.reset()
                self._bfs_explore_alternate()

                if not self.is_positive_trace():
                    trace_key = self._trace_to_key(self.trace)
                    if trace_key not in neg_traces:
                        neg_traces.add(trace_key)
                        self._save_trace_to_file(self.trace, is_positive=False,
                                                  trace_num=len(neg_traces))
                        print(f"  BFS trace {len(neg_traces)}/{num_traces} (alternate)")

        if len(pos_traces) < num_traces:
            print(f"\nWarning: Only found {len(pos_traces)} unique optimal paths")
        if len(neg_traces) < num_traces:
            print(f"\nWarning: Only found {len(neg_traces)} unique alternate paths")

        print(f"\nBFS exploration complete!")
        print(f"  Optimal paths: {len(pos_traces)}")
        print(f"  Alternate paths: {len(neg_traces)}")

    def _save_trace_to_file(self, trace, is_positive, trace_num):
        """Save a trace to file."""
        subdir = "pos" if is_positive else "neg"
        prefix = "pos_trace_" if is_positive else "neg_trace_"

        trace_dir = self.session_dir / subdir
        trace_file = trace_dir / f"{prefix}{trace_num}.jsonl"

        with open(trace_file, 'w') as f:
            for entry in trace:
                f.write(json.dumps(entry) + '\n')

    def display(self):
        """Display the current game state."""
        os.system('clear' if os.name == 'posix' else 'cls')

        # ANSI color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        RESET = '\033[0m'

        count = self._calculate_count(self.player_cards)
        dealer = self._get_dealer_showing()

        print("=" * 50)
        print(f"BLACKJACK - Strategy: {self.strategy.value}")
        print("=" * 50)

        # Show dealer's hand
        print(f"\n{CYAN}Dealer showing:{RESET} {self._card_name(dealer)}")
        if self.game_over and self.stood:
            dealer_count = self._calculate_count(self.dealer_cards)
            dealer_cards_str = ', '.join(self._card_name(c) for c in self.dealer_cards)
            print(f"Dealer's hand: {dealer_cards_str} (Total: {dealer_count})")

        # Show player's hand
        player_cards_str = ', '.join(self._card_name(c) for c in self.player_cards)
        print(f"\n{YELLOW}Your hand:{RESET} {player_cards_str}")
        print(f"Count: {count}")

        # Show constants
        print(f"\n{CYAN}Dealer stands at:{RESET} {self.DEALER_STAND_THRESHOLD}")

        # Show strategy recommendation
        if not self.game_over:
            should_hit = self._should_hit(count, dealer)
            rec = "HIT" if should_hit else "STAND"
            print(f"\n{CYAN}Strategy says:{RESET} {rec}")

        print("\nControls: h=Hit | s=Stand | q=Quit")

        if self.game_over:
            if self.busted:
                print(f"\n{RED}BUST! You went over 21.{RESET}")
            elif self.stood:
                # Dealer plays out
                while self._calculate_count(self.dealer_cards) < self.DEALER_STAND_THRESHOLD:
                    self.dealer_cards.append(self._draw_card())

                dealer_count = self._calculate_count(self.dealer_cards)

                if dealer_count > 21:
                    print(f"\n{GREEN}Dealer busts! YOU WIN!{RESET}")
                elif count > dealer_count:
                    print(f"\n{GREEN}YOU WIN! {count} beats {dealer_count}{RESET}")
                elif count == dealer_count:
                    print(f"\n{YELLOW}PUSH - It's a tie.{RESET}")
                else:
                    print(f"\n{RED}Dealer wins. {dealer_count} beats {count}{RESET}")

            # Show if strategy was followed
            if self.is_positive_trace():
                print(f"\n{GREEN}Strategy followed correctly!{RESET}")
            else:
                print(f"\n{RED}Strategy was violated!{RESET}")

    def _card_name(self, value):
        """Get display name for a card value."""
        if value == 1:
            return 'A'
        else:
            return str(value)

    def get_key(self):
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def play(self):
        """Main game loop for a single game instance."""
        self.display()

        quit_session = False
        quit_mid_game = False

        while not self.game_over:
            key = self.get_key()

            if key == 'q':
                quit_session = True
                quit_mid_game = True
                break
            elif key == 'h':
                self.hit()
                self.display()
            elif key == 's':
                self.stand()
                self.display()

        if not quit_mid_game:
            self.display()
            trace_file = self.save_trace()
            print(f"\nTrace saved to: {trace_file}")

        return quit_session


def generate_random_configs(
    num_configs: int = 3,
    strategies: list = None,
    seed: int = None
) -> list:
    """
    Generate blackjack configurations for different strategies.

    Args:
        num_configs: Number of configurations to generate
        strategies: List of strategies to use (default: all strategies)
        seed: Random seed for reproducibility

    Returns:
        List of configuration dictionaries compatible with spec_generator
    """
    if seed is not None:
        random.seed(seed)

    if strategies is None:
        strategies = [Strategy.THRESHOLD, Strategy.CONSERVATIVE, Strategy.BASIC]

    configs = []
    for i, strategy in enumerate(strategies[:num_configs]):
        game = BlackjackGame(strategy=strategy)
        config = game.get_config(name=f"config_{i + 1}")
        configs.append(config)

    # If more configs requested than strategies, cycle through
    while len(configs) < num_configs:
        strategy = strategies[len(configs) % len(strategies)]
        game = BlackjackGame(strategy=strategy)
        config = game.get_config(name=f"config_{len(configs) + 1}")
        configs.append(config)

    return configs


def wait_for_key():
    """Wait for any keypress."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    """Run the game session."""
    parser = argparse.ArgumentParser(description="Blackjack Game for TSL_f Mining")
    parser.add_argument('--strategy', choices=['threshold', 'conservative', 'basic'],
                        default='threshold',
                        help='Strategy for pos/neg classification (default: threshold)')
    parser.add_argument('--gen', type=int, metavar='N',
                        help='Generate N positive and N negative traces automatically')
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='Output directory for traces')
    parser.add_argument('--config', type=int, metavar='N',
                        help='Generate N configurations as JSON for synthesis')
    parser.add_argument('--config-output', type=str, metavar='FILE',
                        help='Output JSON file for --config mode (default: stdout)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Config generation mode
    if args.config is not None:
        strategy_list = None
        if args.strategy:
            strategy_map = {
                'threshold': Strategy.THRESHOLD,
                'conservative': Strategy.CONSERVATIVE,
                'basic': Strategy.BASIC,
            }
            strategy_list = [strategy_map[args.strategy]]

        configs = generate_random_configs(
            num_configs=args.config,
            strategies=strategy_list,
            seed=args.seed
        )

        output = json.dumps(configs, indent=2)

        if args.config_output:
            with open(args.config_output, 'w') as f:
                f.write(output)
            print(f"Generated {len(configs)} configurations to: {args.config_output}", file=sys.stderr)
        else:
            print(output)
        return

    # Parse strategy
    strategy_map = {
        'threshold': Strategy.THRESHOLD,
        'conservative': Strategy.CONSERVATIVE,
        'basic': Strategy.BASIC,
    }
    strategy = strategy_map[args.strategy]

    # Create session directory
    if args.output:
        session_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/blackjack/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    if args.gen is not None:
        # Automatic trace generation mode
        print("=" * 50)
        print("BLACKJACK - TRACE GENERATION MODE")
        print("=" * 50)
        print(f"  Strategy: {strategy.value}")
        print(f"  Target: {args.gen} positive + {args.gen} negative traces")
        print(f"  Output: {session_dir}")

        game = BlackjackGame(session_dir=session_dir, strategy=strategy)
        game.generate_traces(args.gen)

        print(f"\nAll traces saved to: {session_dir}")
    else:
        # Interactive mode
        print("=" * 50)
        print("BLACKJACK - Interactive Mode")
        print("=" * 50)
        print(f"  Strategy: {strategy.value}")
        print("\nPositive traces = strategy followed")
        print("Negative traces = strategy violated")
        print("\nControls:")
        print("  h - Hit (take a card)")
        print("  s - Stand")
        print("  q - Quit")
        print("\n(Press any key to start)")

        wait_for_key()

        while True:
            game = BlackjackGame(session_dir=session_dir, strategy=strategy)
            quit_session = game.play()

            if quit_session:
                print("\n" + "=" * 50)
                print("Session ended. Thanks for playing!")
                break
            else:
                print("\n" + "=" * 50)
                print("Press any key to play again, or Ctrl+C to exit...")
                wait_for_key()


if __name__ == "__main__":
    main()
