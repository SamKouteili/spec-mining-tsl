#!/usr/bin/env python3
"""
Test Harness for Blackjack Evaluation

Evaluates models on:
1. Win rate: How many games out of N are won
2. Strategy adherence: How many decisions follow the target strategy

Supports BC, DT, Alergia, and TSLf models.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass


# ============== Strategy Definitions ==============

DEALER_STAND_THRESHOLD = 17
WEAK_DEALER_MAX = 6
STAND_VS_WEAK_MIN = 13
BUST_THRESHOLD = 21


def should_hit_threshold(count: int, dealer: int) -> bool:
    """THRESHOLD strategy: hit if count < 17."""
    return count < DEALER_STAND_THRESHOLD


def should_hit_conservative(count: int, dealer: int) -> bool:
    """CONSERVATIVE strategy: stand at 12+ vs weak dealer, otherwise threshold."""
    if count >= 17:
        return False
    elif count >= 12 and 2 <= dealer <= 6:
        return False
    elif count <= 11:
        return True
    else:
        return True


def should_hit_basic(count: int, dealer: int) -> bool:
    """BASIC strategy: stand at 17+, hit at 11-, stand at 13+ vs weak dealer.

    Note: dealer 2-6 is weak. Ace (1) is NOT weak.
    """
    if count >= 17:
        return False
    elif count <= 11:
        return True
    elif count >= 13 and 2 <= dealer <= WEAK_DEALER_MAX:
        return False
    else:
        return True


STRATEGY_FUNCTIONS = {
    "threshold": should_hit_threshold,
    "conservative": should_hit_conservative,
    "basic": should_hit_basic,
}


# ============== Result Dataclasses ==============

@dataclass
class GameResult:
    """Result of a single game."""
    won: bool
    followed_strategy: bool
    num_decisions: int
    strategy_violations: int
    final_count: int
    dealer_count: int


@dataclass
class EvalResult:
    """Aggregate evaluation result."""
    games_played: int
    games_won: int
    games_strategy_followed: int
    total_decisions: int
    total_violations: int

    @property
    def win_rate(self) -> float:
        return self.games_won / self.games_played if self.games_played > 0 else 0.0

    @property
    def strategy_adherence(self) -> float:
        return self.games_strategy_followed / self.games_played if self.games_played > 0 else 0.0

    @property
    def decision_accuracy(self) -> float:
        return 1.0 - (self.total_violations / self.total_decisions) if self.total_decisions > 0 else 0.0


# ============== Blackjack Game Simulation ==============

class BlackjackSimulator:
    """Simulates blackjack games for evaluation."""

    def __init__(self, strategy: str, seed: Optional[int] = None):
        self.strategy = strategy
        self.should_hit = STRATEGY_FUNCTIONS[strategy]
        self.rng = random.Random(seed)

    def _create_deck(self) -> List[int]:
        """Create and shuffle a deck."""
        deck = []
        for _ in range(4):
            deck.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        self.rng.shuffle(deck)
        return deck

    def _calculate_count(self, cards: List[int]) -> int:
        """Calculate hand value, treating Ace optimally."""
        total = sum(cards)
        if 1 in cards and total + 10 <= BUST_THRESHOLD:
            total += 10
        return total

    def _get_state_features(self, count: int, dealer: int) -> Dict:
        """Build state features dict for model."""
        state = {"count": count, "dealer": dealer, "stood": False}

        if self.strategy == "threshold":
            state["standThreshold"] = DEALER_STAND_THRESHOLD
        elif self.strategy == "conservative":
            state["standThreshold"] = DEALER_STAND_THRESHOLD
            state["standVsWeakMin"] = 12  # Conservative stands at 12+ vs weak dealer
            # Derived boolean: is dealer weak (2-6)?
            state["isWeakDealer"] = 2 <= dealer <= WEAK_DEALER_MAX
        else:  # basic
            state["standThreshold"] = DEALER_STAND_THRESHOLD
            state["standVsWeakMin"] = STAND_VS_WEAK_MIN
            state["isWeakDealer"] = 2 <= dealer <= WEAK_DEALER_MAX

        return state

    def play_game(self, get_action: Callable[[Dict], int]) -> GameResult:
        """
        Play one game using the provided action function.

        Args:
            get_action: Function that takes state dict and returns 0 (hit) or 1 (stand)

        Returns:
            GameResult with win/loss and strategy adherence info
        """
        deck = self._create_deck()

        # Deal initial hands
        player_cards = [deck.pop(), deck.pop()]
        dealer_cards = [deck.pop(), deck.pop()]
        dealer_showing = dealer_cards[0]

        num_decisions = 0
        strategy_violations = 0
        busted = False
        stood = False

        # Player's turn
        while not stood and not busted:
            count = self._calculate_count(player_cards)

            # Check what strategy says
            strategy_says_hit = self.should_hit(count, dealer_showing)

            # Get model's action
            state = self._get_state_features(count, dealer_showing)
            action = get_action(state)  # 0=hit, 1=stand

            num_decisions += 1

            # Check strategy adherence
            if (action == 0 and not strategy_says_hit) or (action == 1 and strategy_says_hit):
                strategy_violations += 1

            if action == 1:  # Stand
                stood = True
            else:  # Hit
                if len(deck) == 0:
                    deck = self._create_deck()
                player_cards.append(deck.pop())
                count = self._calculate_count(player_cards)
                if count > BUST_THRESHOLD:
                    busted = True

        player_count = self._calculate_count(player_cards)

        # Dealer's turn (only if player didn't bust)
        if not busted:
            while self._calculate_count(dealer_cards) < DEALER_STAND_THRESHOLD:
                if len(deck) == 0:
                    deck = self._create_deck()
                dealer_cards.append(deck.pop())

        dealer_count = self._calculate_count(dealer_cards)

        # Determine winner
        if busted:
            won = False
        elif dealer_count > BUST_THRESHOLD:
            won = True
        elif player_count > dealer_count:
            won = True
        elif player_count == dealer_count:
            won = False  # Push counts as not won for simplicity
        else:
            won = False

        return GameResult(
            won=won,
            followed_strategy=(strategy_violations == 0),
            num_decisions=num_decisions,
            strategy_violations=strategy_violations,
            final_count=player_count,
            dealer_count=dealer_count
        )

    def evaluate(self, get_action: Callable[[Dict], int], num_games: int,
                 seed: Optional[int] = None) -> EvalResult:
        """
        Evaluate a model over multiple games.

        Args:
            get_action: Function that takes state dict and returns 0 (hit) or 1 (stand)
            num_games: Number of games to play
            seed: Random seed for reproducibility

        Returns:
            EvalResult with aggregate statistics
        """
        if seed is not None:
            self.rng = random.Random(seed)

        total_won = 0
        total_followed = 0
        total_decisions = 0
        total_violations = 0

        for _ in range(num_games):
            result = self.play_game(get_action)
            if result.won:
                total_won += 1
            if result.followed_strategy:
                total_followed += 1
            total_decisions += result.num_decisions
            total_violations += result.strategy_violations

        return EvalResult(
            games_played=num_games,
            games_won=total_won,
            games_strategy_followed=total_followed,
            total_decisions=total_decisions,
            total_violations=total_violations
        )


# ============== Model Adapters ==============

def make_bc_action_fn(model, strategy: str) -> Callable[[Dict], int]:
    """Create action function for BC model."""
    import torch
    try:
        from .baselines.bc_baseline import extract_features
    except ImportError:
        from baselines.bc_baseline import extract_features

    def get_action(state: Dict) -> int:
        features = extract_features(state, strategy)
        features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
        features_tensor = torch.tensor(features_arr, dtype=torch.float32)
        action = model.predict_action(features_tensor).item()
        return action

    return get_action


def make_dt_action_fn(model, strategy: str) -> Callable[[Dict], int]:
    """Create action function for DT model."""
    try:
        from .baselines.bc_baseline import extract_features
    except ImportError:
        from baselines.bc_baseline import extract_features

    def get_action(state: Dict) -> int:
        features = extract_features(state, strategy)
        features_arr = np.array([features], dtype=np.float32)
        action = model.predict(features_arr)[0]
        return action

    return get_action


def make_alergia_action_fn(model, strategy: str, use_sampling: bool = False) -> Callable[[Dict], int]:
    """Create action function for Alergia SMM model."""
    try:
        from .baselines.alergia_baseline import encode_observation, get_smm_action
    except ImportError:
        from baselines.alergia_baseline import encode_observation, get_smm_action

    def get_action(state: Dict) -> int:
        # Reset model to initial state at start of each game would need tracking
        # For now, we reset per decision (stateless approximation)
        model.reset_to_initial()

        obs = encode_observation(
            state["count"],
            strategy,
            state.get("standThreshold", 17),
            state.get("standVsWeakMin", 13),
            state.get("isWeakDealer", False)
        )

        action_str = get_smm_action(model, obs, use_sampling)

        if action_str == "STAND":
            return 1
        elif action_str == "HIT":
            return 0
        else:
            # Default to hit if unknown (safer for not busting early)
            return 0 if state["count"] < 17 else 1

    return get_action


def make_random_action_fn(strategy: str) -> Callable[[Dict], int]:
    """Create random action function (baseline)."""
    rng = random.Random(42)

    def get_action(state: Dict) -> int:
        return rng.randint(0, 1)

    return get_action


def make_oracle_action_fn(strategy: str) -> Callable[[Dict], int]:
    """Create oracle action function that perfectly follows strategy."""
    should_hit = STRATEGY_FUNCTIONS[strategy]

    def get_action(state: Dict) -> int:
        if should_hit(state["count"], state["dealer"]):
            return 0  # Hit
        else:
            return 1  # Stand

    return get_action


# ============== CLI ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test harness for blackjack evaluation")
    parser.add_argument("--strategy", choices=["threshold", "conservative", "basic"],
                        default="threshold", help="Strategy to evaluate")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Testing oracle (perfect strategy) on {args.strategy} strategy...")
    simulator = BlackjackSimulator(args.strategy, seed=args.seed)

    # Test oracle
    oracle_fn = make_oracle_action_fn(args.strategy)
    result = simulator.evaluate(oracle_fn, args.num_games, seed=args.seed)

    print(f"\nOracle Results:")
    print(f"  Win rate: {result.win_rate:.1%} ({result.games_won}/{result.games_played})")
    print(f"  Strategy adherence: {result.strategy_adherence:.1%}")
    print(f"  Decision accuracy: {result.decision_accuracy:.1%}")

    # Test random
    random_fn = make_random_action_fn(args.strategy)
    result = simulator.evaluate(random_fn, args.num_games, seed=args.seed)

    print(f"\nRandom Results:")
    print(f"  Win rate: {result.win_rate:.1%} ({result.games_won}/{result.games_played})")
    print(f"  Strategy adherence: {result.strategy_adherence:.1%}")
    print(f"  Decision accuracy: {result.decision_accuracy:.1%}")
