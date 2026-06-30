import json
import math
import random
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RouteSpec:
    """
    Parsed from your matatu JSON. One entry per (route_number, destination) pair
    after branch expansion — branching is handled at generation time via weights,
    not by pre-expanding into separate routes.
    """
    route_number: str
    pickup_latlng: Tuple[float, float]       # (lat, lon)
    destinations: List[Tuple[float, float]]  # [(lat, lon), ...] ordered as in JSON
    dest_names: List[str]                    # for logging/debugging only


@dataclass
class PPOAction:
    """
    One action vector decoded into structured per-route parameters.
    PPO outputs raw floats; this class holds the post-processed values
    ready for scenario generation.
    """
    # Parallel lists indexed by route
    vehicle_counts: List[int]          # discretized from raw PPO output
    departure_spreads: List[float]     # std dev in seconds
    branch_weights: List[List[float]]  # per-route softmax weights over destinations


@dataclass 
class NairobiRushHourWindows:
    """
    Nairobi-specific rush hour windows in seconds after midnight.
    Adjust based on actual TomTom scoring windows once confirmed.
    Morning: 06:30 - 09:30
    Evening: 16:00 - 19:30
    """
    morning_peak_center: float = 7.5 * 3600   # 07:30 as seconds
    morning_window_start: float = 6.5 * 3600  # 06:30
    morning_window_end: float = 9.5 * 3600    # 09:30

    evening_peak_center: float = 17.5 * 3600  # 17:30
    evening_window_start: float = 16.0 * 3600 # 16:00
    evening_window_end: float = 19.5 * 3600   # 19:30


# ---------------------------------------------------------------------------
# Route JSON parser
# ---------------------------------------------------------------------------

def load_matatu_routes(json_path: str) -> List[RouteSpec]:
    """
    Parses your route JSON into RouteSpec objects.
    Uses destination_latlng as authoritative (not destination_hexid).
    Filters out entries with missing/null coordinates.
    """
    with open(json_path, encoding='utf8') as f:
        data = json.load(f)

    routes = []
    for entry in data.get('non_null_objects', []):
        route_number = entry.get('route_number', 'unknown')
        pickup = entry.get('pickup_point', {})

        pickup_latlng = pickup.get('pickup_latlng', {})
        p_lat = pickup_latlng.get('latitude')
        p_lon = pickup_latlng.get('longitude')
        if p_lat is None or p_lon is None:
            continue  # skip routes with no valid pickup coords

        destinations = []
        dest_names = []
        for dest in entry.get('destinations', []):
            d_latlng = dest.get('destination_latlng', {})
            d_lat = d_latlng.get('latitude')
            d_lon = d_latlng.get('longitude')
            if d_lat is None or d_lon is None:
                continue
            # Skip destinations that are identical to pickup (data artifact)
            if abs(d_lat - p_lat) < 1e-6 and abs(d_lon - p_lon) < 1e-6:
                continue
            destinations.append((d_lat, d_lon))
            dest_names.append(dest.get('destination', ''))

        if not destinations:
            continue  # route has no valid destinations after filtering

        routes.append(RouteSpec(
            route_number=route_number,
            pickup_latlng=(p_lat, p_lon),
            destinations=destinations,
            dest_names=dest_names,
        ))

    return routes


# ---------------------------------------------------------------------------
# Map building lookup (nearest road → building snap)
# ---------------------------------------------------------------------------

class ABStreetMapIndex:
    """
    Caches building positions from the dumped map JSON and snaps
    (lat, lon) coordinates to the nearest building ID.
    Called once at startup, not per-episode.
    """

    def __init__(self, map_json_path: str):
        with open(map_json_path, encoding='utf8') as f:
            map_data = json.load(f)

        # Build a list of (building_id, lat, lon) from map buildings
        self.buildings = []
        for b in map_data.get('buildings', []):
            pos = b.get('label_pt', b.get('polygon', {}).get('pts', [{}])[0])
            lat = pos.get('y')   # AB Street uses y=lat, x=lon in map coords
            lon = pos.get('x')
            if lat is not None and lon is not None:
                self.buildings.append((b['id'], lat, lon))

        if not self.buildings:
            raise ValueError(
                "No buildings found in map JSON — check the dump format. "
                "Run: cargo run --bin cli -- dump-json <map.bin> > map.json"
            )

    def nearest_building(self, lat: float, lon: float) -> int:
        """
        Returns the A/B Street building ID nearest to (lat, lon).
        Simple Euclidean distance in degree-space — sufficient for
        within-Nairobi distances (no projection needed at this scale).
        """
        best_id, best_dist = None, float('inf')
        for b_id, b_lat, b_lon in self.buildings:
            dist = (b_lat - lat) ** 2 + (b_lon - lon) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = b_id
        return best_id


# ---------------------------------------------------------------------------
# Action space encoder/decoder
# ---------------------------------------------------------------------------

class MatatuActionSpace:
    """
    Manages the mapping between PPO's flat float vector and the structured
    per-route action parameters.

    Action vector layout (flat, for PPO output layer sizing):
      For route i with D_i destinations:
        [vehicle_count_i, departure_spread_i, branch_w_i_0, ..., branch_w_i_{D-1}]
      Concatenated across all routes.

    Bounds (for PPO action clipping / normalization):
      vehicle_count:     raw float → sigmoid → scale to [min_vehicles, max_vehicles]
      departure_spread:  raw float → softplus → scale to [min_spread_s, max_spread_s]
      branch_weights:    raw floats → softmax (per route)
    """

    MIN_VEHICLES = 1
    MAX_VEHICLES = 30        # max matatus per route per rush-hour window
    MIN_SPREAD_S = 60.0      # 1 minute minimum spread
    MAX_SPREAD_S = 45 * 60   # 45 minute maximum spread

    def __init__(self, routes: List[RouteSpec]):
        self.routes = routes
        # Precompute slice indices into the flat action vector
        self.slices: List[Tuple[int, int, int]] = []  # (count_idx, spread_idx, weights_start)
        idx = 0
        for route in routes:
            count_idx = idx
            spread_idx = idx + 1
            weights_start = idx + 2
            self.slices.append((count_idx, spread_idx, weights_start))
            idx += 2 + len(route.destinations)
        self.total_dims = idx

    def decode(self, raw_action: np.ndarray) -> PPOAction:
        """
        Converts PPO's raw float output into structured PPOAction.
        raw_action shape: (total_dims,)
        """
        assert len(raw_action) == self.total_dims, (
            f"Action vector length {len(raw_action)} != expected {self.total_dims}"
        )

        vehicle_counts = []
        departure_spreads = []
        branch_weights = []

        for i, route in enumerate(self.routes):
            count_idx, spread_idx, weights_start = self.slices[i]
            n_dests = len(route.destinations)

            # vehicle count: sigmoid → integer in [MIN, MAX]
            raw_count = raw_action[count_idx]
            count = int(self.MIN_VEHICLES + (self.MAX_VEHICLES - self.MIN_VEHICLES)
                        * (1.0 / (1.0 + math.exp(-raw_count))))
            vehicle_counts.append(count)

            # departure spread: softplus → seconds in [MIN, MAX]
            raw_spread = raw_action[spread_idx]
            softplus = math.log(1 + math.exp(raw_spread))
            spread = min(
                self.MAX_SPREAD_S,
                self.MIN_SPREAD_S + softplus * (self.MAX_SPREAD_S - self.MIN_SPREAD_S) / 10.0
            )
            departure_spreads.append(spread)

            # branch weights: softmax over destination logits
            raw_weights = raw_action[weights_start:weights_start + n_dests]
            exp_w = np.exp(raw_weights - np.max(raw_weights))  # numerically stable
            weights = (exp_w / exp_w.sum()).tolist()
            branch_weights.append(weights)

        return PPOAction(
            vehicle_counts=vehicle_counts,
            departure_spreads=departure_spreads,
            branch_weights=branch_weights,
        )


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

class MatatuScenarioGenerator:
    """
    Generates an A/B Street scenario JSON from a PPOAction.
    One call per episode, before /sim/load.

    Matatus: Drive-mode trips strictly along route corridors,
             departure times clustered around Nairobi rush-hour peaks.
    Background traffic: Drive/Walk/Bike trips sampled from the building
             pool, volume controlled by a separate background_scale factor
             (not part of PPO action — treat as a fixed hyperparameter
             or a separate outer-loop parameter for now).
    """

    def __init__(
        self,
        routes: List[RouteSpec],
        map_index: ABStreetMapIndex,
        action_space: MatatuActionSpace,
        rush_hours: Optional[NairobiRushHourWindows] = None,
        background_vehicles: int = 200,
        rng_seed: Optional[int] = None,
    ):
        self.routes = routes
        self.map_index = map_index
        self.action_space = action_space
        self.rush = rush_hours or NairobiRushHourWindows()
        self.background_vehicles = background_vehicles
        self.rng = random.Random(rng_seed)

    def _rush_hour_departure(
        self,
        peak_center: float,
        spread_seconds: float,
        window_start: float,
        window_end: float,
    ) -> float:
        """
        Samples a departure time (seconds after midnight) from a Gaussian
        centered on peak_center with std=spread_seconds, clipped to window.
        """
        for _ in range(20):  # rejection sample within window
            t = self.rng.gauss(peak_center, spread_seconds)
            if window_start <= t <= window_end:
                return t
        # Fallback: uniform within window if rejection keeps missing
        return self.rng.uniform(window_start, window_end)

    def _matatu_people(self, action: PPOAction) -> list:
        """Generates matatu Drive-mode trip people from action parameters."""
        people = []

        for i, route in enumerate(self.routes):
            n_vehicles = action.vehicle_counts[i]
            spread = action.departure_spreads[i]
            weights = action.branch_weights[i]

            for _ in range(n_vehicles):
                # Sample destination branch according to PPO-learned weights
                dest_idx = self.rng.choices(
                    range(len(route.destinations)), weights=weights, k=1
                )[0]
                dest_latlng = route.destinations[dest_idx]

                # Snap latlng to nearest building ID in A/B Street map
                origin_bldg = self.map_index.nearest_building(*route.pickup_latlng)
                dest_bldg = self.map_index.nearest_building(*dest_latlng)

                if origin_bldg == dest_bldg:
                    continue  # snapped to same building — skip this vehicle

                # Each matatu runs both morning and evening rush
                trips = []
                for peak_center, window_start, window_end in [
                    (self.rush.morning_peak_center,
                     self.rush.morning_window_start,
                     self.rush.morning_window_end),
                    (self.rush.evening_peak_center,
                     self.rush.evening_window_start,
                     self.rush.evening_window_end),
                ]:
                    departure = self._rush_hour_departure(
                        peak_center, spread, window_start, window_end
                    )
                    trips.append({
                        'departure': departure,
                        'destination': {'TripEndpoint': {'Bldg': dest_bldg}},
                        'mode': 'Drive',
                        'purpose': 'Transit',  # closest purpose tag for matatus
                    })

                # Sort trips by departure time (A/B Street requirement)
                trips.sort(key=lambda t: t['departure'])

                people.append({
                    'origin': {'TripEndpoint': {'Bldg': origin_bldg}},
                    'trips': trips,
                })

        return people

    def _background_people(self, all_building_ids: List[int]) -> list:
        """
        Generates background Drive/Walk/Bike demand on the same network.
        Volume is fixed (not PPO-controlled this phase) — treat as env hyperparameter.
        """
        people = []
        modes = ['Drive', 'Drive', 'Drive', 'Walk', 'Bike']  # realistic Nairobi mode split skew

        for _ in range(self.background_vehicles):
            origin = self.rng.choice(all_building_ids)
            dest = self.rng.choice(all_building_ids)
            if origin == dest:
                continue

            departure = self._rush_hour_departure(
                self.rush.morning_peak_center,
                20 * 60,  # 20min spread for background — wider than matatu peaks
                self.rush.morning_window_start,
                self.rush.morning_window_end,
            )
            people.append({
                'origin': {'TripEndpoint': {'Bldg': origin}},
                'trips': [{
                    'departure': departure,
                    'destination': {'TripEndpoint': {'Bldg': dest}},
                    'mode': self.rng.choice(modes),
                    'purpose': 'Shopping',
                }],
            })

        return people

    def generate(
        self,
        raw_ppo_action: np.ndarray,
        output_path: str,
        scenario_name: str = 'matatu_rush',
    ) -> dict:
        """
        Main entry point. Decodes PPO action, generates scenario JSON,
        writes to output_path, returns the scenario dict for inspection.
        """
        action = self.action_space.decode(raw_ppo_action)
        all_building_ids = [b[0] for b in self.map_index.buildings]

        people = self._matatu_people(action)
        people += self._background_people(all_building_ids)

        # Shuffle so matatu and background trips are interleaved in the file
        self.rng.shuffle(people)

        scenario = {
            'scenario_name': scenario_name,
            'people': people,
        }

        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(scenario, f, indent=2)

        return scenario


# ---------------------------------------------------------------------------
# Episode runner — wires generator into the A/B Street headless API loop
# ---------------------------------------------------------------------------

class MatatuEpisodeRunner:
    """
    Runs one full PPO episode:
      1. Decode action → generate scenario JSON
      2. Import scenario (subprocess) → /sim/load → /sim/goto-time
      3. Pull /data/get-finished-trips + /data/all-trip-time-lower-bounds
      4. Compute congestion reward
      5. Return (reward, episode_stats) to PPO training loop
    """

    def __init__(
        self,
        generator: MatatuScenarioGenerator,
        map_bin_path: str,
        scenario_bin_dir: str,
        host: str = 'http://localhost:1234',
        simulate_until: str = '20:00:00',  # covers both rush windows
    ):
        self.generator = generator
        self.map_bin_path = map_bin_path
        self.scenario_bin_dir = scenario_bin_dir
        self.host = host
        self.simulate_until = simulate_until

    def _import_scenario(self, scenario_json_path: str, scenario_bin_path: str):
        """
        Compiles scenario JSON → binary using A/B Street CLI.
        Blocking subprocess — run before /sim/load.
        """
        import subprocess
        result = subprocess.run([
            'cargo', 'run', '--release', '--bin', 'cli', '--',
            'import-scenario',
            f'--input={scenario_json_path}',
            f'--map={self.map_bin_path}',
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Scenario import failed:\n{result.stderr}"
            )

    def _compute_congestion_reward(
        self,
        finished_trips: list,
        lower_bounds: dict,
        rush_windows: NairobiRushHourWindows,
    ) -> float:
        """
        Reward = negative mean congestion ratio across rush-hour trips.
        congestion_ratio = (actual_duration - lower_bound) / lower_bound
        Only trips that *departed* within rush-hour windows count —
        background trips outside those windows are excluded from reward
        signal to keep the TomTom alignment meaningful.
        """
        ratios = []
        for trip in finished_trips:
            if trip.get('duration') is None:
                continue  # cancelled trip

            trip_id = str(trip['id'])
            lb = lower_bounds.get(trip_id)
            if lb is None or lb <= 0:
                continue

            actual = trip['duration']
            ratio = (actual - lb) / lb
            ratios.append(ratio)

        if not ratios:
            return 0.0

        mean_congestion = sum(ratios) / len(ratios)
        return -mean_congestion  # PPO maximizes: less congestion = higher reward

    def run_episode(
        self,
        raw_ppo_action: np.ndarray,
        episode_id: int,
        tmp_dir: str = '/tmp',
    ) -> Tuple[float, dict]:
        """
        Returns (reward, stats_dict).
        stats_dict contains raw metrics for logging/TomTom comparison.
        """
        scenario_json = f"{tmp_dir}/matatu_ep{episode_id}.json"
        scenario_bin = f"{self.scenario_bin_dir}/matatu_ep{episode_id}.bin"

        # 1. Generate scenario from PPO action
        scenario = self.generator.generate(
            raw_ppo_action,
            output_path=scenario_json,
            scenario_name=f'matatu_ep{episode_id}',
        )

        # 2. Import + load
        self._import_scenario(scenario_json, scenario_bin)
        requests.post(f"{self.host}/sim/load", json={
            'scenario': scenario_bin,
            'modifiers': [],
            'edits': None,
        })

        # 3. Run simulation to cover both rush windows
        requests.get(f"{self.host}/sim/goto-time", params={'t': self.simulate_until})

        # 4. Pull results — use batch lower bounds (one API call, not N)
        finished_trips = requests.get(
            f"{self.host}/data/get-finished-trips"
        ).json()
        lower_bounds_raw = requests.get(
            f"{self.host}/data/all-trip-time-lower-bounds"
        ).json()

        # lower_bounds comes back as [[trip_id, seconds], ...]
        lower_bounds = {str(entry[0]): entry[1] for entry in lower_bounds_raw}

        # 5. Compute reward
        reward = self._compute_congestion_reward(
            finished_trips,
            lower_bounds,
            self.generator.rush,
        )

        stats = {
            'episode_id': episode_id,
            'n_people': len(scenario['people']),
            'n_finished_trips': sum(
                1 for t in finished_trips if t.get('duration') is not None
            ),
            'n_cancelled_trips': sum(
                1 for t in finished_trips if t.get('duration') is None
            ),
            'reward': reward,
        }

        return reward, stats