"""
calibration/mircostructure.py

Purpose: Calibrates market microstructure parameters from historical order book data:

1. Order arrival intensities

2. Fill probabilities by queue position

3. Spread regime transition probabilities These feed directly into the LOB simulator.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class LOBParameterCalibrator:
    """
    Calibrates market microstructure parameters from historical order book data.

    This class processes a stream of order book events to estimate:
    1. Order Arrival Intensities (lambda) for different order types.
    2. Fill Probabilities for limit orders based on their queue position (depth).
    3. Transition Probabilities between different bid-ask spread regimes.

    These parameters are essential for building realistic LOB simulators.
    """

    def __init__(self, lob_data: pd.DataFrame, tick_size: float = 0.01):
        """
        Initializes the calibrator with LOB data.

        Args:
            lob_data (pd.DataFrame): DataFrame containing LOB events. Must include
                columns: 'timestamp', 'event_type', 'order_id', 'side',
                'price', 'quantity', 'best_bid', 'best_ask'.
            tick_size (float): The minimum price increment of the asset.
        """
        if not isinstance(lob_data, pd.DataFrame):
            raise TypeError("lob_data must be a pandas DataFrame.")

        required_cols = {'timestamp', 'event_type', 'order_id', 'side', 'price',
                         'quantity', 'best_bid', 'best_ask'}
        if not required_cols.issubset(lob_data.columns):
            raise ValueError(f"lob_data is missing required columns. "
                             f"Required: {required_cols}")

        self.data = lob_data.sort_values('timestamp').copy()
        self.tick_size = tick_size
        self.total_time_seconds = (self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]).total_seconds()
        
        # Calibrated parameters will be stored here
        self.arrival_intensities = {}
        self.fill_probabilities = {}
        self.spread_transition_matrix = None
        
        print(f"Initialized calibrator with {len(self.data)} events.")
        print(f"Total observation time: {self.total_time_seconds:.2f} seconds.")

    def calibrate_all(self, max_depth: int = 10, n_spread_regimes: int = 3):
        """
        Runs all calibration methods to estimate the parameters.

        Args:
            max_depth (int): The maximum queue depth (in ticks) to consider for
                             fill probability calculation.
            n_spread_regimes (int): The number of regimes to discretize the
                                    bid-ask spread into.
        """
        print("\n--- Starting Calibration ---")
        self._calculate_arrival_intensities()
        self._calculate_fill_probabilities(max_depth)
        self._calculate_spread_regime_transitions(n_spread_regimes)
        print("--- Calibration Complete ---")
        return self.get_parameters()

    def _calculate_arrival_intensities(self):
        """Calculates the arrival rate (lambda) for different order types."""
        print("Calibrating order arrival intensities...")
        # Limit Orders
        limit_buy_count = len(self.data[(self.data['event_type'] == 'LIMIT') & (self.data['side'] == 'BID')])
        limit_sell_count = len(self.data[(self.data['event_type'] == 'LIMIT') & (self.data['side'] == 'ASK')])

        # Market Orders (inferred from trades that cross the spread)
        market_buy_count = len(self.data[(self.data['event_type'] == 'TRADE') & (self.data['price'] >= self.data['best_ask'])])
        market_sell_count = len(self.data[(self.data['event_type'] == 'TRADE') & (self.data['price'] <= self.data['best_bid'])])

        # Cancellations
        cancel_count = len(self.data[self.data['event_type'] == 'CANCEL'])

        if self.total_time_seconds > 0:
            self.arrival_intensities = {
                'limit_buy': limit_buy_count / self.total_time_seconds,
                'limit_sell': limit_sell_count / self.total_time_seconds,
                'market_buy': market_buy_count / self.total_time_seconds,
                'market_sell': market_sell_count / self.total_time_seconds,
                'cancel': cancel_count / self.total_time_seconds,
            }
        else:
            self.arrival_intensities = {k: 0 for k in ['limit_buy', 'limit_sell', 'market_buy', 'market_sell', 'cancel']}
        
        print("...intensities calculated.")

    def _calculate_fill_probabilities(self, max_depth: int):
        """Calculates the probability of an order being filled given its queue position."""
        print(f"Calibrating fill probabilities (max_depth={max_depth})...")
        limit_orders = {} # {order_id: initial_depth}
        
        orders_placed_at_depth = defaultdict(int)
        orders_filled_at_depth = defaultdict(int)

        # Track active limit orders to determine their outcomes
        for _, row in self.data.iterrows():
            if row['event_type'] == 'LIMIT':
                if row['side'] == 'BID':
                    depth = round((row['best_ask'] - row['price']) / self.tick_size)
                else: # ASK
                    depth = round((row['price'] - row['best_bid']) / self.tick_size)
                
                if 0 <= depth < max_depth:
                    limit_orders[row['order_id']] = depth
                    orders_placed_at_depth[depth] += 1
            
            elif row['event_type'] == 'TRADE':
                # This assumes 'order_id' for a trade refers to the resting limit order that was hit
                if row['order_id'] in limit_orders:
                    initial_depth = limit_orders.pop(row['order_id'])
                    orders_filled_at_depth[initial_depth] += 1

            elif row['event_type'] == 'CANCEL':
                if row['order_id'] in limit_orders:
                    # Remove from active orders if cancelled
                    limit_orders.pop(row['order_id'])

        # Calculate probabilities
        for depth in range(max_depth):
            placed_count = orders_placed_at_depth[depth]
            filled_count = orders_filled_at_depth[depth]
            self.fill_probabilities[depth] = filled_count / placed_count if placed_count > 0 else 0
        
        print("...fill probabilities calculated.")

    def _calculate_spread_regime_transitions(self, n_regimes: int):
        """Models spread dynamics as a Markov chain and finds the transition matrix."""
        print(f"Calibrating spread regime transitions (n_regimes={n_regimes})...")
        spreads = (self.data['best_ask'] - self.data['best_bid']) / self.tick_size
        spreads = spreads[spreads > 0].dropna()

        if len(spreads) < 2:
            print("...not enough spread data to calculate transitions.")
            return

        # Define regime boundaries using quantiles
        quantiles = np.linspace(0, 1, n_regimes + 1)
        regime_bins = spreads.quantile(quantiles).to_numpy()
        regime_bins[0] = 0 # Ensure the first bin starts at 0
        
        # Discretize spreads into regimes
        spread_regimes = pd.cut(spreads, bins=regime_bins, labels=False, include_lowest=True)
        spread_regimes = spread_regimes.dropna().astype(int)

        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(spread_regimes) - 1):
            from_regime = spread_regimes.iloc[i]
            to_regime = spread_regimes.iloc[i+1]
            transition_counts[from_regime][to_regime] += 1
            
        # Create transition matrix
        matrix = pd.DataFrame(0.0, index=range(n_regimes), columns=range(n_regimes))
        for from_regime, transitions in transition_counts.items():
            total_transitions = sum(transitions.values())
            for to_regime, count in transitions.items():
                if total_transitions > 0:
                    matrix.loc[from_regime, to_regime] = count / total_transitions
        
        self.spread_transition_matrix = matrix
        self.spread_regime_bins = regime_bins
        print("...spread transitions calculated.")

    def get_parameters(self) -> dict:
        """Returns the calibrated parameters as a dictionary."""
        return {
            "arrival_intensities": self.arrival_intensities,
            "fill_probabilities_by_depth": self.fill_probabilities,
            "spread_transition_matrix": self.spread_transition_matrix,
            "spread_regime_bins_in_ticks": getattr(self, 'spread_regime_bins', None)
        }

def generate_sample_lob_data(num_events=5000) -> pd.DataFrame:
    """Generates a synthetic DataFrame of LOB events for demonstration."""
    print(f"\nGenerating {num_events} synthetic LOB events...")
    events = []
    base_price = 100.0
    tick_size = 0.01
    best_bid, best_ask = base_price, base_price + tick_size
    order_id_counter = 0

    start_time = pd.Timestamp.now()
    
    for i in range(num_events):
        event_type = np.random.choice(['LIMIT', 'CANCEL', 'TRADE'], p=[0.6, 0.3, 0.1])
        side = np.random.choice(['BID', 'ASK'])
        timestamp = start_time + pd.to_timedelta(i * 100, unit='ms')

        event = {
            'timestamp': timestamp,
            'best_bid': best_bid,
            'best_ask': best_ask
        }
        
        if event_type == 'LIMIT':
            order_id_counter += 1
            event.update({
                'event_type': 'LIMIT',
                'order_id': order_id_counter,
                'side': side,
                'price': best_bid - np.random.randint(0, 5) * tick_size if side == 'BID' else best_ask + np.random.randint(0, 5) * tick_size,
                'quantity': np.random.randint(1, 10) * 100
            })
            # Update BBO if a new best price is set
            if side == 'BID' and event['price'] > best_bid: best_bid = event['price']
            if side == 'ASK' and event['price'] < best_ask: best_ask = event['price']

        elif event_type == 'CANCEL' and order_id_counter > 0:
            event.update({
                'event_type': 'CANCEL',
                'order_id': np.random.randint(1, order_id_counter),
                'side': side,
                'price': None, 'quantity': None
            })

        elif event_type == 'TRADE' and order_id_counter > 0:
            # Assume a trade hits a resting order
            event.update({
                'event_type': 'TRADE',
                'order_id': np.random.randint(1, order_id_counter),
                'side': 'ASK' if side == 'BID' else 'BID', # Trade hits the opposite side
                'price': best_ask if side == 'BID' else best_bid, # Market order crosses spread
                'quantity': np.random.randint(1, 5) * 100
            })
            # Trades can move the BBO
            if side == 'BID': best_ask += tick_size
            else: best_bid -= tick_size
        else:
            continue
        
        events.append(event)

    return pd.DataFrame(events)


if __name__ == '__main__':
    # 1. Generate sample data
    sample_data = generate_sample_lob_data(num_events=20000)

    # 2. Initialize the calibrator with the data
    calibrator = LOBParameterCalibrator(sample_data, tick_size=0.01)

    # 3. Run the calibration process
    calibrated_params = calibrator.calibrate_all(max_depth=10, n_spread_regimes=3)
    
    # 4. Print the results
    print("\n\n--- CALIBRATED PARAMETERS ---")
    print("\n1. Order Arrival Intensities (events/sec):")
    for k, v in calibrated_params['arrival_intensities'].items():
        print(f"  - {k}: {v:.4f}")

    print("\n2. Fill Probabilities by Depth (in ticks from opposite BBO):")
    for depth, prob in calibrated_params['fill_probabilities_by_depth'].items():
        print(f"  - Depth {depth}: {prob:.4f}")

    print("\n3. Spread Regime Transition Matrix:")
    print("   (Bins in ticks:", [round(b, 2) for b in calibrated_params['spread_regime_bins_in_ticks']], ")")
    print(calibrated_params['spread_transition_matrix'].round(4))