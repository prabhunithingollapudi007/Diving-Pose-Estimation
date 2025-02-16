import numpy as np

class KeypointFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = {}  # Dictionary to store keypoints per instance

    def filter_keypoints(self, instance_id, keypoints):
        """Applies a moving average filter to keypoints for a given instance."""
        keypoints = np.array(keypoints)  # Convert to NumPy array
        
        # Initialize history for this instance if not present
        if instance_id not in self.history:
            self.history[instance_id] = []

        # Store keypoints in history
        self.history[instance_id].append(keypoints)

        # Keep only the last `window_size` frames
        if len(self.history[instance_id]) > self.window_size:
            self.history[instance_id].pop(0)

        # Compute moving average
        avg_keypoints = np.mean(self.history[instance_id], axis=0)
        return avg_keypoints.tolist()
