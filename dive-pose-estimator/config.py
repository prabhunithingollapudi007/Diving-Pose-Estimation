MIN_BBOX_AREA = 5000
MAX_BBOX_DISTANCE = 200
MIN_BBOX_START_HEIGHT_LOWER_LIMIT = 0.3
MIN_BBOX_START_HEIGHT_UPPER_LIMIT = 0.7
MAX_CONSECUTIVE_INVALID_FRAMES = 10
FILTER_WINDOW_SIZE = 5
FILTER_SIGMA = 1
DIVER_ON_BOARD_HEIGHT_PIXEL = 1010
WATER_HEIGHT_PIXEL = 375
INITIAL_DIVER_HEIGHT_METERS = 1.75
BOARD_HEIGHT_METERS = 5

COLORS = [

    (255, 0, 0),  # Nose
    (255, 85, 0),  # Left eye
    (255, 170, 0),  # Right eye
    (255, 255, 0),  # Left ear
    (170, 255, 0),  # Right ear
    (85, 255, 0),  # Left shoulder
    (0, 255, 0),  # Right shoulder
    (0, 255, 85),  # Left elbow
    (0, 255, 170),  # Right elbow
    (0, 255, 255),  # Left wrist
    (0, 170, 255),  # Right wrist
    (0, 85, 255),  # Left hip
    (0, 0, 255),  # Right hip
    (85, 0, 255),  # Left knee
    (170, 0, 255),  # Right knee
    (255, 0, 255),  # Left ankle
    (255, 0, 170)  # Right ankle
]

STAGES = [
    "Absprung",
    "Ansatz",
    "Beginn Streckung",
    "Ende Streckung",
    "Eintauchen"
]