FEATURE_NAMES = [
    'temperature', 'vibration', 'pressure', 'rpm',
    'voltage', 'current', 'operating_hours'
]

def reading_to_vector(reading: dict):
    return [[float(reading[name]) for name in FEATURE_NAMES]]
