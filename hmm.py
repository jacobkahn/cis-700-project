class HMM:
    def __init__(self):
        self.states = {}
        self.edges = {}

class HMMNode:
    def __init__(self, emission_map):
        self.emission_map = emission_map

    def emit(self):
        return emission_map.emit()

# class HMMNodeEmissionMap:
#     def __init__(self, )
