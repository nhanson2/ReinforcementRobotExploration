from matplotlib.pyplot import new_figure_manager
import numpy as np

# http://incompleteideas.net/tiles/tiles3.py-remove

class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfull_count) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count

def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates

def tiles(iht_or_size, num_tilings, floats, ints=[], read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [np.floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles

# shout out to Richard Sutton

def active_tiles(state, action, num_tilings, iht, env):
    # hack as heck tbh    
    temp = action[0][0]
    try:
        position = state[0][0]
        velocity = state[1][0]
    except:
        position = state[0]
        velocity = state[1]
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    return tiles(iht, num_tilings, [8*position/(max_position - min_position), 8*velocity/(max_velocity - min_velocity)], [temp])

def explore_count(action, state, weights, num_tilings, iht, env):
    beta = 0.5
    active = active_tiles(state, action, num_tilings, iht, env)
    weights[active] += 1
    additive = beta/np.sqrt(np.sum(weights[active]))
    new_action = action + np.random.normal(loc=0.0,scale=additive)
    return weights, new_action

def explore_ucb(action, state, weights, num_tilings, iht, env):
    active = active_tiles(state, action, num_tilings, iht, env)
    weights[active] += 1
    additive = np.sqrt(np.log(np.sum(weights[active]))/np.sum(weights))
    new_action = action + np.random.normal(loc=0.0,scale=additive)
    return weights, new_action