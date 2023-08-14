import numpy as np
# from collections import OrderedDict
from uuid import uuid4

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _max_count = 1000
    track_id = None
    _count = 0
    is_activated = False
    state = TrackState.New
    # history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    # @staticmethod
    def next_id(self):
        # BaseTrack._count += 1
        # return BaseTrack._count
        if len(self._count) > self._max_count:
            self._count = 0
        new_id = str(uuid4())
        return new_id

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
