def create_tracker(tracker_config, device, half):
    
    
    from tracker.byte_tracker import BYTETracker
    bytetracker = BYTETracker(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=25
    )
    return bytetracker