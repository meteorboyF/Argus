"""Spoken status, place memory and local route execution over verified geometry."""
import math
import time


def usable_pose(orch):
    if not orch.depth.calibrated or orch.slam is None:
        return None
    health = getattr(getattr(orch.depth,'health',None),'state',None)
    if getattr(health,'value',None) != 'ok':
        return None
    pose = orch.slam.state
    return pose if pose.tracking and pose.pose_valid and pose.quality >= .5 else None


def status(orch):
    tasks = getattr(orch,'tasks',None)
    task = tasks.label if tasks and tasks.busy else ''
    active = f'I am working on {task}. ' if task else 'I am listening. '
    if not orch.depth.calibrated:
        return active+'Stereo calibration is missing, so metric warnings and navigation are unavailable.'
    if usable_pose(orch) is None:
        return active+'Location tracking is unavailable. Do not rely on route directions.'
    return active+'Local tracking is available. Walking acceptance still needs verification.'


def run(orch, skill, name=''):
    if skill == 'inventory':
        from .object_inventory import inventory
        return inventory(orch)
    if skill == 'status':
        return status(orch)
    if skill == 'safety_status':
        return 'I cannot confirm that it is safe to walk or cross. The geometry safety loop warns independently; keep using your mobility aid.'
    if skill == 'watch':
        from .object_search import search
        return search(orch,name)
    if skill == 'location':
        pose = usable_pose(orch)
        if pose is None:
            return 'I cannot establish your location. Calibrated, current tracking is required.'
        return 'I am tracking a local position relative to where this session began. I do not know your street address or room name unless you mark a place.'
    mapping = getattr(orch,'mapping',None)
    if mapping is None:
        return 'Local mapping is unavailable until stereo calibration and tracking are ready.'
    if skill == 'places':
        names = sorted(mapping.map.places)
        return 'Remembered places this session: '+', '.join(names)+'.' if names else 'No places have been marked in this session.'
    pose = usable_pose(orch)
    if pose is None:
        return 'Tracking is unavailable. I cannot mark a place or give route directions.'
    if skill == 'mark_place':
        if not name:
            return 'What should I call this place?'
        if mapping.map.mark(name,pose):
            return f'I marked this place as {name} for this session.'
        return 'I could not mark this place with reliable tracking.'
    if skill == 'navigate':
        if name not in mapping.map.places:
            return f'I have no marked place called {name}. Say remember this place as, followed by its name, while you are there.'
        if pose.epoch != mapping.map.epoch:
            return 'The map frame changed. Please mark the destination again.'
        path = mapping.map.route(pose.x_m,pose.z_m,name)
        if not path:
            return 'I cannot plan a route through recently observed space. Unknown areas are blocked.'
        if not getattr(orch.cfg.navigation,'guided_routes_enabled',False):
            return f'I found a local geometric route to {name}. Spoken walking directions are disabled until navigation acceptance is verified.'
        return follow_route(orch,name)
    return 'That assistance action is unavailable.'


def follow_route(orch, name, timeout_s=120):
    """Opt-in route follower; safety, stale depth and pose can all inhibit it."""
    orch.seat_cancel.clear()
    said, said_at = '', 0
    deadline = time.monotonic()+timeout_s
    while time.monotonic()<deadline and not orch._stop.is_set() and not orch.seat_cancel.is_set():
        if orch.tasks.cancelled():
            return ''
        pose = usable_pose(orch)
        observation = orch.latest_observation()
        if (pose is None or observation is None or observation[1] is None
                or pose.epoch != orch.mapping.map.epoch):
            return 'Stop. Tracking or depth is unavailable. Route guidance has ended.'
        path = orch.mapping.map.route(pose.x_m,pose.z_m,name)
        if not path:
            return 'Stop. I no longer have a route through observed space.'
        target = orch.mapping.map.places[name]
        if math.hypot(pose.x_m-target.x,pose.z_m-target.z)<.5:
            return f'You are near the marked place {name}. Confirm your surroundings with your mobility aid.'
        from .telemetry import bus
        from .depth_freshness import depth_is_current
        hazards = getattr(orch,'_hazards',None)
        if (hazards is None or not hazards.last.valid or hazards.last.drop
                or not depth_is_current(getattr(orch,'_hazards_ts',None))):
            return 'Stop. Current ground and hazard observations are unavailable.'
        if bus.get('safety_level') != 'CLEAR':
            return 'Stop. Safety observations do not permit route guidance.'
        guidance = orch.navigator.evaluate(observation[1]) if orch.navigator else None
        if guidance is None or guidance.status not in ('clear',):
            return 'Stop. The current corridor is blocked or uncertain.'
        waypoint = path[min(4,len(path)-1)]
        heading = math.degrees(math.atan2(waypoint[0]-pose.x_m,waypoint[1]-pose.z_m))-pose.yaw_deg
        heading = (heading+180)%360-180
        phrase = ('Continue toward the marked place using your mobility aid.' if abs(heading)<12
                  else f'Turn slowly {"right" if heading>0 else "left"} toward the route.')
        if phrase != said or time.monotonic()-said_at>6:
            from .speech import Priority
            orch.speaker.speak(phrase,Priority.GUIDANCE)
            said,said_at = phrase,time.monotonic()
        orch._stop.wait(.5)
    return 'Route guidance stopped.'
