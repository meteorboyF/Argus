"""Rule-based navigation guidance derived from the ARGUS World Model."""


def generate_navigation_guidance(world: dict) -> str | None:
    """
    Returns a spoken navigation instruction string, or None if the scene is clear.

    Priority order: hazards → proximity → floor → doors → clear.
    """
    hazard  = world.get("hazard")
    nearest = world.get("nearest_obstacle_dist", 999.0)
    floor   = world.get("navigable_floor", True)
    objects = world.get("objects", [])

    # 1. Hazard (highest priority)
    if hazard == "stairs_down":
        return "CAUTION: stairs going down ahead. Stop and feel with your foot."
    if hazard == "stairs_up":
        return "CAUTION: stairs going up ahead."

    # 2. Close obstacle
    if nearest < 0.8:
        return "Stop. Obstacle directly ahead."

    # 3. No navigable floor
    if not floor:
        return "No clear floor detected. Proceed with caution."

    # 4. Doors in range
    doors = [
        o for o in objects
        if o.get("label") in ("door_closed", "door_open") and not o.get("private")
    ]
    if doors:
        door = min(doors, key=lambda o: o["distance"])
        state = "open" if door["label"] == "door_open" else "closed"
        dist  = door["distance"]
        direc = door["direction"]
        return f"There is a {state} door {dist:.1f} metres ahead to your {direc}."

    # 5. Nearby non-private objects worth mentioning
    visible = [o for o in objects if not o.get("private") and o["distance"] < 3.0]
    if visible:
        closest = min(visible, key=lambda o: o["distance"])
        return (
            f"{closest['label'].replace('_', ' ').capitalize()} "
            f"{closest['distance']:.1f} metres to your {closest['direction']}."
        )

    return None  # scene is clear — caller decides whether to speak
