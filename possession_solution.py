"""Generate a possession submission from the automatic tactical tracking and tagging.

Run from this directory:
    python possession_solution.py --out submission.csv

The pipeline is deterministic, uses no ground truth or video, and needs only numpy.
"""

import argparse
from dataclasses import dataclass
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)
CONTROL_RADIUS = 3.0
CONTROL_MARGIN = 0.75
MAX_CONTROL_SPEED = 6.0
MAX_SPEED_DIFFERENCE = 4.0
MAX_BALL_SPEED = 45.0
CONTROL_FRAMES = 8
RESTART_FRAMES = 3
MAX_HOLD_FRAMES = 5 * FPS


UNKNOWN, HOME, AWAY, DEAD = -1, 0, 1, 2
OUTPUT_LABELS = np.array(["X", "H", "A", "D"])
ATTACK_CODES = {
    "attacking transition",
    "ball in final third",
    "ball in the box",
    "build up",
    "creating chances",
    "long ball",
    "players in the box",
    "progression",
}


@dataclass
class TagSignals:
    evidence: np.ndarray
    set_piece: np.ndarray
    restart_team: np.ndarray


@dataclass
class TrackingSignals:
    evidence: np.ndarray
    player_count: np.ndarray
    ball_seen: np.ndarray
    ball_out: np.ndarray
    ball_xy: np.ndarray
    ball_speed: np.ndarray
    distances: np.ndarray
    player_speed: np.ndarray


def _normalise_name(value):
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", "", value.lower())
    return value.replace("atletic", "athletic")


def _team_names_and_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected exactly two teams, found {}".format(meta_path, len(teams)))

    home_id = teams[0].get("id")
    team_ids = {team.get("id") for team in teams}
    if None in team_ids or len(team_ids) != 2:
        raise ValueError("{}: team IDs must be distinct and nonempty".format(meta_path))
    team_names = tuple((team.findtext("Name") or "").strip() for team in teams)
    if not all(team_names):
        raise ValueError("{}: team metadata is missing a name".format(meta_path))

    slots = []
    for player in root.iter("Player"):
        team_id = player.get("teamId")
        if team_id not in team_ids:
            raise ValueError("{}: player has an unknown teamId {!r}".format(meta_path, team_id))
        slots.append(HOME if team_id == home_id else AWAY)
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return team_names, np.asarray(slots, dtype=np.int8)


def _team_code(tag_name, team_names):
    normalised_tag = _normalise_name(tag_name)
    if not normalised_tag or normalised_tag == "na":
        return None
    for code, name in enumerate(team_names):
        normalised_name = _normalise_name(name)
        if normalised_tag in normalised_name or normalised_name in normalised_tag:
            return code
    return None


def _tag_evidence(tag_path, n_frames, fps, team_names):
    evidence = np.zeros(n_frames, dtype=np.float32)
    set_piece = np.zeros(n_frames, dtype=bool)
    restart_votes = np.zeros((n_frames, 2), dtype=np.int32)
    for instance in ET.parse(tag_path).iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        if code not in ATTACK_CODES and code != "set pieces":
            continue
        tags = {
            (label.findtext("group") or "").strip(): (label.findtext("text") or "").strip()
            for label in instance.findall("label")
        }
        team = _team_code(tags.get("Team", ""), team_names)
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError) as error:
            raise ValueError("{}: invalid Smart Tagging interval".format(tag_path)) from error
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            raise ValueError("{}: invalid Smart Tagging interval".format(tag_path))
        start = int(round(start_s * fps))
        end = int(round(end_s * fps))
        start, end = max(0, start), min(n_frames, end)
        if end <= start:
            continue
        if code == "set pieces":
            set_piece[start:end] = True
            if team is not None:
                restart_votes[start:end, team] += 1
            continue
        if team is None:
            continue
        sign = 1.0 if team == HOME else -1.0
        evidence[start:end] += sign * 0.35
    restart_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    for team in (HOME, AWAY):
        unambiguous = (restart_votes[:, team] > 0) & (restart_votes[:, 1 - team] == 0)
        restart_team[unambiguous] = team
    return TagSignals(np.clip(evidence, -0.7, 0.7), set_piece, restart_team)


def _parse_position(group, path, line_number):
    fields = group.split(",")
    if len(fields) < 2:
        raise ValueError("{}:{}: malformed tracking position".format(path, line_number))
    try:
        x, y = float(fields[0]), float(fields[1])
        speed = float(fields[2]) if len(fields) > 2 else np.nan
    except ValueError as error:
        raise ValueError("{}:{}: malformed tracking number".format(path, line_number)) from error
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if np.isfinite(speed) and speed < 0:
        raise ValueError("{}:{}: negative tracking speed".format(path, line_number))
    return x, y, speed


def _tracking_evidence(raw_path, team_of_slot, n_frames, pitch):
    length, width = pitch
    closest = np.full((n_frames, 2), np.inf, dtype=np.float32)
    player_count = np.zeros(n_frames, dtype=np.uint8)
    ball_seen = np.zeros(n_frames, dtype=bool)
    ball_out = np.zeros(n_frames, dtype=bool)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    speeds = np.full(n_frames, np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, 2), np.nan, dtype=np.float32)
    evidence = np.zeros(n_frames, dtype=np.float32)

    line_number = 0
    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: expected three ':'-separated fields".format(
                    raw_path, line_number))
            try:
                frame_id = int(parts[0])
            except ValueError as error:
                raise ValueError("{}:{}: invalid frame id".format(
                    raw_path, line_number)) from error
            frame = frame_id - 1
            if frame < 0 or frame >= n_frames:
                raise ValueError("{}:{}: frame id {} is outside 1..{}".format(
                    raw_path, line_number, frame_id, n_frames))
            if frame != line_number - 1:
                raise ValueError("{}:{}: expected contiguous frame id {}, got {}".format(
                    raw_path, line_number, line_number, frame_id))

            players = parts[1].split(";")
            if len(players) != len(team_of_slot):
                raise ValueError("{}:{}: expected {} player slots, found {}".format(
                    raw_path, line_number, len(team_of_slot), len(players)))

            positions = []
            for slot, group in enumerate(players):
                position = _parse_position(group, raw_path, line_number)
                if position is None:
                    continue
                x, y, speed = position
                player_count[frame] += 1
                positions.append((team_of_slot[slot], x * length, y * width, speed))

            ball = _parse_position(parts[2], raw_path, line_number)
            if ball is None:
                continue
            ball_seen[frame] = True
            ball_x, ball_y, ball_speed = ball
            ball_xy[frame] = (ball_x * length, ball_y * width)
            speeds[frame] = ball_speed
            ball_out[frame] = (
                ball_x < -0.03 or ball_x > 1.03 or ball_y < -0.03 or ball_y > 1.03
            )

            for team, x, y, speed in positions:
                dx, dy = x - ball_x * length, y - ball_y * width
                distance = dx * dx + dy * dy
                if distance < closest[frame, team]:
                    closest[frame, team] = distance
                    player_speed[frame, team] = speed

            home_distance, away_distance = np.sqrt(closest[frame])
            best_distance = min(home_distance, away_distance)
            if best_distance > 8.0:
                continue
            if np.isfinite(home_distance) and np.isfinite(away_distance):
                margin = (away_distance - home_distance) / (
                    away_distance + home_distance + 1.0
                )
                frame_evidence = 3.0 * margin * np.exp(-best_distance / 5.0)
            else:
                nearest_team = HOME if home_distance < away_distance else AWAY
                sign = 1.0 if nearest_team == HOME else -1.0
                frame_evidence = sign * 1.0 * np.exp(-best_distance / 4.0)
            if np.isfinite(ball_speed) and ball_speed > 6.0:
                frame_evidence *= 0.5
            evidence[frame] = frame_evidence

    if line_number != n_frames:
        raise ValueError("{}: expected {} frames, found {}".format(
            raw_path, n_frames, line_number))
    distances = np.sqrt(closest)
    return TrackingSignals(
        evidence, player_count, ball_seen, ball_out, ball_xy, speeds, distances, player_speed
    )


def _sustained_runs(mask, min_frames):
    result = np.zeros(len(mask), dtype=bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start >= min_frames:
            result[start:end] = True
    return result


def _long_sparse_runs(player_count, ball_seen, min_frames):
    return _sustained_runs((player_count <= 4) & ~ball_seen, min_frames)


def _ball_motion(tracking):
    measured = np.full(len(tracking.ball_seen), np.nan, dtype=np.float32)
    adjacent = tracking.ball_seen[1:] & tracking.ball_seen[:-1]
    measured[1:][adjacent] = (
        np.linalg.norm(np.diff(tracking.ball_xy, axis=0)[adjacent], axis=1) * FPS
    )
    speed = tracking.ball_speed.copy()
    missing = ~np.isfinite(speed)
    speed[missing] = measured[missing]
    reliable = (
        tracking.ball_seen & np.isfinite(speed) & (speed <= MAX_BALL_SPEED)
        & (~np.isfinite(measured) | (measured <= MAX_BALL_SPEED))
    )
    return speed, reliable


def _control_candidates(tracking, tags, speed, reliable):
    candidates = np.full(len(speed), UNKNOWN, dtype=np.int8)
    for team in (HOME, AWAY):
        nearest = tracking.distances[:, team]
        other = tracking.distances[:, 1 - team]
        speed_agrees = (
            np.isfinite(tracking.player_speed[:, team])
            & (np.abs(speed - tracking.player_speed[:, team]) <= MAX_SPEED_DIFFERENCE)
        )
        margin = np.full(len(speed), -np.inf, dtype=np.float32)
        np.subtract(other, nearest, out=margin, where=np.isfinite(nearest))
        geometry = (nearest <= CONTROL_RADIUS) & (margin >= CONTROL_MARGIN)
        sign = 1 if team == HOME else -1
        support = sign * (tracking.evidence + tags.evidence) >= 0.35
        controlled = (
            reliable & ~tracking.ball_out & geometry & speed_agrees
            & (speed <= MAX_CONTROL_SPEED) & support
        )
        candidates[controlled] = team
    return candidates


def _stationary_set_pieces(tracking, tags, speed, reliable):
    stationary = reliable & ~tracking.ball_out & (speed <= 0.5)
    result = np.zeros(len(speed), dtype=bool)
    start = 0
    low = high = None
    for frame in range(len(speed) + 1):
        if frame < len(speed) and stationary[frame]:
            point = tracking.ball_xy[frame]
            next_low = point if low is None else np.minimum(low, point)
            next_high = point if high is None else np.maximum(high, point)
            if np.linalg.norm(next_high - next_low) <= 0.5:
                low, high = next_low, next_high
                continue
        if frame - start >= FPS:
            result[start:frame] = tags.set_piece[start:frame]
        start = frame
        if frame < len(speed) and stationary[frame]:
            low = high = tracking.ball_xy[frame]
        else:
            start += 1
            low = high = None
    return result


def _decode_states(candidates, hard_dead, moving, restart_team):
    labels = np.full(len(candidates), UNKNOWN, dtype=np.int8)
    state = UNKNOWN
    pending = UNKNOWN
    pending_start = 0
    pending_count = 0
    last_support = -MAX_HOLD_FRAMES
    restart_owner = UNKNOWN
    owner_frame = -MAX_HOLD_FRAMES
    restart_count = 0
    restart_start = 0
    resume_team = UNKNOWN
    last_dead_support = -MAX_HOLD_FRAMES

    for frame, candidate in enumerate(candidates):
        if hard_dead[frame]:
            last_dead_support = frame
            if state != DEAD:
                restart_owner = UNKNOWN
                owner_frame = -MAX_HOLD_FRAMES
            state = DEAD
            pending = UNKNOWN
            pending_count = 0
            restart_count = 0
        if state == DEAD:
            if candidate != UNKNOWN:
                restart_owner, owner_frame = candidate, frame
            owner = candidate
            if owner == UNKNOWN and restart_team[frame] != UNKNOWN:
                owner = restart_team[frame]
            if owner == UNKNOWN and frame - owner_frame <= 2 * FPS:
                owner = restart_owner
            if not hard_dead[frame] and moving[frame]:
                if restart_count == 0 or owner != resume_team:
                    restart_start, restart_count, resume_team = frame, 1, owner
                else:
                    restart_count += 1
                if restart_count >= RESTART_FRAMES:
                    state = owner
                    labels[restart_start:frame + 1] = owner
                    last_support = frame
                    restart_count = 0
                else:
                    labels[frame] = DEAD if frame - last_dead_support <= MAX_HOLD_FRAMES else UNKNOWN
            else:
                labels[frame] = DEAD if frame - last_dead_support <= MAX_HOLD_FRAMES else UNKNOWN
                restart_count = 0
            continue

        if candidate == state and candidate != UNKNOWN:
            last_support = frame
        if frame - last_support > MAX_HOLD_FRAMES:
            state = UNKNOWN
        labels[frame] = state
        if candidate == UNKNOWN or candidate == state:
            pending, pending_count = UNKNOWN, 0
            continue
        if candidate != pending:
            pending = candidate
            pending_start = frame
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= CONTROL_FRAMES:
            state = candidate
            labels[pending_start:frame + 1] = state
            last_support = frame
            pending = UNKNOWN
            pending_count = 0

    return OUTPUT_LABELS[labels + 1]


def build_submission(data_dir=None):
    n_frames = N_FRAMES
    data_root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(HERE, "data")
    tactical_dir = os.path.join(data_root, "tactical")
    paths = [
        os.path.join(tactical_dir, "BAR-ATH_tactical_atd.xml"),
        os.path.join(tactical_dir, "BAR-ATH_tactical_atd.txt"),
        os.path.join(tactical_dir, "BAR-ATH_tactical_smart_tagging.xml"),
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise IOError(
            "match data not found:\n  {}\n\nPlace the authorised tactical files in\n  {}\n"
            "or set MTAH_DATA_DIR to the bundle root.".format(
                "\n  ".join(missing), tactical_dir
            )
        )
    team_names, team_of_slot = _team_names_and_slots(paths[0])
    tracking = _tracking_evidence(
        paths[1], team_of_slot, n_frames, PITCH
    )
    tags = _tag_evidence(paths[2], n_frames, FPS, team_names)
    speed, reliable = _ball_motion(tracking)
    candidates = _control_candidates(tracking, tags, speed, reliable)
    hard_dead = _sustained_runs(tracking.ball_out, CONTROL_FRAMES) | _long_sparse_runs(
        tracking.player_count, tracking.ball_seen, min_frames=FPS
    )
    hard_dead |= _stationary_set_pieces(tracking, tags, speed, reliable)
    moving = reliable & ~tracking.ball_out & (speed > 1.0)
    return _decode_states(candidates, hard_dead, moving, tags.restart_team)


def write_submission(path, labels):
    if len(labels) != N_FRAMES:
        raise ValueError("expected {} labels, got {}".format(
            N_FRAMES, len(labels)))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=os.path.join(HERE, "submission.csv"),
        help="destination CSV (default: submission.csv in this folder)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="root of the data bundle (default: MTAH_DATA_DIR or ./data)",
    )
    args = parser.parse_args()

    try:
        labels = build_submission(args.data_dir)
    except OSError as error:
        parser.error(str(error))
    write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("Wrote {} labels to {}".format(len(labels), os.path.abspath(args.out)))
    print("Counts: {}".format("  ".join("{} {}".format(k, v) for k, v in counts.items())))
    print("Validate with: python validate_submission.py \"{}\"".format(args.out))


if __name__ == "__main__":
    main()
