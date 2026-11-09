import unittest
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import possession_solution
from possession_solution import (
    AWAY,
    DEAD,
    HOME,
    UNKNOWN,
    TagSignals,
    TrackingSignals,
    _ball_motion,
    _control_candidates,
    _decode_states,
    _long_sparse_runs,
    _stationary_set_pieces,
    _sustained_runs,
    _tag_evidence,
    _tracking_evidence,
    _team_code,
)


class PossessionSolutionTests(unittest.TestCase):
    def test_team_tags_map_to_metadata_team_order(self):
        names = ("FC Barcelona", "Athletic Club")
        self.assertEqual(_team_code("Barcelona", names), HOME)
        self.assertEqual(_team_code("Atletic Club", names), AWAY)
        self.assertIsNone(_team_code("N/A", names))

    def test_decoder_confirms_transitions(self):
        candidates = np.r_[np.full(30, HOME), np.full(30, AWAY), np.full(30, HOME)]
        labels = self.decode(candidates)
        self.assertTrue(np.all(labels[:30] == "H"))
        self.assertTrue(np.all(labels[30:60] == "A"))
        self.assertTrue(np.all(labels[60:] == "H"))

    def decode(self, candidates, dead=None, moving=None, restarts=None):
        n = len(candidates)
        return _decode_states(
            np.asarray(candidates),
            np.zeros(n, dtype=bool) if dead is None else dead,
            np.zeros(n, dtype=bool) if moving is None else moving,
            np.full(n, UNKNOWN) if restarts is None else restarts,
        )

    def test_brief_opponent_touch_does_not_transfer_possession(self):
        candidates = np.r_[np.full(20, HOME), np.full(7, AWAY), np.full(20, HOME)]
        self.assertTrue(np.all(self.decode(candidates) == "H"))

    def test_long_loss_of_control_abstains_and_requires_reacquisition(self):
        candidates = np.r_[np.full(20, HOME), np.full(140, UNKNOWN), np.full(20, AWAY)]
        labels = self.decode(candidates)
        self.assertTrue(np.all(labels[20:145] == "H"))
        self.assertTrue(np.all(labels[145:160] == "X"))
        self.assertTrue(np.all(labels[160:] == "A"))

    def test_dead_ball_persists_until_confirmed_motion(self):
        candidates = np.r_[np.full(20, HOME), np.full(40, AWAY)]
        dead = np.zeros(60, dtype=bool)
        dead[20:25] = True
        moving = np.zeros(60, dtype=bool)
        moving[45:] = True
        labels = self.decode(candidates, dead, moving)
        self.assertTrue(np.all(labels[:20] == "H"))
        self.assertTrue(np.all(labels[20:45] == "D"))
        self.assertTrue(np.all(labels[45:] == "A"))

    def test_restart_uses_recent_contact_when_ball_is_kicked_away(self):
        candidates = np.r_[np.full(30, HOME), np.full(20, UNKNOWN)]
        dead = np.zeros(50, dtype=bool)
        dead[:25] = True
        moving = np.zeros(50, dtype=bool)
        moving[30:] = True
        labels = self.decode(candidates, dead, moving)
        self.assertTrue(np.all(labels[:30] == "D"))
        self.assertTrue(np.all(labels[30:] == "H"))

    def test_missing_ball_does_not_confirm_restart(self):
        dead = np.zeros(50, dtype=bool)
        dead[:10] = True
        labels = self.decode(
            np.full(50, UNKNOWN), dead, restarts=np.full(50, AWAY)
        )
        self.assertTrue(np.all(labels == "D"))

    def test_motion_without_known_restart_team_abstains_instead_of_inventing_owner(self):
        dead = np.zeros(50, dtype=bool)
        dead[:10] = True
        labels = self.decode(np.full(50, UNKNOWN), dead, np.ones(50, dtype=bool))
        self.assertTrue(np.all(labels[:10] == "D"))
        self.assertTrue(np.all(labels[10:] == "X"))

    def test_stale_dead_state_abstains_without_assuming_live_play(self):
        dead = np.zeros(160, dtype=bool)
        dead[:5] = True
        labels = self.decode(np.full(160, UNKNOWN), dead)
        self.assertTrue(np.all(labels[:130] == "D"))
        self.assertTrue(np.all(labels[130:] == "X"))

    def test_tagged_restart_with_motion_can_resume_without_nearby_player(self):
        dead = np.zeros(50, dtype=bool)
        dead[:10] = True
        labels = self.decode(
            np.full(50, UNKNOWN), dead, np.ones(50, dtype=bool), np.full(50, AWAY)
        )
        self.assertTrue(np.all(labels[:10] == "D"))
        self.assertTrue(np.all(labels[10:] == "A"))

    def test_isolated_low_tracking_frame_is_not_a_dead_ball(self):
        players = np.full(40, 20, dtype=np.uint8)
        players[15] = 0
        ball_seen = np.ones(40, dtype=bool)

        dead = _long_sparse_runs(players, ball_seen, min_frames=25)

        self.assertFalse(dead.any())
        self.assertEqual(UNKNOWN, -1)
        self.assertEqual(AWAY, 1)
        self.assertEqual(DEAD, 2)

    def test_tracking_parser_uses_nearby_team_and_one_based_frame_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tracking.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("1:0.11,0.5,1;0.8,0.5,1:0.1,0.5,1\n")

            tracking = _tracking_evidence(
                path, np.array([HOME, AWAY]), 1, (100.0, 60.0)
            )

        self.assertGreater(tracking.evidence[0], 0)
        self.assertEqual(tracking.player_count[0], 2)
        self.assertTrue(tracking.ball_seen[0])
        self.assertFalse(tracking.ball_out[0])

    def signals(self, n=60):
        return TrackingSignals(
            np.ones(n), np.full(n, 22), np.ones(n, dtype=bool),
            np.zeros(n, dtype=bool), np.zeros((n, 2)), np.ones(n),
            np.tile([1.0, 5.0], (n, 1)), np.ones((n, 2)),
        )

    def tags(self, n=60):
        return TagSignals(np.zeros(n), np.zeros(n, dtype=bool), np.full(n, UNKNOWN))

    def test_fast_pass_near_opponent_is_not_control(self):
        tracking = self.signals()
        tracking.distances[:] = [5.0, 1.0]
        tracking.evidence[:] = -1
        speed = np.full(60, 15.0)
        candidates = _control_candidates(
            tracking, self.tags(), speed, np.ones(60, dtype=bool)
        )
        self.assertTrue(np.all(candidates == UNKNOWN))

    def test_duel_speed_mismatch_or_missing_speed_is_not_control(self):
        for distance, player_speed in (([1.0, 1.2], 1.0), ([1.0, 5.0], 8.0),
                                       ([1.0, 5.0], np.nan)):
            with self.subTest(distance=distance, player_speed=player_speed):
                tracking = self.signals()
                tracking.distances[:] = distance
                tracking.player_speed[:] = player_speed
                candidates = _control_candidates(
                    tracking, self.tags(), tracking.ball_speed, tracking.ball_seen
                )
                self.assertTrue(np.all(candidates == UNKNOWN))

    def test_attacking_tags_without_ball_cannot_create_control(self):
        tracking = self.signals()
        tracking.ball_seen[:] = False
        tags = self.tags()
        tags.evidence[:] = 0.7
        candidates = _control_candidates(
            tracking, tags, tracking.ball_speed, tracking.ball_seen
        )
        self.assertTrue(np.all(candidates == UNKNOWN))

    def test_missing_player_distances_are_handled_without_invalid_arithmetic(self):
        tracking = self.signals()
        tracking.distances[:] = np.inf
        with np.errstate(invalid="raise"):
            candidates = _control_candidates(
                tracking, self.tags(), tracking.ball_speed, tracking.ball_seen
            )
        self.assertTrue(np.all(candidates == UNKNOWN))

    def test_ball_motion_rejects_teleport_and_does_not_bridge_missing_frames(self):
        tracking = self.signals(4)
        tracking.ball_xy[:] = [[0, 0], [20, 0], [np.nan, np.nan], [25, 0]]
        tracking.ball_seen[2] = False
        tracking.ball_speed[:] = np.nan
        _, reliable = _ball_motion(tracking)
        self.assertFalse(reliable.any())

    def test_stationary_ball_requires_set_piece_context(self):
        tracking = self.signals()
        tracking.ball_speed[:] = 0
        tags = self.tags()
        self.assertFalse(_stationary_set_pieces(
            tracking, tags, tracking.ball_speed, tracking.ball_seen
        ).any())
        tags.set_piece[:] = True
        self.assertTrue(_stationary_set_pieces(
            tracking, tags, tracking.ball_speed, tracking.ball_seen
        ).all())

    def test_moving_or_drifting_ball_in_set_piece_is_not_dead(self):
        tracking = self.signals()
        tags = self.tags()
        tags.set_piece[:] = True
        tracking.ball_speed[:] = 5
        self.assertFalse(_stationary_set_pieces(
            tracking, tags, tracking.ball_speed, tracking.ball_seen
        ).any())
        tracking.ball_speed[:] = 0
        tracking.ball_xy[:, 0] = np.arange(60) * 0.1
        self.assertFalse(_stationary_set_pieces(
            tracking, tags, tracking.ball_speed, tracking.ball_seen
        ).any())

    def test_isolated_out_of_pitch_detection_is_not_a_stoppage(self):
        mask = np.zeros(60, dtype=bool)
        mask[20] = True
        self.assertFalse(_sustained_runs(mask, 8).any())
        mask[20:28] = True
        self.assertEqual(np.flatnonzero(_sustained_runs(mask, 8)).tolist(), list(range(20, 28)))

    def test_set_piece_tags_with_unknown_team_and_conflicting_teams(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tags.xml")
            instances = []
            for team in ("N/A", "Barcelona", "Atletic Club"):
                instances.append(
                    "<instance><code>SET PIECES</code><start>0</start><end>2</end>"
                    "<label><group>Team</group><text>{}</text></label></instance>".format(team)
                )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("<root>{}</root>".format("".join(instances)))
            tags = _tag_evidence(path, 50, 25, ("FC Barcelona", "Athletic Club"))
        self.assertTrue(tags.set_piece.all())
        self.assertTrue(np.all(tags.restart_team == UNKNOWN))
        self.assertTrue(np.all(tags.evidence == 0))

    def test_build_submission_with_synthetic_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            tactical = os.path.join(folder, "tactical")
            os.makedirs(tactical)
            metadata = (
                '<root><Team id="home"><Name>FC Barcelona</Name></Team>'
                '<Team id="away"><Name>Athletic Club</Name></Team>'
                '<Player teamId="home"/><Player teamId="away"/></root>'
            )
            with open(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"),
                      "w", encoding="utf-8") as handle:
                handle.write(metadata)
            with open(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"),
                      "w", encoding="utf-8") as handle:
                for frame in range(1, 61):
                    handle.write("{}:0.11,0.5,1;0.8,0.5,1:0.1,0.5,1\n".format(frame))
            with open(os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml"),
                      "w", encoding="utf-8") as handle:
                handle.write("<root></root>")

            with patch.object(possession_solution, "N_FRAMES", 60):
                labels = possession_solution.build_submission(folder)

        self.assertEqual(len(labels), 60)
        self.assertTrue(np.all(labels == "H"))


if __name__ == "__main__":
    unittest.main()
