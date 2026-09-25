"""Runtime component composition, including the migration-era local mode."""
from __future__ import annotations

from dataclasses import dataclass

from display.hardware_presenter import HardwarePresenter
from playback.client_player import ClientPlayer
from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot
from services.data_coordinator import DataCoordinator


@dataclass
class LegacyStandaloneRuntime:
    """Compose refresh, rendering, playback and hardware in one process."""

    data: DataCoordinator
    renderer: ScreenRenderer
    player: ClientPlayer
    presenter: HardwarePresenter
    preferences: ServerPreferenceSnapshot

    def _hydrate_standings(self, screen_id: str) -> None:
        """Publish data needed by standings renderers before freezing a snapshot."""

        if screen_id.startswith("NFL ") and (
            "Standings" in screen_id or "Overview" in screen_id
        ):
            self.data.read_nfl_league_standings()
        elif screen_id.startswith("NHL Standings"):
            self.data.read_nhl_league_standings(
                include_wildcard_order=screen_id.endswith(" v2")
            )
        elif screen_id.startswith("MLB ") and "Standings" in screen_id:
            self.data.read_mlb_league_standings()

    def step(self):
        item = self.player.next()
        if item is None:
            return None
        self._hydrate_standings(item.screen_id)
        snapshot = self.data.snapshot()
        profile = self.presenter.profile
        if profile is None:
            raise RuntimeError("Standalone presentation requires a render profile")
        artifact = self.renderer.render(item.screen_id, profile, self.preferences, snapshot)
        self.presenter.present(artifact)
        return artifact
