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

    def step(self):
        item = self.player.next()
        if item is None:
            return None
        snapshot = self.data.snapshot()
        profile = self.presenter.profile
        if profile is None:
            raise RuntimeError("Standalone presentation requires a render profile")
        artifact = self.renderer.render(item.screen_id, profile, self.preferences, snapshot)
        self.presenter.present(artifact)
        return artifact
