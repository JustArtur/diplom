from __future__ import annotations

MIN_HEIGHT = 60.0
# верх ERA5 ~50 гПа, около 20 км
MAX_HEIGHT = 20000.0
# потолок шара: штраф в reward, позиция клампится по ERA5
BALLOON_MAX_ALTITUDE = 24_000.0
# лимит vz чтобы не разъехалась симуляция
MAX_VERTICAL_SPEED = 50.0
