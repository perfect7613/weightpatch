"""Start an episode on the deployed Modal app and return at once.

    modal deploy -m weightpatch.worker.modal_app
    python scripts/spawn_episode.py configs/episodes/tiny-aya-gaming.yaml run-name

The call is spawned, so the run does not depend on this process or terminal.
"""
import sys
from pathlib import Path

import modal

spec, name = sys.argv[1], sys.argv[2]
call = modal.Function.from_name("weightpatch", "episode").spawn(Path(spec).read_text(), name=name)
print("spawned", call.object_id)
