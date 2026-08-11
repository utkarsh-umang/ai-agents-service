"""Per-job isolation for the short-video pipeline.

These tests exist because the original prototype shared three fixed working
directories between all jobs and emptied them at the start of every step. The
failure mode was not theoretical: two concurrent jobs would delete each other's
source video mid-ffmpeg. Everything here would fail against that design.

No network access — ffmpeg is stubbed.
"""

from __future__ import annotations

import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agents.agents.short_video_generator.helpers.cut_video import create_clips
from ai_agents.agents.short_video_generator.helpers.storage_keys import (
    short_video_clip_key,
    short_video_source_key,
)
from ai_agents.agents.short_video_generator.helpers.workspace import job_workspace


def _clip(start: str, end: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        title=f"clip {index}",
        topic="topic",
        why_it_works="because",
        start_timestamp=start,
        end_timestamp=end,
        duration_seconds=70,
    )


@pytest.fixture
def fake_ffmpeg(tmp_path: Path) -> Path:
    """An ffmpeg stand-in that creates whatever output path it is handed."""
    script = tmp_path / "fake_ffmpeg"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, pathlib\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'clip')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------

def test_workspace_is_removed_on_success():
    with job_workspace("job-1") as work:
        (work / "source.mp4").write_bytes(b"x")
        captured = work
    assert not captured.exists()


def test_workspace_is_removed_when_the_body_raises():
    """A failed job must not leak a partial download onto disk."""
    captured: Path | None = None
    with pytest.raises(RuntimeError):
        with job_workspace("job-boom") as work:
            captured = work
            (work / "half-downloaded.mp4").write_bytes(b"x" * 1024)
            raise RuntimeError("boom")
    assert captured is not None and not captured.exists()


def test_workspace_keep_leaves_it_behind(tmp_path: Path):
    with job_workspace("job-keep", parent=tmp_path, keep=True) as work:
        captured = work
    assert captured.exists()


def test_concurrent_workspaces_never_collide(tmp_path: Path):
    """The regression test. Each job writes its own id and must read it back.

    Under the old fixed-directory design every job wrote to the same path and
    unlinked the directory contents on entry, so this returns wrong data or
    raises FileNotFoundError.
    """

    def run(job_id: int) -> tuple[int, str]:
        with job_workspace(f"job-{job_id}", parent=tmp_path) as work:
            marker = work / "source.mp4"
            marker.write_text(str(job_id))
            # Yield to the other threads while "downloading".
            for _ in range(200):
                pass
            return job_id, marker.read_text()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(24)))

    for job_id, observed in results:
        assert observed == str(job_id), f"job {job_id} read job {observed}'s file"

    # Every workspace cleaned up after itself.
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# create_clips
# --------------------------------------------------------------------------

def test_create_clips_writes_only_into_the_given_directory(tmp_path, fake_ffmpeg):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    out = tmp_path / "clips"

    analysis = SimpleNamespace(
        clips=[_clip("00:00:10", "00:01:20", 0), _clip("00:02:00", "00:03:10", 1)]
    )
    results = create_clips(source, analysis, out, ffmpeg_bin=fake_ffmpeg)

    assert [r["index"] for r in results] == [0, 1]
    assert sorted(p.name for p in out.iterdir()) == ["clip_0.mp4", "clip_1.mp4"]
    # It must not have touched anything outside out_dir.
    assert source.exists()


def test_create_clips_does_not_delete_pre_existing_files(tmp_path, fake_ffmpeg):
    """The old implementation emptied its output directory on entry."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    out = tmp_path / "clips"
    out.mkdir()
    bystander = out / "another-jobs-clip.mp4"
    bystander.write_bytes(b"do not delete me")

    analysis = SimpleNamespace(clips=[_clip("00:00:10", "00:01:20", 0)])
    create_clips(source, analysis, out, ffmpeg_bin=fake_ffmpeg)

    assert bystander.read_bytes() == b"do not delete me"


def test_create_clips_surfaces_ffmpeg_failure(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    failing = tmp_path / "failing_ffmpeg"
    failing.write_text("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('bad timestamp\\n')\nsys.exit(1)\n")
    failing.chmod(failing.stat().st_mode | stat.S_IEXEC)

    analysis = SimpleNamespace(clips=[_clip("00:00:10", "00:01:20", 0)])
    with pytest.raises(RuntimeError, match="ffmpeg failed on clip 0"):
        create_clips(source, analysis, tmp_path / "clips", ffmpeg_bin=failing)


def test_create_clips_rejects_a_missing_source(tmp_path, fake_ffmpeg):
    analysis = SimpleNamespace(clips=[_clip("00:00:10", "00:01:20", 0)])
    with pytest.raises(FileNotFoundError):
        create_clips(tmp_path / "nope.mp4", analysis, tmp_path / "clips", ffmpeg_bin=fake_ffmpeg)


# --------------------------------------------------------------------------
# storage keys
# --------------------------------------------------------------------------

def test_clip_keys_are_deterministic_so_retries_overwrite():
    first = short_video_clip_key("folder-a", "job-1", 3)
    again = short_video_clip_key("folder-a", "job-1", 3)
    assert first == again == "short-video/folder-a/job-1/clips/3.mp4"


def test_clip_keys_separate_jobs_and_indexes():
    keys = {
        short_video_clip_key("f", "job-1", 0),
        short_video_clip_key("f", "job-1", 1),
        short_video_clip_key("f", "job-2", 0),
    }
    assert len(keys) == 3


def test_source_key_sits_under_its_own_expiring_prefix():
    assert short_video_source_key("job-1").startswith("short-video-inputs/")
