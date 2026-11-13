"""
VisionPort Video Compatibility Utility.

Provides helpers to encode H.264 video streams compatible with modern web browsers
(Chrome, Edge, Firefox, Safari) and applies MP4 FastStart (relocating the 'moov' atom
before the 'mdat' atom) so that HTML5 <video> elements can seek and play immediately.
"""

import logging
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

logger = logging.getLogger("VisionPort_VideoCompat")


def create_compatible_video_writer(
    output_path: Union[str, Path],
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Instantiates an OpenCV VideoWriter with H.264/AVC encoding prioritized
    for web browser playback.

    Tries Microsoft Media Foundation (MSMF) H264 on Windows first, then AVC1,
    before falling back to standard FourCC codes.

    Args:
        output_path: Destination path for the video file.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        cv2.VideoWriter: An opened video writer instance.

    Raises:
        RuntimeError: If no suitable video encoder could be initialized.
    """
    path_str = str(Path(output_path).resolve())
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        # (api_preference, fourcc_str, label)
        (cv2.CAP_MSMF, "H264", "MSMF H.264"),
        (cv2.CAP_MSMF, "avc1", "MSMF AVC1"),
        (0, "avc1", "Default AVC1"),
        (0, "H264", "Default H264"),
        (0, "mp4v", "Default MP4V Fallback"),
    ]

    for api_pref, codec_str, label in candidates:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            if api_pref != 0:
                writer = cv2.VideoWriter(path_str, api_pref, fourcc, float(fps), (int(width), int(height)))
            else:
                writer = cv2.VideoWriter(path_str, fourcc, float(fps), (int(width), int(height)))

            if writer.isOpened():
                logger.info(
                    "VideoWriter initialized with %s for '%s' (%dx%d @ %.1f FPS)",
                    label, Path(output_path).name, width, height, fps
                )
                return writer
            else:
                writer.release()
        except Exception as exc:
            logger.debug("Failed opening writer with %s: %s", label, exc)

    raise RuntimeError(
        f"Unable to initialize VideoWriter for '{output_path}'. "
        "No supported video encoder found."
    )


def relocate_moov_to_front(video_path: Union[str, Path]) -> bool:
    """
    FastStart relocation for MP4 files.
    Moves the 'moov' atom before the 'mdat' atom and updates sample table
    chunk offsets ('stco' and 'co64') by the byte length of the 'moov' atom.

    This allows HTML5 web video players (Chrome, Edge, Firefox) to buffer
    and seek without having to download or seek to the end of the file.

    Args:
        video_path: Path to the MP4 video file.

    Returns:
        bool: True if relocation succeeded or was already at front; False on error.
    """
    path = Path(video_path).resolve()
    if not path.is_file() or path.stat().st_size < 32:
        return False

    try:
        data = bytearray(path.read_bytes())
        file_len = len(data)
        pos = 0
        atoms = []

        while pos < file_len - 8:
            size = struct.unpack(">I", data[pos:pos + 4])[0]
            atom_type = bytes(data[pos + 4:pos + 8]).decode("latin1", errors="ignore")
            if size == 1:
                # 64-bit extended box size
                if pos + 16 > file_len:
                    break
                size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            if size <= 0 or pos + size > file_len:
                break
            atoms.append((atom_type, pos, size))
            pos += size

        atom_dict = {a[0]: a for a in atoms}
        if "moov" not in atom_dict or "mdat" not in atom_dict:
            # Not a standard MP4 or missing required atoms
            return False

        moov_info = atom_dict["moov"]
        mdat_info = atom_dict["mdat"]

        # If moov is already before mdat, file is already faststart-ready
        if moov_info[1] < mdat_info[1]:
            return True

        moov_pos, moov_size = moov_info[1], moov_info[2]
        moov_bytes = bytearray(data[moov_pos:moov_pos + moov_size])

        # Adjust chunk offsets inside 'stco' (32-bit) and 'co64' (64-bit) boxes
        idx = 0
        while idx < len(moov_bytes) - 8:
            box_header = moov_bytes[idx:idx + 4]
            if box_header == b"stco":
                # 'stco': 4-byte name, 1-byte version, 3-byte flags, 4-byte entry count
                if idx + 12 <= len(moov_bytes):
                    entry_count = struct.unpack(">I", moov_bytes[idx + 8:idx + 12])[0]
                    table_start = idx + 12
                    for i in range(entry_count):
                        off_idx = table_start + i * 4
                        if off_idx + 4 <= len(moov_bytes):
                            cur_offset = struct.unpack(">I", moov_bytes[off_idx:off_idx + 4])[0]
                            moov_bytes[off_idx:off_idx + 4] = struct.pack(">I", cur_offset + moov_size)
                    idx = table_start + entry_count * 4
                else:
                    idx += 1
            elif box_header == b"co64":
                # 'co64': 4-byte name, 1-byte version, 3-byte flags, 4-byte entry count
                if idx + 12 <= len(moov_bytes):
                    entry_count = struct.unpack(">I", moov_bytes[idx + 8:idx + 12])[0]
                    table_start = idx + 12
                    for i in range(entry_count):
                        off_idx = table_start + i * 8
                        if off_idx + 8 <= len(moov_bytes):
                            cur_offset = struct.unpack(">Q", moov_bytes[off_idx:off_idx + 8])[0]
                            moov_bytes[off_idx:off_idx + 8] = struct.pack(">Q", cur_offset + moov_size)
                    idx = table_start + entry_count * 8
                else:
                    idx += 1
            else:
                idx += 1

        before_mdat = data[:mdat_info[1]]
        mdat_and_others = data[mdat_info[1]:moov_pos] + data[moov_pos + moov_size:]
        path.write_bytes(before_mdat + moov_bytes + mdat_and_others)
        logger.info("Successfully relocated 'moov' atom to front of '%s'", path.name)
        return True

    except Exception as exc:
        logger.warning("FastStart moov relocation skipped on '%s': %s", Path(video_path).name, exc)
        return False


def finalize_video_file(video_path: Union[str, Path]) -> bool:
    """
    Finalizes an MP4 video file by applying FastStart moov relocation.
    Call this immediately after writer.release().
    """
    return relocate_moov_to_front(video_path)

