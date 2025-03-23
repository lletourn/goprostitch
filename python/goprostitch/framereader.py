import av  # type: ignore
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import numpy as np
import operator
import subprocess
import sys
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Self
from typing import Type
from types import TracebackType

logger = logging.getLogger(__name__)

NB_SEEK_RETRIES = 3


@dataclass(slots=True)
class FrameIndex:
    frame: int
    timestamp: float
    raw_timestamp: float
    # Types: https://ffmpeg.org/doxygen/2.7/structAVFrame.html
    pkt_pts: Optional[int] = None
    pkt_dts: Optional[int] = None
    pkt_dts_time: Optional[float] = None
    best_effort_timestamp_time: Optional[float] = None
    pkt_duration_time: Optional[float] = None
    pict_type: Optional[str] = None
    coded_picture_number: Optional[int] = None
    stream_index: Optional[int] = None


@dataclass
class FrameReaderFrame:
    frame_id: int
    frame_data: av.frame.Frame
    frame: np.ndarray


def get_ffprobe_json_output(cmd_args: List[str]) -> Dict[str, Any]:
    data_format = None

    cmd = ['ffprobe'] + cmd_args
    logger.info("Running cmd: %s", ' '.join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    outs, errs = process.communicate()
    if process.returncode != 0:
        logger.error("Command {} failed with returncode {} Error: {}".format(cmd, process.returncode, errs))
        raise ValueError("Failed with returncode {} Error: {}".format(process.returncode, errs))
    elif errs:
        logger.warning(f"Command finished running with error code 0 but has error message: {errs}")

    data_format = {}
    if outs is not None:
        data_format = json.loads(outs)
    return data_format


def _get_stream_index(video: str, video_stream_index: int) -> str:
    cmd_format = ['-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=index', '-of', 'json', video]
    video_streams = get_ffprobe_json_output(cmd_format)
    if len(video_streams["streams"]) < video_stream_index:
        raise ValueError(f'video contains {len(video_streams["streams"])} video streams, asking for video stream index {video_stream_index}')

    return str(video_streams["streams"][video_stream_index]["index"])


def build_frame_map(video: str, stream_idx: int) -> Dict[int, FrameIndex]:
    selected_stream = _get_stream_index(video, stream_idx)

    cmd_format = ['-v', 'error', '-print_format', 'json=compact=1', '-show_format', video]
    container_format = get_ffprobe_json_output(cmd_format)
    video_start_time = float(container_format['format']['start_time'])

    cmd_args = None
    cmd_args = ['-v', 'error', '-print_format', 'json=compact=1', '-select_streams', selected_stream, '-show_packets', video]
    frames_data = get_ffprobe_json_output(cmd_args)

    frames = OrderedDict()
    try:
        all_frames = list()
        for packet_data in frames_data['packets']:
            if packet_data['flags'][-1] == '_':
                frame = FrameIndex(frame=-1, timestamp=float(packet_data['pts_time']) - video_start_time, raw_timestamp=float(packet_data['pts_time']))
                frame.stream_index = packet_data['stream_index']
                frame.pkt_pts = int(packet_data['pts'])
                if 'dts_time' in packet_data:
                    frame.pkt_dts_time = float(packet_data['dts_time']) - video_start_time
                if 'dts' in packet_data:
                    frame.pkt_dts = int(packet_data['dts'])
                if 'duration_time' in packet_data:
                    frame.pkt_duration_time = float(packet_data['duration_time'])
                frame.best_effort_timestamp_time = None
                all_frames.append(frame)

        # Ordering is necessary because of B-Frames. A future frame can be put before a current frame.
        # In short, videos are DTS ordered, not PTS ordered.
        all_frames = sorted(all_frames, key=operator.attrgetter("pkt_pts"))
        for frame_idx, frame_data in enumerate(all_frames):
            frame_data.frame = frame_idx
            frames[frame_idx] = frame_data

    except Exception:
        logger.exception("Exception while probing video %s.", video)
        raise

    return frames


class FrameReader:
    def __init__(
            self,
            video_filename: str,
            max_errors: int = 5,
            frame_skips_before_seek: int = 500):
        self.__video_filename = video_filename
        self.__max_errors = max_errors
        self.__frame_skips_before_seek = frame_skips_before_seek

        # This is temporary. As soon as open is called, it will be filled out
        self.__frame_to_timestamp: Dict[int, FrameIndex] = dict()

        # Frames are a list because 2 frames can have the same pts
        # Players can use them in any order, so our frame_id -> pts is best effort and might not match the order used by players
        self.__pts_to_frame_id: DefaultDict[int, List[int]] = defaultdict(list)

        self.__video_container: Optional[av.InputContainer] = None  # type: ignore
        self.__video_stream: Optional[av.Stream] = None  # type: ignore

        self.__last_decoded_frame: Optional[FrameReaderFrame] = None
        self.__first_frame_pts: int = -1 * sys.maxsize

    def open(self) -> None:
        self.__video_container = av.open(self.__video_filename)
        self.__video_stream = self.__video_container.streams.video[0]

        if len(self.__frame_to_timestamp) == 0:
            self.build_frame_map()

        for frame_id, frame_index_details in self.__frame_to_timestamp.items():
            assert frame_index_details.pkt_pts is not None, "We don't support None pts"
            self.__pts_to_frame_id[frame_index_details.pkt_pts].append(frame_id)

        assert 0 in self.__frame_to_timestamp, "Frame zero isn't in the dict. We can only work with complete frame to timestamp maps"
        assert self.__frame_to_timestamp[0].pkt_pts is not None, "We don't support None pts"
        self.__first_frame_pts = self.__frame_to_timestamp[0].pkt_pts

    def close(self) -> None:
        if self.__video_container:
            self.__video_container.close()
        self.__video_container = None
        self.__video_stream = None
        self.__last_decoded_frame = None
        self.__first_frame_pts = -1 * sys.maxsize

    def __enter__(self: Self) -> Self:
        if self.__video_container is None:
            self.open()
        return self

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def frame_id(self) -> int:
        if self.__last_decoded_frame:
            if self.__last_decoded_frame.frame_id + 1 >= self.get_frame_count():
                return -1
            return self.__last_decoded_frame.frame_id + 1
        return 0

    @property
    def fps(self) -> float:
        """
            Don't call this on VariableFrameRate videos!
        """
        assert self.__video_stream, "Video needs to be opened"
        return self.__video_stream.guessed_rate

    @property
    def width(self) -> int:
        assert self.__video_stream, "Video needs to be opened"
        return self.__video_stream.codec_context.width

    @property
    def height(self) -> int:
        assert self.__video_stream, "Video needs to be opened"
        return self.__video_stream.codec_context.height

    def build_frame_map(self) -> Dict[int, FrameIndex]:
        """
            Until this PR in PyAV is merged
            https://github.com/PyAV-Org/PyAV/pull/1138
            we can't use PyAV to build frame to pts mappings, we need to rely on ffprobe
        """
        assert self.__video_stream, "Video needs to be opened"
        self.__frame_to_timestamp = build_frame_map(self.__video_filename, self.__video_stream.index)
        assert self.__frame_to_timestamp[0].pkt_pts is not None, "We don't support None pts"
        self.__first_frame_pts = self.__frame_to_timestamp[0].pkt_pts

        return self.__frame_to_timestamp

    def get_frame_count(self) -> int:
        return len(self.__frame_to_timestamp)

    def get_frame(self, frame_id: int) -> FrameReaderFrame:
        needs_seek = False
        if self.__last_decoded_frame:
            if frame_id == self.__last_decoded_frame.frame_id:
                return self.__last_decoded_frame

            if self.__last_decoded_frame.frame_id + 1 == frame_id:
                self.__next_frame()
            else:  # seek
                needs_seek = True
        else:
            needs_seek = True

        if needs_seek:
            self.__seek(frame_id)

        assert self.__last_decoded_frame, "A frame needs to be decoded here"
        return self.__last_decoded_frame

    def __next_frame(self) -> None:
        assert self.__video_container, "Video needs to be opened"
        assert self.__video_stream, "Video needs to be opened"

        for frame in self.__video_container.decode(self.__video_stream):
            frame_id = self.__pts_to_frame_id[frame.pts][0]
            # if there are multiple frames on that pts:
            #   if there are no previously decoded frames, we are at the first
            #   if the previously decoded frame is not on this pts YET, we are on the 1st frame of the pts
            #   if the previously decoded frame is on this pts, find the next frame id of the pts
            if len(self.__pts_to_frame_id[frame.pts]) > 1:
                if self.__last_decoded_frame is not None and self.__last_decoded_frame.frame_data.pts == frame.pts:
                    for frame_id in self.__pts_to_frame_id[frame.pts]:
                        if frame_id > self.__last_decoded_frame.frame_id:
                            break

            self.__last_decoded_frame = FrameReaderFrame(frame_id=frame_id, frame_data=frame, frame=frame.to_ndarray(format="bgr24"))
            break
        assert self.__last_decoded_frame, "Couldn't decode a single frame"

        assert self.__last_decoded_frame.frame_data.pts is not None, "We don't support None pts"
        assert self.__last_decoded_frame.frame_data.time is not None, "We don't support None frame time"

    def __seek(self, frame_id: int) -> None:
        assert self.__video_container, "Video needs to be opened"

        starting_frame_id = frame_id

        found_frame = False
        for retry_seek in range(NB_SEEK_RETRIES):
            logger.debug("Seeking to: %s", starting_frame_id)
            pkt_pts = self.__frame_to_timestamp[starting_frame_id].pkt_pts
            assert pkt_pts is not None, "We don't handle None pts"

            current_frame_id = 0
            if self.__last_decoded_frame:
                current_frame_id = self.__last_decoded_frame.frame_id

            if (frame_id >= current_frame_id and (frame_id-current_frame_id) > self.__frame_skips_before_seek) or frame_id < current_frame_id:
                self.__video_container.seek(pkt_pts, backward=True, any_frame=False, stream=self.__video_stream)

            while True:
                self.__next_frame()

                assert self.__last_decoded_frame, "Frame should have been decoded"
                if self.__last_decoded_frame.frame_id == frame_id:
                    found_frame = True
                    break
                elif self.__last_decoded_frame.frame_id > frame_id:
                    logger.error("We are past the seeked frame?! Target: %s Current: %s", frame_id, self.__last_decoded_frame.frame_id)
                    starting_frame_id -= self.__frame_skips_before_seek
                    break

            if found_frame:
                break

        if not found_frame:
            raise RuntimeError(f"Couldn't seek to frame: {frame_id}")

    def release(self) -> None:
        self.close()

    def reset_video(self) -> None:
        self.close()
        self.open()

    def restart_from_zero(self) -> None:
        assert self.__video_container, "Video needs to be opened"
        assert self.__video_stream, "Video needs to be opened"
        self.__video_container.seek(self.__first_frame_pts, backward=True, any_frame=False, stream=self.__video_stream)
        self.__last_decoded_frame = None
