import av
import collections
from dataclasses import dataclass
import enum
import logging
import numpy
import time


logger = logging.getLogger(__name__)


class InputPacketType(enum.Enum):
    AUDIO = 1
    VIDEO = 2


@dataclass
class InputPacket:
    type: InputPacketType
    data: numpy.ndarray
    pts: int
    idx: int


@dataclass
class InputAudioPacket(InputPacket):
    format: str
    layout: str
    sample_rate: int


class InputProcessor():

    def __init__(self, video_filename):
        self.video_filename = video_filename
        self.video_container = None
        self.packets = None
        self.frame_iter = None
        self.frame_idx = 0

        self.timecode = None
        self.read_video_frames = collections.deque()
        self.read_audio_frames = collections.deque()

        self._open()

    def _open(self):
        self.video_container = av.open(self.video_filename)
        self.packets = self.video_container.demux()
        self.frame_iter = iter(self.packets)
        logger.debug(f"Opened video {self.video_filename}")

    def ready(self):
        return self.timecode is not None

    # def get_video(self, frame_idx):
    #     while True:
    #         if len(self.read_video_frames) == 0:
    #             return None

    #         frame_idx_since_midnight = self.read_video_frames[0].idx + self.index_timecode[0]
    #         if frame_idx_since_midnight < frame_idx:
    #             self.read_video_frames.popleft()
    #         elif frame_idx_since_midnight == frame_idx:
    #             return self.read_video_frames.popleft()
    #         else:
    #             return None

    def pop_video(self):
        if len(self.read_video_frames) == 0:
            return None
        return self.read_video_frames.popleft()

    def pop_audio(self):
        if len(self.read_audio_frames) == 0:
            return None
        return self.read_audio_frames.popleft()

    def next(self):
        packet = None
        try:
            packet = next(self.frame_iter)
        except StopIteration:
            return False
        self._process_packet(packet)
        return True

    def _process_packet(self, packet):
        raw_frames = packet.decode()
        if packet.stream.type == 'data':
            if packet.stream.name is None:
                # raw_bytes = bytes(packet)
                # https://github.com/gopro/gpmf-parser
                # Raw data to be stored in Big Endian
                # frame_from_midnight = int.from_bytes(raw_bytes, "big")  # Skipping because of bad sync
                frame_from_midnight = 0
                self.timecode = frame_from_midnight
        elif packet.stream.type == 'video':
            for raw_frame in raw_frames:
                self.read_video_frames.append(raw_frame)
                self.frame_idx += 1
        elif packet.stream.type == 'audio':
            for raw_frame in raw_frames:
                self.read_audio_frames.append(raw_frame)
        else:
            logger.warning(f"Unhandled stream type: {packet.stream.type}")

    def close(self):
        if self.video_container:
            self.video_container.close()
            self.video_container = None


def process_input(video_filename, packet_queue):
    try:
        processor = InputProcessor(video_filename)

        video_idx = 0
        audio_idx = 0
        while processor.next():
            while packet_queue.full():
                time.sleep(0.1)
            v = processor.pop_video()
            if v is not None:
                p = InputPacket(type=InputPacketType.VIDEO, data=v.to_ndarray(format='bgr24'), pts=v.pts, idx=video_idx)
                video_idx += 1
                packet_queue.put(p)

            while packet_queue.full():
                time.sleep(0.1)
            a = processor.pop_audio()
            if a is not None:
                p = InputAudioPacket(type=InputPacketType.AUDIO, data=a.to_ndarray(), pts=a.pts, idx=audio_idx, format=a.format.name, layout=a.layout.name, sample_rate=a.sample_rate)
                audio_idx += 1
                packet_queue.put(p)
        processor.close()
    except Exception:
        logger.exception(f"Input processing {video_filename} caught exception")
