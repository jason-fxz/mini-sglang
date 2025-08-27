import atexit
import logging
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)


class SafeProfiler:
    def __init__(self, trace_dir="traces", activities=None, **kwargs):
        self.trace_dir = trace_dir
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.kwargs = dict(record_shapes=True, profile_memory=True, with_stack=False)
        self.kwargs.update(kwargs)
        self.prof = None
        self._flushed = False

    def __enter__(self):
        self.prof = profile(activities=self.activities, **self.kwargs)
        self.prof.__enter__()  # 手动进入上下文
        _install_handlers(self.flush)  # 安装信号与 atexit 兜底
        return self.prof

    def __exit__(self, exc_type, exc, tb):
        self.flush()
        # 继续传播异常（如果有）
        return False

    def flush(self):
        if self._flushed:
            return
        try:
            # 确保 CUDA 队列全部提交/完成，trace 才完整
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        try:
            # 停止并导出
            if self.prof is not None:
                # 若用 tensorboard_trace_handler 可在这里调用它；此处导出 chrome trace
                self.prof.stop()
                # 给文件名加上 rank 和时间，避免多进程/多次覆盖
                ts = time.strftime("%Y%m%d-%H%M%S")
                tp = "" if dist.get_world_size() == 1 else f"_TP{dist.get_rank()}"
                tracename = f"trace_{ts}{tp}.json"
                tracepath = os.path.join(self.trace_dir, tracename)
                # 确保路径存在
                os.makedirs(self.trace_dir, exist_ok=True)
                self.prof.export_chrome_trace(tracepath)
                logger.info(f"[profiler] exported: {tracepath}")
        finally:
            self._flushed = True


def _install_handlers(flush_fn):
    # 1) atexit 兜底
    atexit.unregister  # hint for some IDE linters
    atexit.register(flush_fn)

    # 2) 信号处理：SIGINT/SIGTERM 时优先 flush
    def _handler(signum, frame):
        try:
            flush_fn()
        finally:
            # 恢复默认行为并重新发送信号给本进程，保持正确退出码
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
