import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("OptionPricerTelemetry")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

@contextmanager
def measure_latency(batch_size: int, model_name: str):
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        end_time = time.perf_counter_ns()
        total_ns = end_time - start_time
        ns_per_option = total_ns / batch_size
        logger.info(f"[{model_name}] Priced {batch_size:,} options in {total_ns/1e6:.2f}ms ({ns_per_option:.2f} ns/option)")
