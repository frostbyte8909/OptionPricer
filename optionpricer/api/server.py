import grpc
from concurrent import futures
import time
import numpy as np

import optionpricer.api.pricer_pb2 as pricer_pb2
import optionpricer.api.pricer_pb2_grpc as pricer_pb2_grpc

from optionpricer.core import OptionContract, MarketState, ExoticType, OptionType
from optionpricer.models.black_scholes import black_scholes
from optionpricer.models.monte_carlo import price_exotic
from optionpricer.models.bjerksund_stensland import bjerksund_stensland_american
from optionpricer.models.fdm import crank_nicolson_fdm
from optionpricer.telemetry import measure_latency, logger
from optionpricer.exceptions import InvalidInputError

class OptionPricerServicer(pricer_pb2_grpc.OptionPricerServiceServicer):
    
    def _map_exotic_type(self, pb_exotic) -> ExoticType:
        mapping = {
            pricer_pb2.VANILLA: ExoticType.VANILLA,
            pricer_pb2.ASIAN: ExoticType.ASIAN,
            pricer_pb2.LOOKBACK: ExoticType.LOOKBACK,
            pricer_pb2.BARRIER_UI: ExoticType.BARRIER_UI,
            pricer_pb2.BARRIER_UO: ExoticType.BARRIER_UO,
            pricer_pb2.BARRIER_DI: ExoticType.BARRIER_DI,
            pricer_pb2.BARRIER_DO: ExoticType.BARRIER_DO
        }
        return mapping.get(pb_exotic, ExoticType.VANILLA)

    def _map_option_type(self, pb_option) -> OptionType:
        return OptionType.CALL if pb_option == pricer_pb2.CALL else OptionType.PUT

    def PriceBatch(self, request, context):
        try:
            S = np.array(request.market.spot)
            K = np.array(request.contract.strike)
            T = np.array(request.contract.expiry)
            r = np.array(request.market.rate)
            sig = np.array(request.market.volatility)
            q = np.array(request.market.dividend) if request.market.dividend else np.zeros_like(S)
            
            H = np.array(request.contract.barrier_level) if request.contract.barrier_level else np.zeros_like(S)
            rebate = np.array(request.contract.rebate) if request.contract.rebate else np.zeros_like(S)
            
            option_type = self._map_option_type(request.contract.option_type)
            exotic_type = self._map_exotic_type(request.contract.exotic_type)
            american = request.contract.american
            
            contract = OptionContract(
                strike=K, expiry=T, option_type=option_type, american=american,
                exotic_type=exotic_type, barrier_level=H, rebate=rebate
            )
            market = MarketState(spot=S, rate=r, volatility=sig, dividend=q)
            
            batch_size = max(len(S), len(K))
            
            start_ns = time.perf_counter_ns()
            
            prices = None
            greeks = {}
            
            if exotic_type == ExoticType.VANILLA:
                if american:
                    prices = bjerksund_stensland_american(contract, market)
                else:
                    if request.model == "fdm":
                        prices = crank_nicolson_fdm(contract, market)
                    else:
                        out = black_scholes(contract, market, return_greeks=True)
                        if isinstance(out, tuple):
                            prices, greeks_dict = out
                            greeks = {k: v.item() if np.isscalar(v) else v[0] for k, v in greeks_dict.items()}
                        else:
                            prices = out
            else:
                prices = price_exotic(contract, market)

            end_ns = time.perf_counter_ns()
            ns_per_option = (end_ns - start_ns) / max(1, batch_size)
            
            prices_list = [prices] if np.isscalar(prices) else prices.ravel().tolist()
            
            logger.info(f"gRPC Batch Processed: {batch_size} options at {ns_per_option:.2f} ns/opt")
            
            return pricer_pb2.PricingResponse(
                prices=prices_list,
                greeks=greeks,
                compute_time_ns_per_option=ns_per_option
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise e

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pricer_pb2_grpc.add_OptionPricerServiceServicer_to_server(OptionPricerServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"OptionPricer gRPC Engine listening on port {port}...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
