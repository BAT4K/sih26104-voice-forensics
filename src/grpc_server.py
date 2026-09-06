import sys
from pathlib import Path
import os
import uuid
import time
import numpy as np
import logging
from concurrent import futures
import grpc

# Setup sys.path to find proto module
sys.path.append(str(Path(__file__).resolve().parent.parent / "proto"))

import vfd_pb2
import vfd_pb2_grpc
from src.inference import analyse, StreamingScorer
from src.api import _decode
from src.risk import CallContext

log = logging.getLogger("vfd.grpc")
logging.basicConfig(level=logging.INFO)

class VoiceForensicServicer(vfd_pb2_grpc.VoiceForensicServiceServicer):
    
    def AnalyzeAudio(self, request, context):
        rid = request.request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        
        try:
            # Decode audio bytes in memory
            x, sr = _decode(request.audio_chunk)
            
            call_ctx = CallContext(claimed_identity=request.speaker_id)
            result = analyse(x, sr, speaker_id=request.speaker_id, context=call_ctx)
            
            log.info(f"gRPC Analyze [{rid}] - verdict={result.verdict} risk={result.risk_score}")
            
            return vfd_pb2.AnalyzeResponse(
                request_id=rid,
                verdict=result.verdict,
                risk_score=result.risk_score,
                is_fake=result.verdict == "AI-generated voice",
                recommended_action=result.recommended_action
            )
        except Exception as e:
            log.error(f"gRPC Analyze Error [{rid}]: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vfd_pb2.AnalyzeResponse()

    def StreamAudio(self, request_iterator, context):
        rid = uuid.uuid4().hex[:12]
        scorer = StreamingScorer()
        log.info(f"gRPC Stream [{rid}] started")
        
        for req in request_iterator:
            if req.reset:
                scorer.reset()
                yield vfd_pb2.StreamResponse(request_id=rid, status="RESET")
                continue
                
            if not req.audio_chunk:
                continue
                
            chunk = np.frombuffer(req.audio_chunk, dtype=np.float32)
            try:
                out = scorer.push(chunk, 16000) # Target SR
                if out is not None:
                    yield vfd_pb2.StreamResponse(
                        request_id=rid,
                        risk_score=out.get("risk_score", 0.0),
                        is_fake=out.get("is_fake", False),
                        status="OK"
                    )
            except Exception as e:
                log.error(f"gRPC Stream Error [{rid}]: {e}")
                break
                
        log.info(f"gRPC Stream [{rid}] ended")

def serve():
    port = os.environ.get("VFD_GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vfd_pb2_grpc.add_VoiceForensicServiceServicer_to_server(VoiceForensicServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info(f"gRPC Server listening on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
