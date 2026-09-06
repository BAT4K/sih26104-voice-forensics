import grpc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "proto"))

import vfd_pb2
import vfd_pb2_grpc

def run():
    print("Connecting to gRPC server at localhost:50051...")
    # NOTE: In production telecom integration, use secure_channel with TLS certificates.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = vfd_pb2_grpc.VoiceForensicServiceStub(channel)
        
        print("Sending dummy audio bytes for AnalyzeAudio...")
        # Simulating a 1-second 16kHz mono float32 audio chunk
        dummy_audio = (b'\x00' * (16000 * 4)) 
        
        request = vfd_pb2.AnalyzeRequest(
            audio_chunk=dummy_audio,
            request_id="test-demo-123",
            speaker_id="cfo_001"
        )
        
        try:
            response = stub.AnalyzeAudio(request)
            print("\n=== gRPC Analysis Result ===")
            print(f"Request ID: {response.request_id}")
            print(f"Verdict: {response.verdict}")
            print(f"Risk Score: {response.risk_score}")
            print(f"Is Fake: {response.is_fake}")
            print(f"Recommended Action: {response.recommended_action}")
            print("============================\n")
        except grpc.RpcError as e:
            print(f"gRPC Call Failed: {e.details()}")

if __name__ == '__main__':
    run()
