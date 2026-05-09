import os
import glob
from features.engine.stream_handler import BatchStreamer, JSONWriter

def test_stream_and_write():
    # Use one of the real 2k log files to test streamer
    data_dir = "data/logs"
    log_files = glob.glob(os.path.join(data_dir, "*_2k.log"))
    
    if not log_files:
        print("[WARNING] No real logs to stream, skipping data test.")
        log_file = "dummy.log"
        with open(log_file, "w") as f:
            for i in range(20):
                f.write(f"Line {i}\n")
    else:
        log_file = log_files[0]
        
    print(f"Testing Streamer on {log_file}")
    streamer = BatchStreamer(log_file, batch_size=10)
    batches = list(streamer.get_batches())
    
    print(f"Total Batches: {len(batches)}")
    print(f"First Batch Size: {len(batches[0])}")
    
    if len(batches) > 0:
        assert len(batches[0]) == 10
    
    # 2. Test Writing
    output_f = "output/test_output.json"
    writer = JSONWriter(output_f, write_batch_size=5)
    for i in range(12):
        writer.add_record({"id": i, "content": "test log"})
    writer.close()
    
    assert os.path.exists(output_f)
    print("[OK] Stream and write testing passed!")
    
    # Cleanup
    if os.path.exists("dummy.log"):
        os.remove("dummy.log")
    if os.path.exists(output_f):
        os.remove(output_f)

if __name__ == "__main__":
    test_stream_and_write()