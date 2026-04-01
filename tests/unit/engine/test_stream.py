# test_stream.py
from src.features.engine.stream_handler import BatchStreamer, JSONWriter

# 1. Test Streaming
streamer = BatchStreamer("data/logs/Hadoop/Hadoop_2k.log", batch_size=10)
batches = list(streamer.get_batches())
print(f"Total Batches: {len(batches)}") # Should be 200 for a 2k file
print(f"First Batch Size: {len(batches[0])}")

# 2. Test Writing
writer = JSONWriter("output/test_output.json", write_batch_size=5)
for i in range(12):
    writer.add_record({"id": i, "content": "test log"})
writer.close()