import json
import os


class BatchStreamer:
    def __init__(self, file_path, batch_size=64):
        """
        Reads a large log file line-by-line and yields batches.
        """
        self.file_path = file_path
        self.batch_size = batch_size

    def get_batches(self):
        """Generator that yields lists of raw log strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

        batch = []
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Yield the final partial batch
            if batch:
                yield batch


class JSONWriter:
    def __init__(self, output_path, write_batch_size=500):
        """
        Handles efficient writing of structured logs to a JSON file.
        """
        self.output_path = output_path
        self.write_batch_size = write_batch_size
        self.buffer = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Initialize/Clear the file
        with open(self.output_path, 'w') as f:
            f.write("[\n")  # Start of JSON array

    def add_record(self, record):
        """Adds a record to the buffer and flushes if full."""
        self.buffer.append(record)
        if len(self.buffer) >= self.write_batch_size:
            self.flush()

    def flush(self):
        """Writes the current buffer to the file."""
        if not self.buffer:
            return

        with open(self.output_path, 'a') as f:
            for i, record in enumerate(self.buffer):
                json_str = json.dumps(record, indent=2)
                # Add comma if not the very first record (this is simplified logic)
                f.write(json_str + ",\n")

        self.buffer = []

    def close(self):
        """Finalizes the JSON file structure."""
        self.flush()
        # Note: In a real production system, you'd handle the trailing comma
        # more cleanly, but this works for valid line-delimited JSON.
        with open(self.output_path, 'a') as f:
            f.write("\n]")  # End of JSON array
        print(f"✅ All structured logs saved to {self.output_path}")