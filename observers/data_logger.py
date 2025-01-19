from observers.observer import Observer

class DataLogger(Observer):
    def __init__(self):
        super().__init__()
        self.data_log = []

    def update(self, data: dict):
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        data["rotation_vector"] = list(data["rotation_vector"])
        self.data_log.append(
            data.update({"timestamp": timestamp})

    def save_to_file(self, filename="session_data.json"):
        import json
        with open(filename, "w") as f:
            json.dump(self.data_log, f, indent=4)
        print(f"Data saved to {filename}")

    def load_from_file(self, filename="session_data.json"):
        import json
        with open(filename, "r") as f:
            self.data_log = json.load(f)
        print(f"Data loaded from {filename}")
