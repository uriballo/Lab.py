from src.Lab import ExperimentRunner, load_config_file

if __name__ == "__main__":
    config = load_config_file("sample_config.json")
    runner = ExperimentRunner(config)
    runner.run_all_experiments()