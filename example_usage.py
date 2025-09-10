# #!/usr/bin/env python3
# """
# Example usage of the new configuration system.

# This script demonstrates how to use the modular configuration system
# with different feature extractors and classifiers.
# """

# import subprocess
# import sys
# import os

# def run_experiment(command, description):
#     """Run an experiment with the given command and description."""
#     print(f"\n{'='*60}")
#     print(f"Running: {description}")
#     print(f"Command: {command}")
#     print(f"{'='*60}")
    
#     try:
#         result = subprocess.run(command, shell=True, capture_output=True, text=True)
#         if result.returncode == 0:
#             print("✅ Experiment completed successfully!")
#             print("Output:")
#             print(result.stdout)
#         else:
#             print("❌ Experiment failed!")
#             print("Error:")
#             print(result.stderr)
#     except Exception as e:
#         print(f"❌ Error running experiment: {e}")

# def main():
#     """Run example experiments with different configurations."""
    
#     print("🚀 Plaque Analysis - Configuration System Examples")
#     print("This script demonstrates different configurations for the plaque analysis project.")
    
#     # Example 1: Basic usage with defaults
#     run_experiment(
#         "python main_new.py --num_epochs 2",  # Reduced epochs for quick demo
#         "Basic usage with default configurations (simple_cnn + custom_mlp)"
#     )
    
#     # Example 2: ResNet transfer learning
#     run_experiment(
#         "python main_new.py --feature_extractor resnet18 --freeze_feature_extractor --num_epochs 2",
#         "ResNet-18 transfer learning with frozen feature extractor"
#     )
    
#     # Example 3: Deep MLP classifier
#     run_experiment(
#         "python main_new.py --classifier deep_mlp --num_epochs 2",
#         "Simple CNN with deep MLP classifier"
#     )
    
#     # Example 4: Custom training parameters
#     run_experiment(
#         "python main_new.py --feature_extractor resnet50 --classifier simple_mlp --batch_size 16 --learning_rate 0.0005 --num_epochs 2",
#         "ResNet-50 with simple MLP and custom training parameters"
#     )
    
#     # Example 5: Custom feature extractor parameters
#     run_experiment(
#         "python main_new.py --feature_extractor simple_cnn --num_output_features 64 --classifier custom_mlp --num_epochs 2",
#         "Simple CNN with 64 output features"
#     )
    
#     print(f"\n{'='*60}")
#     print("🎉 All example experiments completed!")
#     print("Check the 'reports' folder for detailed results.")
#     print(f"{'='*60}")

# if __name__ == "__main__":
#     # Check if main_new.py exists
#     if not os.path.exists("main_new.py"):
#         print("❌ Error: main_new.py not found!")
#         print("Please make sure you're in the correct directory.")
#         sys.exit(1)
    
#     # Check if config files exist
#     config_files = [
#         "configs/general_config.json",
#         "configs/feature_extractor_config.json", 
#         "configs/classifier_config.json"
#     ]
    
#     missing_files = [f for f in config_files if not os.path.exists(f)]
#     if missing_files:
#         print("❌ Error: Missing configuration files:")
#         for f in missing_files:
#             print(f"  - {f}")
#         print("Please make sure all configuration files are present.")
#         sys.exit(1)
    
#     main()
