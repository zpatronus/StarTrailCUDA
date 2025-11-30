import subprocess
import time
import os
import glob


def find_startrail_executable():
    # Search for executables starting with 'startrail' in the current directory and subdirectories
    matches = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("startrail") and os.access(
                os.path.join(root, file), os.X_OK
            ):
                matches.append(os.path.join(root, file))
    return matches


def prompt_algorithm():
    while True:
        choice = input(
            "Select algorithm to run (1 for LINEARAPPROX, 2 for LINEAR): "
        ).strip()
        if choice == "1":
            return "LINEARAPPROX"
        elif choice == "2":
            return "LINEAR"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def run_command():
    # Find the executable
    executables = find_startrail_executable()
    if not executables:
        print("No executable starting with 'startrail' found.")
        return
    elif len(executables) > 1:
        print("Multiple executables found. Please select one:")
        for idx, exe in enumerate(executables, 1):
            print(f"{idx}: {exe}")
        choice = int(input("Enter the number of the executable to use: ")) - 1
        if choice < 0 or choice >= len(executables):
            print("Invalid selection.")
            return
        executable = executables[choice]
    else:
        executable = executables[0]

    # Prompt for algorithm
    algorithm = prompt_algorithm()

    base_command = [
        executable,
        "-i",
        "../video_download/test_starrail_15sec_h264.mp4",
        "-o",
        "output.mp4",
        "-a",
        algorithm,
    ]

    # Format the command for display
    command_str = " ".join(base_command)

    runtimes = []

    for i in range(5):
        print(f"\nRunning iteration {i+1}:")
        print(f"Command: {command_str}")

        # Use Python's time module to measure runtime
        start_time = time.time()
        result = subprocess.run(
            base_command, capture_output=True, text=True, shell=False
        )
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"Iteration {i+1} runtime: {runtime:.2f} seconds")

        if result.returncode != 0:
            print(f"Error in iteration {i+1}: {result.stderr}")

    # Filter out any failed runs and calculate average of middle 3 valid runs
    valid_runtimes = [rt for rt in runtimes if rt is not None]

    if len(valid_runtimes) >= 3:
        middle_runtimes = valid_runtimes[
            1:4
        ]  # indices 1, 2, 3 (2nd, 3rd, 4th runs)
        average_runtime = sum(middle_runtimes) / len(middle_runtimes)

        print("\n" + "=" * 50)
        print("FINAL RESULTS:")
        print("=" * 50)
        for i, rt in enumerate(runtimes, 1):
            status = f"{rt:.2f}s" if rt is not None else "Failed"
            print(f"Run {i}: {status}")

        print(f"\nAverage of middle 3 runs: {average_runtime:.2f} seconds")
    else:
        print("Not enough successful runs to calculate average")


if __name__ == "__main__":
    run_command()
