import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import sys
import random
from PIL import Image
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--labeled_plaques_folder", type=str, default="labeled_plaques")
    parser.add_argument(
        "--unlabeled_plaques_folder", type=str, default="unlabeled_plaques"
    )
    parser.add_argument(
        "--unlabeled_sample_size",
        type=int,
        default=2000,
        help="Number of unlabeled plaques to sample (default: 2000)",
    ),
    parser.add_argument("--labeled_result_folder", type=str, default="labeled_images"),
    parser.add_argument(
        "--unlabeled_result_folder", type=str, default="sampled_unlabeled_images"
    ),
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output_sampled_data_table_file",
        type=str,
        default="data_table_sampled.csv",
        help="Output CSV file name (default: data_table_sampled.csv)",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default="labelfileidx.npz",
        help="Label file name (default: labelfileidx.npz)",
    ),
    parser.add_argument(
        "--label_names_file",
        type=str,
        default="label_names.csv",
        help="Label names file name (default: label_names.csv)",
    ),
    parser.add_argument(
        "--save_images",
        type=bool,
        default=True,
        help="Save downsampled images to folders",
    )
    parser.add_argument(
        "--clear_intermediate",
        type=bool,
        default=True,
        help="Clear previous intermediate results (CSV and image folders) before running",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Append checkpoint to CSV every N newly processed rows (default: 500)",
    )
    return parser.parse_args()


def get_total_plaques_count(data_folder, unlabeled_folder):
    """Count total number of plaques across all files"""
    files = os.listdir(os.path.join(data_folder, unlabeled_folder))
    total_plaques = 0

    print("Counting total plaques...")
    for image in tqdm(files, desc="Counting plaques"):
        file_path = os.path.join(data_folder, unlabeled_folder, image)
        try:
            with h5py.File(file_path, "r") as f:
                total_plaques += f["plaques"].attrs["length"]
        except Exception as e:
            print(f"Error counting plaques in {image}: {e}")

    return total_plaques, files


def create_data_table(
    data_folder,
    labeled_folder,
    unlabeled_folder,
    unlabeled_sample_size,
    label_file,
    random_seed=42,
    save_images=False,
    labeled_result_folder=None,
    unlabeled_result_folder=None,
    label_names_file=None,
    output_path=None,
    clear_intermediate=False,
    checkpoint_every=500,
):
    """Create data table with labeled samples from npz file and sampled unlabeled data, optionally saving images"""
    random.seed(random_seed)

    # Load label names if saving images
    label_to_name = {}
    if (
        save_images
        and label_names_file
        and os.path.exists(os.path.join(data_folder, label_names_file))
    ):
        try:
            label_name_df = pd.read_csv(os.path.join(data_folder, label_names_file))
            label_to_name = {
                str(row["Value"]): row["Name"]
                for index, row in label_name_df.iterrows()
            }
            print(f"Loaded {len(label_to_name)} label names")
        except Exception as e:
            print(f"Error loading label names: {e}")

    # Create folders for images if saving
    if save_images:
        if clear_intermediate:
            if os.path.exists(os.path.join(data_folder, labeled_folder)):
                shutil.rmtree(os.path.join(data_folder, labeled_folder))
            if os.path.exists(os.path.join(data_folder, unlabeled_folder)):
                shutil.rmtree(os.path.join(data_folder, unlabeled_folder))
        # Ensure folders exist but do not clear unless requested
        os.makedirs(os.path.join(data_folder, labeled_folder), exist_ok=True)
        os.makedirs(os.path.join(data_folder, unlabeled_folder), exist_ok=True)

    # First, get total count and file list
    total_plaques, files = get_total_plaques_count(data_folder, unlabeled_folder)
    print(f"Total plaques available: {total_plaques}")
    print(f"Unlabeled sample size: {unlabeled_sample_size}")

    rows = []

    # Resume support: load existing CSV if present and not clearing
    processed_pairs = set()
    existing_unlabeled_count = 0
    existing_labeled_count = 0
    file_exists_pre = False
    if (
        output_path is not None
        and not clear_intermediate
        and os.path.exists(output_path)
    ):
        try:
            existing_df = pd.read_csv(output_path)
            for _, r in existing_df.iterrows():
                processed_pairs.add((str(r["Image"]), str(r["Index"]), str(r["Label"])))
            # Count existing unlabeled (Label is NaN) and labeled (Label not NaN)
            if "Label" in existing_df.columns:
                existing_unlabeled_count = existing_df["Label"].isna().sum()
                existing_labeled_count = existing_df["Label"].notna().sum()
            file_exists_pre = True
            print(
                f"Resuming from existing CSV with {len(existing_df)} rows: {existing_labeled_count} labeled, {existing_unlabeled_count} unlabeled"
            )
        except Exception as e:
            print(f"Warning: could not read existing CSV for resume: {e}")

    # Helper: checkpoint appender
    def append_rows_if_needed(force=False):
        nonlocal rows, file_exists_pre
        if output_path is None:
            return
        if not rows:
            return
        if (len(rows) >= checkpoint_every) or force:
            write_header = not os.path.exists(output_path) or (
                clear_intermediate
                and not file_exists_pre
                and not os.path.exists(output_path)
            )
            try:
                df_chunk = pd.DataFrame(
                    rows, columns=["Image", "Index", "Roundness", "Area", "Label"]
                )
                df_chunk.to_csv(
                    output_path,
                    index=False,
                    mode="a",
                    header=(not file_exists_pre and write_header),
                )
                # After first write, we should not write headers again
                file_exists_pre = True
                rows = []
            except Exception as e:
                print(f"Error writing checkpoint to CSV: {e}")

    # Loop 1: Add all labeled samples from npz file
    print("Loop 1: Adding labeled samples from npz file...")
    try:
        with np.load(os.path.join(data_folder, label_file)) as f:
            for image, index, label in tqdm(
                zip(f["file_name"], f["local_idx"], f["label"]),
                total=len(f["file_name"]),
                desc="Processing labeled samples",
            ):
                file_path = os.path.join(data_folder, labeled_folder, image)
                try:
                    # Skip if already processed in previous run
                    if (str(image), str(index), str(label)) in processed_pairs:
                        continue
                    with h5py.File(file_path, "r") as h5f:
                        plaque = h5f["plaques"][str(index)]

                        # Add to data table
                        rows.append(
                            [
                                image,
                                str(index),
                                plaque.attrs["roundness"],
                                plaque.attrs["area"],
                                (
                                    label_to_name[str(int(label))]
                                    if not pd.isna(label)
                                    and str(int(label)) in label_to_name
                                    else ""
                                ),
                            ]
                        )
                        # Checkpoint periodically
                        append_rows_if_needed()

                        # Save image if requested
                        if save_images:
                            # Determine folder and filename
                            if str(int(label)) in label_to_name:
                                class_name = label_to_name[str(int(label))]
                            else:
                                class_name = f"class_{int(label)}"

                            image_folder_path = os.path.join(
                                data_folder, labeled_result_folder, class_name
                            )
                            if not os.path.exists(image_folder_path):
                                os.makedirs(image_folder_path)

                            image_filename = (
                                f'{image.replace(".hdf5", "")}_index_{index}.png'
                            )
                            image_file_path = os.path.join(
                                image_folder_path, image_filename
                            )

                            # Save image if it doesn't exist
                            if not os.path.exists(image_file_path):
                                plaque_image = plaque["plaque"][:]
                                img = Image.fromarray(plaque_image)
                                # img_resized = img.resize(downsample_size)
                                # img_resized.save(image_file_path)
                                img.save(image_file_path)

                except Exception as e:
                    print(
                        f"Error processing labeled sample {image}, index {index}: {e}"
                    )
    except Exception as e:
        print(f"Error loading npz file: {e}")

    # Count labeled including existing
    labeled_count = (existing_labeled_count if existing_labeled_count else 0) + len(
        [1 for r in rows if r[4] != None and r[4] != "" and not pd.isna(r[4])]
    )
    print(f"Added {labeled_count} labeled samples")

    # Loop 2: Sample unlabeled data
    if unlabeled_sample_size > 0:
        print("Loop 2: Sampling unlabeled data...")

        # Extend processed pairs with newly added in this run
        processed_pairs.update((row[0], row[1], row[4]) for row in rows)

        # Create list of all available unlabeled pairs
        unlabeled_pairs = []
        for image in tqdm(files, desc="Collecting unlabeled pairs"):
            file_path = os.path.join(data_folder, unlabeled_folder, image)
            try:
                with h5py.File(file_path, "r") as f:
                    length = f["plaques"].attrs["length"]
                    for index in range(length):
                        if (image, str(index), str(None)) not in processed_pairs:
                            unlabeled_pairs.append((image, str(index)))
            except Exception as e:
                print(f"Error collecting unlabeled pairs from {image}: {e}")

        print(f"Available unlabeled pairs: {len(unlabeled_pairs)}")

        # Determine how many unlabeled are still needed considering resume
        remaining_needed = max(0, unlabeled_sample_size - existing_unlabeled_count)
        if remaining_needed == 0:
            print(
                "No additional unlabeled samples needed; target already met in previous runs."
            )
            sampled_unlabeled = []
        else:
            # Sample from unlabeled pairs
            if remaining_needed > len(unlabeled_pairs):
                print(
                    f"Warning: Need {remaining_needed} unlabeled samples but only {len(unlabeled_pairs)} available"
                )
                sampled_unlabeled = unlabeled_pairs
            else:
                sampled_unlabeled = random.sample(unlabeled_pairs, remaining_needed)

        # Add sampled unlabeled data
        for image, index in tqdm(
            sampled_unlabeled, desc="Processing unlabeled samples"
        ):
            file_path = os.path.join(data_folder, unlabeled_folder, image)
            try:
                with h5py.File(file_path, "r") as f:
                    plaque = f["plaques"][index]

                    # Add to data table
                    rows.append(
                        [
                            image,
                            index,
                            plaque.attrs["roundness"],
                            plaque.attrs["area"],
                            None,  # Unlabeled
                        ]
                    )
                    # Checkpoint periodically
                    append_rows_if_needed()

                    # Save image if requested
                    if save_images:
                        image_filename = (
                            f'{image.replace(".hdf5", "")}_index_{index}.png'
                        )
                        image_file_path = os.path.join(
                            data_folder, unlabeled_result_folder, image_filename
                        )

                        # Save image if it doesn't exist
                        if not os.path.exists(image_file_path):
                            plaque_image = plaque["plaque"][:]
                            img = Image.fromarray(plaque_image)
                            img.save(image_file_path)

            except Exception as e:
                print(f"Error processing unlabeled sample {image}, index {index}: {e}")

    # Final flush of any remaining rows
    append_rows_if_needed(force=True)

    # If we resumed, return only the rows processed in this run, not the full dataset
    print(
        f"Final collection: {len(rows)} plaques (labeled: {labeled_count}, unlabeled: {len(rows) - labeled_count})"
    )
    if output_path is not None and os.path.exists(output_path):
        try:
            final_df = pd.read_csv(output_path)
            print(f"Total dataset size on disk: {len(final_df)} (existing + new)")
        except Exception:
            pass

    if save_images:
        print(f"Images saved to:")
        print(f"  Labeled: {data_folder}/{labeled_result_folder}")
        print(f"  Unlabeled: {data_folder}/{unlabeled_result_folder}")

    return rows


def main():
    args = parse_arguments()
    DATA_FOLDER = args.data_folder
    LABELED_PLAQUES_FOLDER = args.labeled_plaques_folder
    UNLABELED_PLAQUES_FOLDER = args.unlabeled_plaques_folder
    UNLABELED_SAMPLE_SIZE = args.unlabeled_sample_size
    RANDOM_SEED = args.random_seed
    OUTPUT_SAMPLED_DATA_TABLE_FILE = args.output_sampled_data_table_file
    LABEL_FILE = args.label_file
    LABEL_NAMES_FILE = args.label_names_file
    LABELED_RESULT_FOLDER = args.labeled_result_folder
    UNLABELED_RESULT_FOLDER = args.unlabeled_result_folder
    SAVE_IMAGES = args.save_images
    CLEAR_INTERMEDIATE = args.clear_intermediate
    CHECKPOINT_EVERY = args.checkpoint_every

    print(f"Starting sampled data table generation...")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Labeled plaques folder: {LABELED_PLAQUES_FOLDER}")
    print(f"Unlabeled plaques folder: {UNLABELED_PLAQUES_FOLDER}")
    print(f"Unlabeled sample size: {UNLABELED_SAMPLE_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output sampled data table file: {OUTPUT_SAMPLED_DATA_TABLE_FILE}")
    print(f"Label names file: {LABEL_NAMES_FILE}")
    print(f"Labeled result folder: {DATA_FOLDER}/{LABELED_RESULT_FOLDER}")
    print(f"Unlabeled result folder: {DATA_FOLDER}/{UNLABELED_RESULT_FOLDER}")
    print(f"Save images: {SAVE_IMAGES}")
    print(f"Clear intermediate: {CLEAR_INTERMEDIATE}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY}")

    output_path = os.path.join(DATA_FOLDER, OUTPUT_SAMPLED_DATA_TABLE_FILE)
    # Clear previous CSV if requested
    if CLEAR_INTERMEDIATE and os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Cleared previous CSV at {output_path}")
        except Exception as e:
            print(f"Warning: could not remove existing CSV: {e}")

    # Create data table with labeled and unlabeled samples (and save images if requested)
    rows = create_data_table(
        DATA_FOLDER,
        LABELED_PLAQUES_FOLDER,
        UNLABELED_PLAQUES_FOLDER,
        UNLABELED_SAMPLE_SIZE,
        LABEL_FILE,
        RANDOM_SEED,
        save_images=SAVE_IMAGES,
        labeled_result_folder=LABELED_RESULT_FOLDER,
        unlabeled_result_folder=UNLABELED_RESULT_FOLDER,
        label_names_file=LABEL_NAMES_FILE,
        output_path=output_path,
        clear_intermediate=CLEAR_INTERMEDIATE,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    # If nothing new processed and no csv exists, exit
    if (rows is None or len(rows) == 0) and not os.path.exists(output_path):
        print("No plaques were collected. Exiting.")
        return

    # If CSV exists, print some stats from disk
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            print(f"\nSampled data table saved to: {output_path}")
            print(f"Final dataset shape on disk: {df.shape}")
            print(f"Sample of the data:")
            print(df.head())

            # Print some statistics
            print(f"\nStatistics:")
            print(f"Total plaques sampled: {len(df)}")
            print(f"Unique images: {df['Image'].nunique()}")
            print(f"Plaques with labels: {df['Label'].notna().sum()}")
            if df["Label"].notna().sum() > 0:
                print(f"Label distribution:")
                print(df["Label"].value_counts())
        except Exception as e:
            print(f"Warning: could not load final CSV for stats: {e}")


if __name__ == "__main__":
    main()
