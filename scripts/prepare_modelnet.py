# Sample low resolution point clouds
# Sample high resolution point clouds


import argparse
import os
import queue as q
import shutil
from multiprocessing import JoinableQueue, Process

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--n_low_res_samples", type=int, default=1024)
    parser.add_argument("--n_high_res_samples", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=20)

    return parser.parse_args()


def get_mesh_paths(root_dir: str) -> list[str]:
    mesh_paths = []

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        is_dir = os.path.isdir(label_path)

        if not is_dir:
            continue

        for split in os.listdir(label_path):
            if split == ".DS_Store":
                continue

            split_path = os.path.join(label_path, split)
            for mesh_name in os.listdir(split_path):
                if mesh_name == ".DS_Store":
                    continue

                mesh_path = os.path.join(split_path, mesh_name)
                mesh_paths.append(mesh_path)

    return mesh_paths


def worker(args):
    queue, n_low_res_samples, n_high_res_samples, low_res_dir, high_res_dir = (
        args["queue"],
        args["n_low_res_samples"],
        args["n_high_res_samples"],
        args["low_res_dir"],
        args["high_res_dir"],
    )
    while True:
        try:
            mesh_path = queue.get()
            mesh = trimesh.load(mesh_path, force="mesh")

            todos = zip(
                [n_low_res_samples, n_high_res_samples],
                [low_res_dir, high_res_dir],
            )

            for n_samples, output_dir in todos:
                samples = trimesh.sample.sample_surface(mesh, n_samples)[0]

                # save samples as .npy
                segments = mesh_path.split("/")
                label, split, mesh_name = segments[-3], segments[-2], segments[-1]

                # create output dir
                save_dir = os.path.join(output_dir, label, split)
                os.makedirs(save_dir, exist_ok=True)

                # save samples
                save_path = os.path.join(save_dir, mesh_name.replace(".off", ".npy"))
                np.save(save_path, samples)

            print(f"Processed {mesh_path}")
        except q.Empty:
            break
        except Exception as e:
            queue.put(mesh_path)
        finally:
            queue.task_done()


def main(args):
    mesh_paths = get_mesh_paths(args.root_dir)
    queue = JoinableQueue()

    for mesh_path in mesh_paths:
        queue.put(mesh_path)

    # create low res and high res dir
    low_res_dir = os.path.join(args.output_dir, "low_res")
    high_res_dir = os.path.join(args.output_dir, "high_res")

    os.makedirs(low_res_dir, exist_ok=True)
    os.makedirs(high_res_dir, exist_ok=True)

    # initialize workers
    processes = []
    for i in range(args.num_workers):
        worker_args = {
            "queue": queue,
            "n_low_res_samples": args.n_low_res_samples,
            "n_high_res_samples": args.n_high_res_samples,
            "low_res_dir": low_res_dir,
            "high_res_dir": high_res_dir,
        }
        process = Process(target=worker, args=(worker_args,))
        process.daemon = True
        processes.append(process)

    # start workers
    for process in processes:
        process.start()

    # block until all tasks are done
    queue.join()

    # rename root dir to off and move into output_dir
    os.rename(args.root_dir, "off")
    shutil.move("./off", args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
