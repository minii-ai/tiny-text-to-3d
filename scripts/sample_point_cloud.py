import argparse
import os
import queue as q
from multiprocessing import JoinableQueue, Process

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True, default=1024)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)

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


def worker(queue: JoinableQueue, n_samples: int, output_dir: str):
    while True:
        try:
            mesh_path = queue.get()
            mesh = trimesh.load(mesh_path, force="mesh")
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

    # initialize workers
    processes = []
    for i in range(args.num_workers):
        process = Process(target=worker, args=(queue, args.n_samples, args.output_dir))
        process.daemon = True
        processes.append(process)

    # start workers
    for process in processes:
        process.start()

    # block until all tasks are done
    queue.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
