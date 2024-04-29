import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True, default=1024)

    return parser.parse_args()


def main(args):
    print(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
