
import pickle
from torchvision.datasets import UCF101
from argparse import ArgumentParser
import os


def main():

    parser = ArgumentParser(description='precomputing metadata of UCF101')

    # dataset
    parser.add_argument('-r', '--root', type=str,
                        default='/dataset/UCF-101/',
                        help='root of dataset. default to /dataset/UCF-101/')
    parser.add_argument('-a', '--annotation_path', type=str,
                        default='/dataset/ucfTrainTestlist/',
                        help='path to the folder of annotation files.'
                        'default to /dataset/ucfTrainTestlist')
    parser.add_argument('-fpc', '--frames_per_clip', type=int, default=8,
                        help='frames per clip. default to 8')
    parser.add_argument('-sbc', '--step_between_clips', type=int, default=8,
                        help='step between clips. default to 8')
    parser.add_argument('-w', '--num_workers', type=int, default=6,
                        help='number of workers. default to 6')

    parser.add_argument('-o', '--metadata_path', type=str,
                        default='/dataset/',
                        help='path to the folder in which the metadata file is stored.'
                        'default to /dataset')

    args = parser.parse_args()
    print(args)

    dataset_dict = UCF101(root=args.root,
                          annotation_path=args.annotation_path,
                          frames_per_clip=args.frames_per_clip,
                          step_between_clips=args.step_between_clips,
                          num_workers=args.num_workers,
                          )
    filename = os.path.join(
        args.metadata_path,
        'UCF101metadata_fpc{}_sbc{}.pickle'.format(args.frames_per_clip,
                                                   args.step_between_clips))
    with open(filename, "wb") as f:
        pickle.dump(dataset_dict.metadata, f)


if __name__ == "__main__":
    main()
