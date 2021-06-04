
import pickle
from torchvision.datasets import UCF101


def main():
    fpc = 1
    sbc = 1
    dataset_dict = UCF101(root='/dataset/UCF-101/',
                          annotation_path='/dataset/ucfTrainTestlist/',
                          frames_per_clip=fpc,
                          step_between_clips=sbc,
                          num_workers=6,  # more is better
                          )
    filename = '/dataset/UCF101metadata_fpc{}_sbc{}.pickle'.format(fpc, sbc)
    with open(filename, "wb") as f:
        pickle.dump(dataset_dict.metadata, f)


if __name__ == "__main__":
    main()
