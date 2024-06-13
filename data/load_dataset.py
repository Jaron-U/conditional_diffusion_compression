from .misc_data_util import transforms as trans
import matplotlib.pyplot as plt
import os

def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), "Data path {} not found.".format(data_path)

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    train = val = None

    if dataset_name == "vimeo":
        from .datasets.vimeo import VIMEO
        transforms = [
                trans.RandomCrop(256, False),
                trans.RandomSequenceCrop(1),
                trans.ImageToTensor(),
                trans.ConcatSequence(),
                #   torchvision.transforms.RandomCrop(data_config['img_size'])
            ]
        transforms = trans.Compose(transforms)
        train = VIMEO(os.path.join(data_path, "vimeo_data"), train=True, transform=transforms, add_noise=False)
        val = VIMEO(os.path.join(data_path, "vimeo_data"), train=False, transform=transforms, add_noise=False)

    return train, val
# if __name__ == '__main__':
#     from datasets.vimeo import VIMEO
#     transforms = [
#             trans.RandomCrop(256, False),
#             trans.RandomSequenceCrop(1),
#             trans.ImageToTensor(),
#             trans.ConcatSequence(),
#             #   torchvision.transforms.RandomCrop(data_config['img_size'])
#         ]
#     transforms = trans.Compose(transforms)
#     dataset = VIMEO('/home/jianglongyu/mydrive/vimeo_data/', train=True, transform=transforms, add_noise=False)
#     print(len(dataset))

#     dataloder = DataLoader(dataset, batch_size=4, shuffle=True)
#     for i, imgs in enumerate(dataloder):
#         print(f'Batch {i} - Images shape: {imgs.shape}')
#         # Plot the first image of the batch
#         img = imgs[0][0].numpy().transpose(1, 2, 0)
#         plt.imshow(img)
#         plt.savefig(f'batch_{i}.png')
#         if i == 2:  # Just show the first 3 batches
#             break
    