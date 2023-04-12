import torch
from hvit.decision_block.filter_block import ImagePatchFilter
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.image_utils import load_jpg_image, load_png_image

def main(mode='cifar10', probabilistic=True, heuristic='contrast'):
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        # Resize the image to (64, 64)
        transforms.Resize((64, 64)),
        # Convert the image to a PyTorch tensor
        transforms.ToTensor(),
        # Normalize the image with mean and standard deviation of 0.5
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the ImagePatchFilter class
    filter = ImagePatchFilter(patch_size=16, top_k=128, heuristic=heuristic, probabilistic=probabilistic, prob=1, verbose=True)

    if mode == 'cifar10':
        # Load the CIFAR10 dataset
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Choose 5 random indices from the dataset
        rand_indices = torch.randint(0, len(dataset), (5,))

        # Get the corresponding images and labels
        img = []
        labels = []
        for i in range(5):
            img_i, label_i = dataset[rand_indices[i]]
            img.append(img_i)
            labels.append(label_i)
        img = torch.stack(img, dim=0)

    if mode == 'static':
        # Alternatively, use a static test image
        img = torch.from_numpy(load_jpg_image('data/images/dog2.jpg')).permute(2,0,1)
        img = torch.unsqueeze(img, dim=0).repeat(1, 1, 1, 1)


    # Apply the ImagePatchFilter to the images
    filtered_image = filter(img)

    # Display the original and filtered images for each of the 5 randomly chosen images
    for i in range(filtered_image.shape[0]):
        # Define the figure and axis objects
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the original image on the first axis
        axs[0].imshow(torchvision.utils.make_grid(img[i], nrow=1).permute(1, 2, 0))
        axs[0].set_title("Original Image")

        # Plot the filtered image on the second axis
        axs[1].imshow(torchvision.utils.make_grid(filtered_image[i], nrow=1).permute(1, 2, 0))
        axs[1].set_title("Filtered Image")

        # Display the plot
        plt.show()

if __name__ == '__main__':
    main(mode='static', probabilistic=False, heuristic='contrast')