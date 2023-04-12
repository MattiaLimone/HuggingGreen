import torch
import torch.nn as nn


class PatchConv(nn.Module):
    def __init__(self, num_channels, patch_size):
        super(PatchConv, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, num_patches, num_channels, patch_height, patch_width = x.shape
        output = torch.zeros(batch_size, num_patches)
        for i in range(batch_size):
            for j in range(num_patches):
                output[i, j] = self.conv(x[i, j, :, :, :])

        return output


class ImagePatchFilter(nn.Module):
    def __init__(self, patch_size: int = 8, top_k: int = 8, heuristic: str = 'contrast', probabilistic: bool = True,
                 prob: float = 0.5, decay_rate: float = 0.0, batch_size: int = 0, verbose: bool = False):
        """
        The ImagePatchFilter class takes in a patch size, top_k, and probabilistic and returns a super class of the
        ImagePatchFilter class

        :param patch_size: The size of the patch to be extracted from the image, defaults to 8 (optional)
        :param top_k: The number of patches to select, defaults to 8 (optional)
        :param probabilistic: If True, the filter will return a probability distribution over the patches. If False, it will
        return the top k patches, defaults to True (optional)
        """
        super(ImagePatchFilter, self).__init__()
        # Set the path prob
        self.counter = 0
        # Set the prob
        self.prob = prob
        # Set the batch size
        self.batch_size = batch_size
        # Set the decay rate
        self.decay_rate = decay_rate
        # Store the current epoch variable
        self.current_epoch = 0
        # Set the patch size
        self.patch_size = patch_size
        # Set the top k
        self.top_k = top_k
        # Set the probabilistic flag
        self.probabilistic = probabilistic
        # Set the verbose flag
        self.verbose = verbose
        # Set the heuristic type
        self.heuristic = heuristic

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def divide_in_patches(self, image):
        # Get the shape of the image tensor
        batch_size, num_channels, height, width = image.size()

        # Calculate the number of patches in the height and width dimensions
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size

        # Reshape the image tensor to divide it into patches
        patches = image.view(batch_size, num_channels, num_patches_height, self.patch_size, num_patches_width,
                             self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(batch_size, num_patches_height * num_patches_width, num_channels, self.patch_size,
                               self.patch_size)

        return patches

    def get_distribution_vector(self, patches, mode='contrast'):
        patches = patches.float()

        if mode == 'contrast':
            max_value, _ = torch.max(patches, dim=3)
            min_value, _ = torch.min(patches, dim=3)
            max_value, _ = torch.max(max_value, dim=3)
            min_value, _ = torch.min(min_value, dim=3)
            max_value, _ = torch.max(max_value, dim=2)
            min_value, _ = torch.min(min_value, dim=2)
            contrast_values = (max_value - min_value + 1e-8) / (max_value + min_value)

            return contrast_values

        if mode == 'entropy':
            entropy_values = torch.special.entr(patches + 1e-8).mean(dim=[2, 3, 4])

            return entropy_values

        if mode == 'variance':
            variance_values = patches.var(dim=[3, 4]).mean(dim=2)

            return variance_values

        if mode == 'conv':
            conv_operator = PatchConv(num_channels=patches.shape[2], patch_size=patches.shape[3])
            conv_values = conv_operator(patches)

            return conv_values

    def get_topk_patches_mask(self, patches, distribution_values):
        """
        For each image in the batch, we get the top k patches based on the contrast values, and set the rest of the
        patches to zero

        :param patches: the patches extracted from the image
        :param distribution_values: a tensor of shape (batch_size, num_patches)
        :return: The patches that have the highest contrast values.
        """
        # Get the shape of the patches tensor
        batch_size, num_patches, _, _, _ = patches.shape

        # Initialize a mask tensor to zero
        mask = torch.zeros_like(patches)

        # Loop over the patches and select the top k patches
        for i in range(batch_size):
            # Get the top k patches
            if self.probabilistic:
                # Sample patches from the probability distribution described by the contrast values
                probs = (distribution_values[i, :] - distribution_values[i, :].min()) / \
                        (distribution_values[i, :].max() - distribution_values[i, :].min())
                probs = torch.nan_to_num(probs, nan=1e-6, posinf=1e-6, neginf=1e-6)
                topk_patches_indices = torch.multinomial(probs, num_samples=self.top_k, replacement=False)
            else:
                # Select the top-k patches based on their contrast values
                topk_patches_indices = torch.topk(distribution_values[i, :],
                                                  self.top_k, largest=True, sorted=True).indices
            # Set the mask to 1 for the top k patches
            mask[i, topk_patches_indices, :, :, :] = 1

        # Apply the mask to the patches tensor
        masked_patches = patches * mask

        return masked_patches

    def update_epoch(self):
        self.current_epoch = self.current_epoch + 1

    def update_branch_prob(self, epoch):
        if self.training:
            self.prob = self.prob / (1 + self.decay_rate * epoch)

    def random_binary(self, probability):
        """
        Returns 0 or 1 with probability `probability`.
        """
        distribution = torch.distributions.Bernoulli(probability)
        binary_value = distribution.sample()
        return binary_value.int()

    def forward(self, images):
        if not self.training:
            return images

        if self.counter == 0:
            print("\nCurrent Epoch: ", self.current_epoch)
            print("Probability: ", self.prob)
        else:
            if self.counter % self.batch_size == 0:
                self.update_epoch()
                self.update_branch_prob(self.current_epoch)
                print("\nCurrent Epoch: ", self.current_epoch)
                print("Probability: ", self.prob)
                self.counter = 0

        self.counter += 1
        flag = self.random_binary(self.prob)
        if flag:
            # Divide the input image into patches
            patches = self.divide_in_patches(images)
            # Calculate the contrast value for each patch
            contrast_values = self.get_distribution_vector(patches, self.heuristic)
            # Create a mask to set the non-selected patches to zero
            mask = self.get_topk_patches_mask(patches, contrast_values)
            return mask
            '''
            # Reshape the mask tensor to match the input image tensor shape
            batch_size, num_patches, num_channels, patch_height, patch_width = mask.size()
            height = num_patches // (images.size(-1) // self.patch_size)
            width = num_patches // (images.size(-2) // self.patch_size)
            mask = mask.view(batch_size, height, width, num_channels, patch_height, patch_width)
            mask = mask.permute(0, 3, 1, 4, 2, 5).contiguous()
            mask = mask.view(batch_size, num_channels, height * patch_height, width * patch_width)
            # Return the filtered image
            return mask
            '''
        else:
            # Return the unfiltered image
            return self.divide_in_patches(images)
