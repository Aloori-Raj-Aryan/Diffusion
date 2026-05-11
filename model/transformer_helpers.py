def image_to_patches(image, patch_size):
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size."

    # Calculate the number of patches along height and width
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Reshape and permute to get patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size * patch_size).permute(1, 0, 2).contiguous()
    
    return patches.view(-1, C * patch_size * patch_size)


def patches_to_image(patches, image_channels):
    num_patches, patch_dim = patches.shape
    patch_size = int((patch_dim // image_channels) ** 0.5)
    assert patch_size * patch_size * image_channels == patch_dim, "Patch dimension must be divisible by the number of channels."

    # Calculate the number of patches along height and width
    num_patches_h = int((num_patches ** 0.5))
    num_patches_w = num_patches_h

    # Reshape and permute to reconstruct the image
    patches = patches.view(num_patches, image_channels, patch_size * patch_size).permute(1, 0, 2).contiguous()
    patches = patches.view(image_channels, num_patches_h, num_patches_w, patch_size, patch_size)
    
    return patches.permute(0, 1, 3, 2, 4).contiguous().view(image_channels, num_patches_h * patch_size, num_patches_w * patch_size)