import torch
import torch.nn as nn


class PatchClassifierCNN(nn.Module):
    """
    Small CNN for classifying 32x32 image patches into paddy / non_paddy / water.
    Works well with very small labeled datasets (few-shot regime).
    """

    def __init__(self, in_channels=7, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (64, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def build_unet_segmentation_model(in_channels=7, num_classes=3):
    """
    Builds a full U-Net segmentation model using segmentation_models_pytorch.
    This is ready to use once real, larger-scale pixel-labeled masks are available.
    Not trained yet in this task -- wired up for future use.
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,  # no ImageNet weights since we have 7 non-RGB channels
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


if __name__ == "__main__":
    model = PatchClassifierCNN(in_channels=7, num_classes=3)
    dummy_input = torch.randn(4, 7, 32, 32)
    output = model(dummy_input)
    print("PatchClassifierCNN output shape:", output.shape)
    print("Expected: torch.Size([4, 3])")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    unet = build_unet_segmentation_model(in_channels=7, num_classes=3)
    dummy_image = torch.randn(1, 7, 64, 64)
    unet_output = unet(dummy_image)
    print("\nU-Net output shape:", unet_output.shape)
    print("Expected: torch.Size([1, 3, 64, 64])")